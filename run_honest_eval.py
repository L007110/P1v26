import os
import torch
import numpy as np
import pandas as pd
import Parameters
import Main
import GNNModel
from Topology import formulate_global_list_dqn
from Parameters import RL_N_STATES_BASE, V2I_LINK_POSITIONS, USE_UMI_NLOS_MODEL, TRANSMITTDE_POWER, \
    V2V_CHANNEL_BANDWIDTH


# ==========================================
# 🛠️ 辅助函数
# ==========================================
def calculate_shannon_capacity(snr_db, bandwidth_hz):
    """ 香农公式 """
    if snr_db < -100: return 0.0
    snr_linear = 10 ** (snr_db / 10.0)
    return bandwidth_hz * np.log2(1 + snr_linear) / 1e6  # Mbps


# ==========================================
# 🚀 核心评估逻辑
# ==========================================
def run_honest_evaluation(model_path, target_density_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动多维诚实评估 (Mode: Inference Only)")
    print(f"📂 模型: {model_path}")

    # === 1. 自动判断模式 ===
    IS_NO_GNN_MODE = "NoGNN" in model_path or "ye" in model_path.lower()

    if IS_NO_GNN_MODE:
        print("ℹ️  检测到 No-GNN 模型 (Ye et al. / Pure DRL)")
        Parameters.USE_GNN_ENHANCEMENT = False
        Parameters.GNN_ARCH = "NONE"
    else:
        print("ℹ️  检测到 GNN 模型 (Proposed / GCN)")
        Parameters.USE_GNN_ENHANCEMENT = True
        if "GCN" in model_path:
            Parameters.GNN_ARCH = "GCN"
        else:
            Parameters.GNN_ARCH = "HYBRID"

    Parameters.RUN_MODE = "TEST"
    Parameters.SCENE_SCALE_X = 1200
    Parameters.SCENE_SCALE_Y = 1200
    SYSTEM_BANDWIDTH = 400e6

    # === 2. 初始化环境 ===
    formulate_global_list_dqn(Parameters.global_dqn_list, device)
    channel_model = Main.new_reward_calculator.channel_model

    # === 3. 加载模型 ===
    checkpoint = torch.load(model_path, map_location=device)
    gnn_model = None

    if IS_NO_GNN_MODE:
        # 【No-GNN 模式】加载 DQN 权重到每个智能体
        print("📥 正在加载分布式 DQN 权重...")
        for dqn in Parameters.global_dqn_list:
            key = f"dqn_{dqn.dqn_id}"
            if key in checkpoint:
                # [FIX 1] dqn 本身就是 Module，直接加载
                dqn.load_state_dict(checkpoint[key])
                dqn.eval()
            else:
                print(f"⚠️ 警告: 没找到 {key} 的权重！")
    else:
        # 【GNN 模式】加载 GNN 全局权重
        print(f"📥 正在加载全局 GNN ({Parameters.GNN_ARCH}) 权重...")
        gnn_model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64).to(device)
        try:
            gnn_model.load_state_dict(checkpoint)
        except:
            print("⚠️ 标准加载失败，尝试 strict=False...")
            gnn_model.load_state_dict(checkpoint, strict=False)
        gnn_model.eval()
        GNNModel.global_gnn_model = gnn_model

    results = []

    # === 4. 循环测试 N ===
    for n in target_density_list:
        print(f"\n⚡ 测试密度 N={n} ...")
        Parameters.TRAINING_VEHICLE_TARGET = n
        Parameters.NUM_VEHICLES = n

        # 重置状态
        for dqn in Parameters.global_dqn_list:
            dqn.curr_state = []
            dqn.delay_list = []
            dqn.snr_list = []
            dqn.v2v_success_list = []
            dqn.feasible_list = []
            dqn.prev_v2i_interference = 0.0
            dqn.epsilon = 0.0

            # 预热
        from Topology import vehicle_movement
        vid = 0
        vlist = []
        for _ in range(500):
            vid, vlist = vehicle_movement(vid, vlist, target_count=n)

        history = {
            "S": [], "F": [], "SNR": [], "Link_Cap": [], "V2V_Sum": [], "V2I_Sum": []
        }

        test_steps = 200

        for step in range(test_steps):
            # A. 移动
            vid, vlist = vehicle_movement(vid, vlist, target_count=n)

            # B. 状态更新与构建 (关键：为 No-GNN 模式准备 state)
            for dqn in Parameters.global_dqn_list:
                dqn.vehicle_exist_curr = False
                dqn.vehicle_in_dqn_range_by_distance = []
                for vehicle in vlist:
                    if (dqn.start[0] <= vehicle.curr_loc[0] <= dqn.end[0] and
                            dqn.start[1] <= vehicle.curr_loc[1] <= dqn.end[1]):
                        dqn.vehicle_exist_curr = True
                        vehicle.distance_to_bs = channel_model.calculate_3d_distance(
                            (dqn.bs_loc[0], dqn.bs_loc[1]), vehicle.curr_loc)
                        dqn.vehicle_in_dqn_range_by_distance.append(vehicle)

                # 排序
                if dqn.vehicle_exist_curr:
                    dqn.vehicle_in_dqn_range_by_distance.sort(key=lambda x: x.distance_to_bs)

                    # [FIX 3] 构建 curr_state (对于 No-GNN 是必须的)
                    # 1. Base State
                    base_state = []
                    for iVehicle in range(min(RL_N_STATES_BASE // 4, len(dqn.vehicle_in_dqn_range_by_distance))):
                        v = dqn.vehicle_in_dqn_range_by_distance[iVehicle]
                        base_state.extend([v.curr_loc[0], v.curr_loc[1], v.curr_dir[0], v.curr_dir[1]])
                    if len(base_state) < RL_N_STATES_BASE:
                        base_state.extend([0.0] * (RL_N_STATES_BASE - len(base_state)))
                    else:
                        base_state = base_state[:RL_N_STATES_BASE]

                    # 2. CSI State
                    if USE_UMI_NLOS_MODEL:
                        dqn.update_csi_states(dqn.vehicle_in_dqn_range_by_distance, is_current=True)

                    # 3. V2I History
                    interf_log = np.log10(dqn.prev_v2i_interference + 1e-20)
                    interf_norm = (interf_log + 20) / 14.0

                    dir_x, dir_y = 0.0, 0.0
                    if V2I_LINK_POSITIONS and dqn.vehicle_in_dqn_range_by_distance:
                        target_rx = V2I_LINK_POSITIONS[0]['rx']
                        curr_pos = dqn.vehicle_in_dqn_range_by_distance[0].curr_loc
                        dx, dy = target_rx[0] - curr_pos[0], target_rx[1] - curr_pos[1]
                        d = np.sqrt(dx ** 2 + dy ** 2) + 1e-9
                        dir_x, dir_y = dx / d, dy / d

                    v2i_state = [interf_norm, dir_x, dir_y]

                    # 赋值给 curr_state
                    dqn.curr_state = base_state + dqn.csi_states_curr + v2i_state
                else:
                    dqn.curr_state = []

            # C. 推理与动作选择

            # 重置功率
            for v in vlist: v.power_W = 0.0

            if IS_NO_GNN_MODE:
                # === 分支 1: Ye et al. (No-GNN) ===
                with torch.no_grad():
                    for dqn in Parameters.global_dqn_list:
                        if dqn.vehicle_exist_curr and len(dqn.curr_state) > 0:
                            state_tensor = torch.FloatTensor(dqn.curr_state).unsqueeze(0).to(device)
                            # [FIX 2] 直接调用 dqn()，不是 dqn.policy_net()
                            q_values = dqn(state_tensor)
                            action_idx = q_values.argmax().item()
                            dqn.action = Parameters.RL_ACTION_SPACE[action_idx]

                            _apply_action_power(dqn)
                        else:
                            dqn.action = None

            else:
                # === 分支 2: GNN Mode ===
                graph = Main.global_graph_builder.build_dynamic_graph(Parameters.global_dqn_list, vlist, step)
                graph = Main.move_graph_to_device(graph, device)

                with torch.no_grad():
                    q_values_all, _ = gnn_model(graph)

                    for dqn in Parameters.global_dqn_list:
                        if dqn.vehicle_exist_curr:
                            dqn_idx = dqn.dqn_id - 1
                            action_idx = q_values_all[dqn_idx].argmax().item()
                            dqn.action = Parameters.RL_ACTION_SPACE[action_idx]
                            _apply_action_power(dqn)
                        else:
                            dqn.action = None

            # D. 物理层计算
            _calculate_physics_metrics(Parameters.global_dqn_list, vlist, channel_model, history, SYSTEM_BANDWIDTH)

        # 统计结果
        _print_and_save_results(n, history, results)

    df = pd.DataFrame(results)
    csv_name = "results_NoGNN.csv" if IS_NO_GNN_MODE else "results_GNN_Test.csv"
    df.to_csv(csv_name, index=False)
    print(f"\n✅ 数据已保存: {csv_name}")


# ==========================================
# 🔌 内部工具函数
# ==========================================
def _apply_action_power(dqn):
    """将 Action 翻译为物理功率"""
    beam_count = dqn.action[0] + 1
    power_ratio = (dqn.action[3] + 1) / 10.0
    gain = Main.new_reward_calculator._calculate_directional_gain(dqn.action[1], dqn.action[2])
    pwr = Parameters.TRANSMITTDE_POWER * power_ratio * beam_count * gain * Parameters.GAIN_ANTENNA_T
    if dqn.vehicle_in_dqn_range_by_distance:
        dqn.vehicle_in_dqn_range_by_distance[0].power_W = pwr


def _calculate_physics_metrics(dqn_list, vlist, channel_model, history, bw):
    """统一物理层计算"""
    active_interferers = [{'tx_pos': v.curr_loc, 'power_W': v.power_W} for v in vlist if v.power_W > 0]
    step_v2v = 0.0
    step_v2i = 0.0

    # V2V
    for dqn in dqn_list:
        if dqn.vehicle_exist_curr:
            Main.new_reward_calculator.calculate_complete_reward(
                dqn, dqn.vehicle_in_dqn_range_by_distance, dqn.action, active_interferers
            )
            if dqn.snr_list:
                snr = dqn.snr_list[-1]
                cap = calculate_shannon_capacity(snr, bw)
                history["SNR"].append(snr)
                history["Link_Cap"].append(cap)
                step_v2v += cap
            if dqn.v2v_success_list: history["S"].append(dqn.v2v_success_list[-1])
            if dqn.feasible_list: history["F"].append(dqn.feasible_list[-1])

    # V2I (Real Physics)
    noise_w = channel_model._calculate_noise_power(bw)
    for link in Parameters.V2I_LINK_POSITIONS:
        d_sig = channel_model.calculate_3d_distance(link['tx'], link['rx'])
        _, _, v2i_sig = channel_model.calculate_snr(Parameters.V2I_TX_POWER, d_sig, bandwidth=bw)

        tot_int = 0.0
        for interf in active_interferers:
            d_i = channel_model.calculate_3d_distance(interf['tx_pos'], link['rx'])
            pl_i, _, _ = channel_model.calculate_path_loss(d_i)
            tot_int += interf['power_W'] * (10 ** (-pl_i / 10))

        sinr = v2i_sig / (noise_w + tot_int + 1e-20)
        step_v2i += calculate_shannon_capacity(10 * np.log10(sinr), bw)

    history["V2V_Sum"].append(step_v2v)
    history["V2I_Sum"].append(step_v2i)


def _print_and_save_results(n, history, results):
    avg_v2v = np.mean(history["V2V_Sum"])
    avg_v2i = np.mean(history["V2I_Sum"])
    raw_s = np.mean(history["S"]) if history["S"] else 0
    print(f"   📊 N={n}: SR={raw_s:.2%} | V2V_Sum={avg_v2v:.0f} | V2I_Sum={avg_v2i:.0f}")

    results.append({
        'density': n,
        'raw_success_rate': raw_s,
        'v2v_sum_mbps': avg_v2v,
        'v2i_sum_mbps': avg_v2i,
        'system_sum_rate_mbps': avg_v2v + avg_v2i
    })


if __name__ == "__main__":
    MODEL = "model_NoGNN.pt"
    if not os.path.exists(MODEL):
        print("⚠️ 没找到 NoGNN 模型，请检查文件名是否为 model_NoGNN.pt")
    else:
        SCENARIOS = [20, 40, 60, 80, 100, 120, 140]
        run_honest_evaluation(MODEL, SCENARIOS)