import numpy as np
import pandas as pd
import sys

# === 导入你的项目模块 ===
import Parameters
from Topology import vehicle_movement
from ChannelModel import global_channel_model
from Ashraf_Algorithm import ashraf_solver


# ==========================================
# 🔧 辅助工具
# ==========================================
def get_param(name, default=None):
    if hasattr(Parameters, name):
        return getattr(Parameters, name)
    if name == "TRANSMITTED_POWER" and hasattr(Parameters, "TRANSMITTDE_POWER"):
        return getattr(Parameters, "TRANSMITTDE_POWER")
    if name == "NUM_CHANNELS" and hasattr(Parameters, "NUM_RB"):
        return getattr(Parameters, "NUM_RB")
    if name == "V2I_POWER" and hasattr(Parameters, "V2I_POWER_DBM"):
        return 10 ** ((getattr(Parameters, "V2I_POWER_DBM") - 30) / 10)
    if default is not None:
        return default
    raise ValueError(f"❌ 错误: Parameters 中缺少 '{name}'")


def calculate_shannon_capacity(sinr_linear, bandwidth_hz):
    if sinr_linear <= 0: return 0.0
    return bandwidth_hz * np.log2(1 + sinr_linear) / 1e6  # Mbps


def calculate_noise_power_watts(bandwidth_hz):
    k = global_channel_model.boltzmann_constant
    T = global_channel_model.temperature
    nf_db = get_param("NOISE_FIGURE", default=9.0)
    nf_linear = 10 ** (nf_db / 10.0)
    return k * T * bandwidth_hz * nf_linear


# ==========================================
# 🛑 严格物理层评估 (兼容 SDMA 和 FDMA)
# ==========================================
def evaluate_full_physics(vehicle_list):
    """
    逻辑说明：
    1. 该函数支持 RB 正交性检查。
       - 如果是 DRL (Main.py)，所有车 RB=0 -> 产生全干扰 -> 对齐 Main.py。
       - 如果是 Ashraf，车辆 RB 不同 -> 无干扰 -> 体现 FDMA 优势。
    2. V2I 依然受全网干扰 (因为 V2I 是宽带接收，或者假设复用了所有 RB)。
    """

    # --- 1. 参数准备 ---
    total_bw = get_param("SYSTEM_BANDWIDTH")
    n_rb = get_param("NUM_CHANNELS")
    rb_bw = total_bw / n_rb

    noise_watts_v2i = calculate_noise_power_watts(total_bw)  # V2I 用全带宽噪声
    noise_watts_v2v = calculate_noise_power_watts(rb_bw)  # V2V 用子信道噪声

    v2v_min_snr = get_param("V2V_MIN_SNR_DB", 10.0)

    metrics = {
        "v2v_success": 0, "v2v_capacity": 0, "v2v_links": 0,
        "v2i_success": 0, "v2i_capacity": 0, "v2i_links": 0
    }

    # 预筛选活跃车辆 (功率 > 0)
    active_interferers = [v for v in vehicle_list if v.power_W > 0]

    # =========================================================
    # Part A: V2I 评估
    # =========================================================
    v2i_links = getattr(Parameters, "V2I_LINK_POSITIONS", [])
    metrics["v2i_links"] = len(v2i_links)

    for i, link in enumerate(v2i_links):
        # 1. V2I 信号
        d_sig = global_channel_model.calculate_3d_distance(link['tx'], link['rx'])
        _, _, v2i_sig_watts = global_channel_model.calculate_snr(
            Parameters.V2I_TX_POWER, d_sig, bandwidth=total_bw
        )

        # 2. V2I 干扰 (Main.py 逻辑：累加所有 V2V)
        interf_watts = 0.0

        # 这里我们需要做一个合理的物理假设：
        # 如果 V2I 用户 i 复用 RB i，那么只有 RB i 上的 V2V 才会干扰它。
        # 如果 Main.py 是全干扰，说明 Main.py 假设 V2I 也是受全带宽干扰。
        # 为了严格对齐 Main.py 的“最坏情况”假设，我们这里遍历所有 active_interferers
        # 或者为了体现 Ashraf 的优势，我们假设 V2I i 只受 RB i 干扰。
        # 为了公平，建议：严格对齐 Main.py (即 V2I 受全干扰)。

        for v in active_interferers:
            # 干扰路径
            d_int = global_channel_model.calculate_3d_distance(v.curr_loc, link['rx'])
            pl_int, _, _ = global_channel_model.calculate_path_loss(d_int)
            # 干扰功率 = v.power_W (EIRP) * PathLoss
            interf_watts += v.power_W * (10 ** (-pl_int / 10.0))

        # 3. 计算 V2I 指标
        sinr = v2i_sig_watts / (noise_watts_v2i + interf_watts + 1e-30)
        # 注意：这里容量用的是 Total BW 还是 RB BW？
        # Main.py 中 V2I 容量通常基于 SYSTEM_BANDWIDTH 计算
        cap = calculate_shannon_capacity(sinr, total_bw)

        metrics["v2i_capacity"] += cap
        if 10 * np.log10(sinr) > get_param("V2I_MIN_SNR_DB", 5.0):
            metrics["v2i_success"] += 1

    # =========================================================
    # Part B: V2V 评估
    # =========================================================
    for tx_v in vehicle_list:
        # 1. 找接收者
        min_dist = float('inf')
        rx_v = None
        for neighbor in vehicle_list:
            if neighbor.id == tx_v.id: continue
            d = global_channel_model.calculate_3d_distance(tx_v.curr_loc, neighbor.curr_loc)
            if d < min_dist:
                min_dist = d
                rx_v = neighbor

        if rx_v is None or min_dist > 500: continue
        metrics["v2v_links"] += 1

        # 2. V2V 信号
        pl_total, _, _ = global_channel_model.calculate_path_loss(min_dist)
        # 直接使用 EIRP (v.power_W)
        sig_watts = tx_v.power_W * (10 ** (-pl_total / 10.0))

        # 3. V2V 干扰 (关键修正：恢复 RB 检查)
        interf_watts = 0.0
        for other in active_interferers:
            if other.id == tx_v.id or other.id == rx_v.id: continue

            # 🔥【关键】这里加回 RB 检查是完全正确的！
            # 原因：DRL 模式下大家 RB 都是 0，这个条件永远为真 -> 退化为全干扰。
            #       Ashraf 模式下大家 RB 不同，条件为假 -> 干扰消除。
            # 这才是物理层的真实表现。
            if other.assigned_rb == tx_v.assigned_rb:
                d_int = global_channel_model.calculate_3d_distance(other.curr_loc, rx_v.curr_loc)
                pl_int, _, _ = global_channel_model.calculate_path_loss(d_int)
                interf_watts += other.power_W * (10 ** (-pl_int / 10.0))

        # 4. SINR & Capacity
        sinr = sig_watts / (noise_watts_v2v + interf_watts + 1e-30)
        sinr_db = 10 * np.log10(sinr)

        metrics["v2v_capacity"] += calculate_shannon_capacity(sinr, rb_bw)
        if sinr_db >= v2v_min_snr:
            metrics["v2v_success"] += 1

    return metrics


# ==========================================
# 🚀 主运行函数
# ==========================================
def run_strict_baselines():
    print("🚀 Running Strict Baselines (Aligned Physics)")

    scenarios = [20, 40, 60, 80, 100, 120]
    n_rb = get_param("NUM_CHANNELS")

    # 🔥【修正1】获取天线增益，确保 EIRP 计算正确
    antenna_gain = get_param("GAIN_ANTENNA_T", 1.0)
    # 基线发射功率 (EIRP) = 传导功率 * 天线增益 * 1.0(全向波束)
    base_eirp = get_param("TRANSMITTED_POWER") * antenna_gain

    final_results = []

    for n in scenarios:
        print(f"\n⚡ Scenario Density N={n} ...")
        Parameters.TRAINING_VEHICLE_TARGET = n

        # 初始化
        vid, vlist = 0, []
        for _ in range(50):
            vid, vlist = vehicle_movement(vid, vlist, target_count=n)

        steps = 50
        logs = {
            "Random": {"v2v_succ": [], "v2v_cap": [], "v2i_cap": []},
            "Ashraf": {"v2v_succ": [], "v2v_cap": [], "v2i_cap": []}
        }

        for s in range(steps):
            vid, vlist = vehicle_movement(vid, vlist, target_count=n)
            if len(vlist) < 2: continue

            # --- Random ---
            for v in vlist:
                v.assigned_rb = np.random.randint(0, n_rb)
                v.power_W = base_eirp  # 赋值 EIRP

            m_rnd = evaluate_full_physics(vlist)
            if m_rnd["v2v_links"] > 0:
                logs["Random"]["v2v_succ"].append(m_rnd["v2v_success"] / m_rnd["v2v_links"])
                logs["Random"]["v2v_cap"].append(m_rnd["v2v_capacity"])
            logs["Random"]["v2i_cap"].append(m_rnd["v2i_capacity"])

            # --- Ashraf ---
            alloc = ashraf_solver.run_step(vlist)
            for v in vlist:
                v.assigned_rb = alloc.get(v.id, np.random.randint(0, n_rb))
                v.power_W = base_eirp  # 赋值 EIRP

            m_ash = evaluate_full_physics(vlist)
            if m_ash["v2v_links"] > 0:
                logs["Ashraf"]["v2v_succ"].append(m_ash["v2v_success"] / m_ash["v2v_links"])
                logs["Ashraf"]["v2v_cap"].append(m_ash["v2v_capacity"])
            logs["Ashraf"]["v2i_cap"].append(m_ash["v2i_capacity"])

            if s % 10 == 0:
                print(
                    f"   Step {s}: Rnd_SR={logs['Random']['v2v_succ'][-1]:.2f}, Ash_SR={logs['Ashraf']['v2v_succ'][-1]:.2f}")

        # 汇总
        for method in ["Random", "Ashraf"]:
            res = {
                "Density": n, "Method": method,
                "V2V_Success_Rate": np.mean(logs[method]["v2v_succ"]),
                "V2V_Sum_Capacity": np.mean(logs[method]["v2v_cap"]),
                "V2I_Sum_Capacity": np.mean(logs[method]["v2i_cap"])
            }
            final_results.append(res)
            print(f"   🏁 {method} N={n}: SR={res['V2V_Success_Rate']:.2%}")

    df = pd.DataFrame(final_results)
    df.to_csv("Strict_Baseline_Results_Full.csv", index=False)
    print("\n✅ 已保存完整指标数据.")


if __name__ == "__main__":
    run_strict_baselines()