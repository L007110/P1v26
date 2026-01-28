import sys
import torch
import os
import Parameters  # 1. 先导入 Parameters

# ==========================================
# 🛑 强制关闭 GNN (必须在 import Main 之前!)
# ==========================================
print("⚡ FORCE DISABLING GNN MODE...")
Parameters.USE_GNN_ENHANCEMENT = False
Parameters.GNN_ARCH = "NONE"
# ==========================================

import Main
import GNNModel
from Topology import formulate_global_list_dqn
# 导入混合密度拦截器，这是让线不交错的关键
from run_smart_curriculum_mix_v5 import VehicleDensityMixer, density_mixer


def train_nognn_with_cl():
    # 自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 Using device: {device}")

    CURRICULUM_LEVELS = [20, 40, 60, 80, 100, 120, 140]
    last_checkpoint = None

    for n in CURRICULUM_LEVELS:
        print(f"\n{'=' * 20} 🚀 Starting Level N={n} {'=' * 20}")

        # 1. 配置当前关卡参数
        Parameters.NUM_VEHICLES = n
        Parameters.TRAINING_VEHICLE_TARGET = n
        # 根据难度动态调整 Epoch (可选)
        # 针对高密度增加训练量
        if n <= 60:
            Parameters.RL_N_EPOCHS = 400
        elif n <= 100:
            Parameters.RL_N_EPOCHS = 600
        else:
            Parameters.RL_N_EPOCHS = 1000
        Parameters.ABLATION_SUFFIX = f"_NoGNN_CL_N{n}"

        # 2. 初始化/重置 DQN 列表
        formulate_global_list_dqn(Parameters.global_dqn_list, device)

        # 3. 【核心】权重继承：带着上一关的经验继续练
        if last_checkpoint and os.path.exists(last_checkpoint):
            print(f"📥 Loading weights from previous level: {last_checkpoint}")
            weights = torch.load(last_checkpoint, map_location=device)

            for dqn in Parameters.global_dqn_list:
                key = f'dqn_{dqn.dqn_id}'
                if key in weights:
                    # 第一步：直接加载（strict=False 会自动忽略嵌套冲突，并加载匹配的权重）
                    # 此时 dqn 自身的 feature_layer 等会被正确加载
                    dqn.load_state_dict(weights[key], strict=False)

                    # 第二步：强行同步目标网络
                    # 既然 dqn 已经拿到了 N=20 的在线权重，我们直接把它复刻给目标网络
                    if hasattr(dqn, 'target_network') and dqn.target_network is not None:
                        dqn.target_network.load_state_dict(dqn.state_dict(), strict=False)
                        print(f"   🔄 Agent {dqn.dqn_id}: Weights Inherited & Target Synced.")

        # 4. 【核心】挂载混合密度拦截器：防止在高密度训练时“忘本”
        # 这样模型在练 80 辆车时，也会偶尔复习 20 辆车的场景
        density_mixer.set_level(n)
        # 确保 Main.rl 内部调用的车辆生成逻辑被拦截器接管
        # 注意：只要 import 了 density_mixer，它通常已经通过 Monkey Patch 挂载好了

        # 5. 执行训练
        try:
            Main.rl(device=device)
        except Exception as e:
            print(f"❌ Level N={n} training failed: {e}")
            break

        # 6. 保存成果
        last_checkpoint = f"model_NoGNN_CL_N{n}.pt"
        save_data = {f'dqn_{dqn.dqn_id}': dqn.state_dict() for dqn in Parameters.global_dqn_list}
        torch.save(save_data, last_checkpoint)
        print(f"✅ Level N={n} finished and saved.")

    print("\n🎉 全流程课程学习已圆满完成！")


if __name__ == "__main__":
    train_nognn_with_cl()