import torch
import torch.optim as optim
import Parameters
import Main
import GNNModel
from Topology import formulate_global_list_dqn


def run_fine_tuning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 启动快速微调 (Fine-tuning for V2I Protection)...")

    # 1. 强制覆盖参数 (确保使用了更严厉的惩罚)
    Parameters.V2I_MULTIPLIER = 1.0  # 确保生效
    Parameters.TRAINING_VEHICLE_TARGET = 80  # 在中等密度下微调
    Parameters.NUM_VEHICLES = 80
    Parameters.RL_N_EPOCHS = 100  # 只需要跑 100 轮左右
    Parameters.ABLATION_SUFFIX = "_Strict_V2I"  # 防止覆盖旧文件

    # 2. 初始化
    formulate_global_list_dqn(Parameters.global_dqn_list, device)

    # 3. 加载你现有的“偏科”模型 (V2V很强那个)
    # 请确保文件名正确，就是你刚才发给我的那个
    pretrained_model = "model_Universal_Final_V5.pt"

    print(f"📥 加载预训练模型: {pretrained_model}")
    model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64).to(device)
    target_model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64).to(device)

    checkpoint = torch.load(pretrained_model, map_location=device)
    model.load_state_dict(checkpoint)
    target_model.load_state_dict(checkpoint)

    # 挂载到全局
    GNNModel.global_gnn_model = model
    GNNModel.global_target_gnn_model = target_model

    # 4. 减小学习率 (微调不需要太大的步长)
    # 之前可能是 0.0003，现在用 0.0001 或更小，防止破坏已有的 V2V 知识
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    # 5. 开始训练
    # 这次训练模型会发现：如果不顾 V2I，Reward 会非常低
    Main.rl(gnn_optimizer=optimizer, device=device)

    # 6. 保存新模型
    new_model_name = "model_Universal_Strict.pt"
    torch.save(model.state_dict(), new_model_name)
    print(f"✅ 微调完成！新模型已保存为: {new_model_name}")
    print("👉 请将 run_paper_comparison_test.py 中的 Proposed 路径改为这个新文件，然后重新测试。")


if __name__ == "__main__":
    run_fine_tuning()