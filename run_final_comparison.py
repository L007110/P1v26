import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# === 核心导入 ===
import Parameters
from Topology import formulate_global_list_dqn, vehicle_movement
from Ashraf_Algorithm import ashraf_solver
import GNNModel
from GraphBuilder import global_graph_builder

# ================= 配置区域 =================
# 1. GNN 模型路径
GNN_MODEL_PATH = "model_Universal_Final_V5.pt"

# 2. No-GNN 模型路径 (请修改为您实际的文件名)
# 通常 No-GNN 模型保存为一个字典: {'dqn_0': state_dict, 'dqn_1': state_dict, ...}
NOGNN_MODEL_PATH = "model_NoGNN_CL_N140.pt"

TEST_DENSITIES = [20, 40, 60, 80, 100, 120, 140]
NUM_TRIALS = 50  # 每个点测50次取平均
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自动获取 RB 数量 (对齐 run_baselines_legacy.py)
if hasattr(Parameters, "NUM_RB"):
    NUM_RB = Parameters.NUM_RB
elif hasattr(Parameters, "NUM_CHANNELS"):
    NUM_RB = Parameters.NUM_CHANNELS
else:
    NUM_RB = 20  # 默认值
# ===========================================

print(f"📍 Benchmarking on: {DEVICE}")
print(f"📂 GNN Model: {GNN_MODEL_PATH}")
print(f"📂 No-GNN Model: {NOGNN_MODEL_PATH}")
print(f"⚙️ Ashraf RB Setting: {NUM_RB}")


def clean_setup(n_vehicles):
    """强制重置环境"""
    Parameters.TRAINING_VEHICLE_TARGET = n_vehicles
    Parameters.NUM_VEHICLES = n_vehicles
    formulate_global_list_dqn(Parameters.global_dqn_list, DEVICE)
    vid = 0
    vlist = []
    # 预热生成
    for _ in range(10):
        vid, vlist = vehicle_movement(vid, vlist, target_count=n_vehicles)
    # 强制修正数量
    if len(vlist) != n_vehicles:
        while len(vlist) < n_vehicles:
            vid, vlist = vehicle_movement(vid, vlist, target_count=n_vehicles)
        vlist = vlist[:n_vehicles]
    return Parameters.global_dqn_list, vlist


def measure_ashraf(n):
    """传统算法 Ashraf (CPU Only) - 配置已对齐 legacy"""
    _, vlist = clean_setup(n)

    # === 关键：设置 RB 数量，确保矩阵维度正确 ===
    ashraf_solver.n_rb = NUM_RB

    timings = []
    for _ in range(NUM_TRIALS):
        # 随机扰动位置
        for v in vlist: v.curr_loc = (np.random.uniform(0, 1000), np.random.uniform(0, 1000))

        t0 = time.perf_counter()
        # 核心计算步骤
        ashraf_solver.run_step(vlist)
        t1 = time.perf_counter()

        timings.append((t1 - t0) * 1000)
    return np.mean(timings)


def measure_nognn(n):
    """Baseline: No-GNN (加载训练好的模型)"""
    dqn_list, vlist = clean_setup(n)

    # === 加载 No-GNN 模型权重 ===
    if os.path.exists(NOGNN_MODEL_PATH):
        try:
            checkpoint = torch.load(NOGNN_MODEL_PATH, map_location=DEVICE)
            # 检查 checkpoint 格式
            if isinstance(checkpoint, dict):
                # 尝试按 dqn_id 加载
                loaded_count = 0
                for dqn in dqn_list:
                    key = f"dqn_{dqn.dqn_id}"
                    if key in checkpoint:
                        dqn.load_state_dict(checkpoint[key])
                        loaded_count += 1

                if n == TEST_DENSITIES[0]:
                    if loaded_count > 0:
                        print(f"✅ No-GNN 模型加载成功: {loaded_count}/{len(dqn_list)} agents loaded.")
                    else:
                        print(f"⚠️ No-GNN 模型加载警告: 字典中未找到匹配的 dqn_id (e.g. 'dqn_0')。使用随机权重。")
            else:
                if n == TEST_DENSITIES[0]:
                    print(f"⚠️ No-GNN 模型格式不匹配 (Expected dict, got {type(checkpoint)})。使用随机权重。")
        except Exception as e:
            if n == TEST_DENSITIES[0]:
                print(f"⚠️ No-GNN 模型加载失败 ({e})。使用随机权重。")
    else:
        if n == TEST_DENSITIES[0]:
            print(f"ℹ️ 未找到文件 {NOGNN_MODEL_PATH}，使用随机权重。")

    for dqn in dqn_list: dqn.eval()

    # Warmup
    if torch.cuda.is_available():
        dqn_list[0](torch.randn(1, Parameters.RL_N_STATES).to(DEVICE))
        torch.cuda.synchronize()

    timings = []
    for _ in range(NUM_TRIALS):
        for v in vlist: v.curr_loc = (np.random.uniform(0, 1000), np.random.uniform(0, 1000))

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 1. 模拟 No-GNN 状态构建 (CPU)
        for dqn in dqn_list:
            dqn.vehicle_in_dqn_range_by_distance = []
            # 简单的感知范围筛选
            for v in vlist:
                dist = np.sqrt((dqn.bs_loc[0] - v.curr_loc[0]) ** 2 + (dqn.bs_loc[1] - v.curr_loc[1]) ** 2)
                if dist < 500: dqn.vehicle_in_dqn_range_by_distance.append(v)
            dqn.curr_state = [0.0] * Parameters.RL_N_STATES  # Fake Input

        # 2. 推理 (GPU)
        with torch.no_grad():
            for dqn in dqn_list:
                if dqn.vehicle_in_dqn_range_by_distance:
                    _ = dqn(torch.tensor(dqn.curr_state).float().to(DEVICE).unsqueeze(0))

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)
    return np.mean(timings)


def measure_gnn_detailed(n):
    """Ours: GNN-CL (加载训练好的模型)"""
    dqn_list, vlist = clean_setup(n)

    # 初始化模型结构
    model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64).to(DEVICE)
    model.eval()

    # === 加载 GNN 模型权重 ===
    if os.path.exists(GNN_MODEL_PATH):
        try:
            state_dict = torch.load(GNN_MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            if n == TEST_DENSITIES[0]:
                print(f"✅ GNN 模型加载成功: {GNN_MODEL_PATH}")
        except Exception as e:
            if n == TEST_DENSITIES[0]:
                print(f"⚠️ GNN 模型加载失败 ({e})。使用随机权重。")
    else:
        if n == TEST_DENSITIES[0]:
            print(f"ℹ️ 未找到文件 {GNN_MODEL_PATH}，使用随机权重。")

    # Warmup
    try:
        g = global_graph_builder.build_dynamic_graph(dqn_list, vlist, 0)
        _move_to_device(g, DEVICE)
        model(g)
    except:
        pass

    t_inf = []
    t_sys = []

    for i in range(NUM_TRIALS):
        for v in vlist: v.curr_loc = (np.random.uniform(0, 1000), np.random.uniform(0, 1000))

        # System Time Start (构图 + 传输 + 推理)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 1. 构图 & 搬运
        graph = global_graph_builder.build_dynamic_graph(dqn_list, vlist, i)
        graph = _move_to_device(graph, DEVICE)

        # Inference Time Start (纯推理)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.perf_counter()

        # 2. 模型推理
        with torch.no_grad():
            _ = model(graph)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t2 = time.perf_counter()

        t_inf.append((t2 - t1) * 1000)
        t_sys.append((t2 - t0) * 1000)

    return np.mean(t_inf), np.mean(t_sys)


def _move_to_device(graph_data, device):
    if graph_data is None: return None
    graph_data['node_features']['features'] = graph_data['node_features']['features'].to(device)
    graph_data['node_features']['types'] = graph_data['node_features']['types'].to(device)
    for et in ['communication', 'interference', 'proximity']:
        if graph_data['edge_features'][et] is not None:
            graph_data['edge_features'][et]['edge_index'] = graph_data['edge_features'][et]['edge_index'].to(device)
            graph_data['edge_features'][et]['edge_attr'] = graph_data['edge_features'][et]['edge_attr'].to(device)
    return graph_data


if __name__ == "__main__":
    # === 新增：超级预热 ===
    print("🔥 Warming up system heavily...")
    # 跑一次最复杂的 N=140 来把内存和库都加载好
    measure_ashraf(140)
    measure_nognn(140)
    measure_gnn_detailed(140)
    print("✅ Warmup done. Starting benchmark.\n")

    results = []
    # 表头
    print(f"\n{'N':<5} | {'Ashraf':<10} | {'NoGNN':<10} | {'GNN-Inf':<10} | {'GNN-Sys':<10}")
    print("-" * 55)

    for n in TEST_DENSITIES:
        t_ash = measure_ashraf(n)
        t_no = measure_nognn(n)
        t_gnn_inf, t_gnn_sys = measure_gnn_detailed(n)

        print(f"{n:<5} | {t_ash:<10.2f} | {t_no:<10.2f} | {t_gnn_inf:<10.2f} | {t_gnn_sys:<10.2f}")
        results.append({
            "Density": n, "Ashraf": t_ash, "NoGNN": t_no,
            "GNN_Inf": t_gnn_inf, "GNN_Sys": t_gnn_sys
        })

    df = pd.DataFrame(results)
    df.to_csv("Final_Complexity_All.csv", index=False)
    print("\n✅ 数据已保存: Final_Complexity_All.csv")
    print("👉 请使用此 CSV 绘制对比曲线。")

    # 自动绘图
    plt.figure(figsize=(10, 6))
    plt.plot(df["Density"], df["Ashraf"], 'r-o', label='Ashraf (Legacy)')
    plt.plot(df["Density"], df["NoGNN"], 'b-^', label='No-GNN (Baseline)')
    plt.plot(df["Density"], df["GNN_Sys"], 'g--', label='Ours (System Total)')
    plt.plot(df["Density"], df["GNN_Inf"], 'g-s', label='Ours (Inference Only)')

    plt.axhline(y=10, color='k', linestyle=':', label='10ms Limit')
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Latency (ms)')
    plt.title('Computational Complexity Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("Final_Complexity_Plot.png")
    print("🖼️ 对比图已保存: Final_Complexity_Plot.png")