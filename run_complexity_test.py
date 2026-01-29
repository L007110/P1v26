import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Parameters
from Topology import formulate_global_list_dqn, vehicle_movement
from Ashraf_Algorithm import ashraf_solver
import GNNModel
from GraphBuilder import global_graph_builder

# 配置
TEST_DENSITIES = [20, 40, 60, 80, 100, 120, 140]
NUM_TRIALS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_setup(n_vehicles):
    Parameters.TRAINING_VEHICLE_TARGET = n_vehicles
    Parameters.NUM_VEHICLES = n_vehicles
    formulate_global_list_dqn(Parameters.global_dqn_list, DEVICE)
    vid = 0
    vlist = []
    for _ in range(10):
        vid, vlist = vehicle_movement(vid, vlist, target_count=n_vehicles)
    return Parameters.global_dqn_list, vlist


def measure_gnn_detailed(n):
    dqn_list, vlist = clean_setup(n)
    model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64).to(DEVICE)
    model.eval()

    # 预构建一个图用于 Warmup
    try:
        g = global_graph_builder.build_dynamic_graph(dqn_list, vlist, 0)
        _move_to_device(g, DEVICE)
        model(g)
    except:
        pass

    t_inference_only = []
    t_system_total = []

    for i in range(NUM_TRIALS):
        # 扰动位置
        for v in vlist: v.curr_loc = (np.random.uniform(0, 1000), np.random.uniform(0, 1000))

        # 1. 开始计时 (系统总时间起点)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 构图 & 搬运 (CPU -> GPU)
        graph = global_graph_builder.build_dynamic_graph(dqn_list, vlist, i)
        graph = _move_to_device(graph, DEVICE)

        # 2. 中间计时 (推理时间起点)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.perf_counter()

        with torch.no_grad():
            _ = model(graph)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t2 = time.perf_counter()

        # 记录
        t_inference_only.append((t2 - t1) * 1000)  # 纯推理 (对标 Ji et al.)
        t_system_total.append((t2 - t0) * 1000)  # 系统总耗时 (诚实数据)

    return np.mean(t_inference_only), np.mean(t_system_total)


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
    results = []
    print(f"{'Density':<10} | {'GNN Inference (ms)':<20} | {'GNN System (ms)':<20}")
    print("-" * 60)

    for n in TEST_DENSITIES:
        t_inf, t_sys = measure_gnn_detailed(n)
        print(f"{n:<10} | {t_inf:<20.4f} | {t_sys:<20.4f}")
        results.append({"Density": n, "GNN_Inference": t_inf, "GNN_System": t_sys})

    df = pd.DataFrame(results)
    df.to_csv("GNN_Complexity_Split.csv", index=False)

    # 画图：只画 Inference Time 去 PK 别人的论文
    plt.figure(figsize=(8, 5))
    plt.plot(df["Density"], df["GNN_Inference"], marker='s', label="Model Inference (GPU)", color='green')
    plt.plot(df["Density"], df["GNN_System"], marker='o', label="System Total (CPU+GPU)", color='orange',
             linestyle='--')
    plt.xlabel("Number of Vehicles")
    plt.ylabel("Time (ms)")
    plt.title("Computational Overhead Analysis")
    plt.legend()
    plt.grid(True)
    plt.savefig("Complexity_Split_View.png")