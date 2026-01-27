
import numpy as np
from sklearn.cluster import SpectralClustering


class AshrafAlgorithm:
    def __init__(self, n_rb=10, max_swap_iter=50):
        self.n_rb = n_rb
        self.max_swap_iter = max_swap_iter

    def run_step(self, vehicle_list):
        """执行 Ashraf 算法的一个时隙决策"""
        if not vehicle_list:
            return {}

        n_vehicles = len(vehicle_list)
        positions = np.array([v.curr_loc for v in vehicle_list])

        # 1. 构建相似度矩阵 (基于距离)
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        distances = np.sqrt(dist_sq)

        sigma, epsilon = 50.0, 200.0
        affinity_matrix = np.zeros((n_vehicles, n_vehicles))
        mask = distances <= epsilon
        affinity_matrix[mask] = np.exp(-dist_sq[mask] / (2 * sigma ** 2))
        np.fill_diagonal(affinity_matrix, 0.0)

        # 2. 谱聚类
        n_clusters = max(2, int(n_vehicles / 10))
        n_clusters = min(n_clusters, n_vehicles)  # 防止簇比车多

        try:
            sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
            labels = sc.fit_predict(affinity_matrix)
        except:
            labels = np.zeros(n_vehicles, dtype=int)

        # 3. 簇内资源分配与交换
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters: clusters[label] = []
            clusters[label].append(idx)

        rb_allocations = {}  # {vehicle_id: rb_index}

        for label, member_indices in clusters.items():
            # A. 初始随机分配
            cluster_rbs = np.random.choice(self.n_rb, size=len(member_indices))
            current_mapping = {midx: rb for midx, rb in zip(member_indices, cluster_rbs)}

            # B. 交换匹配 (Swap Matching)
            for _ in range(self.max_swap_iter):
                if len(member_indices) < 2: break
                a, b = np.random.choice(member_indices, 2, replace=False)

                # 简化代价函数：只看簇内干扰
                cost_before = self._calc_interf(current_mapping, member_indices, positions)

                # 尝试交换
                current_mapping[a], current_mapping[b] = current_mapping[b], current_mapping[a]
                cost_after = self._calc_interf(current_mapping, member_indices, positions)

                # 如果代价变大，换回来
                if cost_after > cost_before:
                    current_mapping[a], current_mapping[b] = current_mapping[b], current_mapping[a]

            for midx, rb in current_mapping.items():
                v_id = vehicle_list[midx].id
                rb_allocations[v_id] = rb

        return rb_allocations

    def _calc_interf(self, mapping, members, positions):
        interf = 0.0
        for i in members:
            for j in members:
                if i >= j: continue
                if mapping[i] == mapping[j]:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    interf += 1.0 / (dist + 1.0)
        return interf


ashraf_solver = AshrafAlgorithm()