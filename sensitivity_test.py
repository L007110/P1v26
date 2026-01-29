# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import torch
import time

# === 1. 导入项目模块 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    import Parameters
    import run_honest_eval
except ImportError as e:
    print("❌ 错误: 无法导入 run_honest_eval.py 或 Parameters.py")
    sys.exit(1)

# === 2. 实验配置 ===
TARGET_MODEL = "model_Universal_Final_V5.pt"
THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5]
TARGET_DENSITY = [80]
OUTPUT_FILE = "sensitivity_analysis_results.csv"

# [修复] 这里加上了正确的文件名 results_GNN_Test.csv
POSSIBLE_RESULT_FILES = [
    "results_GNN_Test.csv",  # <--- 你的 run_honest_eval 实际生成的文件
    "results_CL.csv",
    "results_GNN_Evaluation.csv"
]


def run_sensitivity_analysis():
    print("\n" + "=" * 60)
    print("🚀 启动敏感性分析 (基于 run_honest_eval)")
    print(f"📂 加载模型: {TARGET_MODEL}")
    print(f"🎯 目标密度: {TARGET_DENSITY}")
    print("=" * 60 + "\n")

    sensitivity_results = []

    # 自动寻找模型路径
    model_path = TARGET_MODEL
    if not os.path.exists(model_path):
        alt_path = os.path.join("training_results", TARGET_MODEL)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"❌ 严重错误: 找不到模型文件 {TARGET_MODEL}")
            return

    # 备份参数
    original_threshold = Parameters.V2I_CAPACITY_THRESHOLD

    for th in THRESHOLDS:
        print(f"\n👉 [当前测试] V2I Threshold = {th} bps/Hz")

        # --- A. 注入参数 ---
        Parameters.V2I_CAPACITY_THRESHOLD = th
        Parameters.USE_GNN_ENHANCEMENT = True

        # --- B. 运行评估 ---
        try:
            # 清理旧结果
            for f in POSSIBLE_RESULT_FILES:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass

            # 调用评估
            run_honest_eval.run_honest_evaluation(model_path, TARGET_DENSITY)

        except Exception as e:
            print(f"❌ 运行错误: {e}")
            continue

        # --- C. 抓取结果 ---
        time.sleep(1)
        found = False

        for fpath in POSSIBLE_RESULT_FILES:
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath)
                    if not df.empty:
                        # 兼容不同的列名写法
                        col_density = 'density' if 'density' in df.columns else 'Density'
                        col_success = 'raw_success_rate' if 'raw_success_rate' in df.columns else 'V2V_Success_Rate'

                        # 有些 CSV 可能第一列是大写的 Density
                        if col_density not in df.columns:
                            # 盲猜第一列
                            col_density = df.columns[0]

                        # 查找密度 80
                        target_row = df[df[col_density] == TARGET_DENSITY[0]]

                        if not target_row.empty:
                            row = target_row.iloc[-1]
                            success_rate = row[col_success]
                            v2v_sum = row['v2v_sum_mbps'] if 'v2v_sum_mbps' in df.columns else row['V2V_Sum_Capacity']

                            print(f"✅ 抓取成功: Threshold={th} -> SR={success_rate:.4f}")

                            sensitivity_results.append({
                                "V2I_Threshold": th,
                                "Success_Rate": success_rate,
                                "V2V_Sum_Rate": v2v_sum,
                                "Density": TARGET_DENSITY[0]
                            })
                            found = True
                            break
                except Exception as e:
                    print(f"⚠️ 读取出错: {e}")

        if not found:
            print(f"⚠️ 警告: 未能从 {POSSIBLE_RESULT_FILES} 中读取到数据。")

    # 恢复参数
    Parameters.V2I_CAPACITY_THRESHOLD = original_threshold

    # 保存结果
    if sensitivity_results:
        result_df = pd.DataFrame(sensitivity_results)
        result_df.to_csv(OUTPUT_FILE, index=False)
        print("\n" + "=" * 60)
        print(f"🎉 测试完成！结果已保存至: {OUTPUT_FILE}")
        print("=" * 60)
        print(result_df)
    else:
        print("\n❌ 未收集到任何结果。")


if __name__ == "__main__":
    run_sensitivity_analysis()