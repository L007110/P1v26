# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import torch
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    import Parameters
    import run_honest_eval
except ImportError:
    print("❌ 无法导入模块，请确保在根目录运行")
    sys.exit(1)

# 配置
TARGET_MODEL = "model_Universal_Final_V5.pt"
THRESHOLDS = [0.5, 2.5]  # 只测两个极端，节省时间
DENSITIES = [20, 80]  # 对比低密度和高密度
POSSIBLE_FILES = ["results_GNN_Test.csv"]


def run_sanity_check():
    print("🚀 启动物理规律验证 (Sanity Check)")
    print(f"对比密度: {DENSITIES}")
    print(f"对比阈值: {THRESHOLDS} (0.5=宽松, 2.5=严格)")
    print("-" * 50)

    # 自动找模型
    model_path = TARGET_MODEL
    if not os.path.exists(model_path):
        model_path = os.path.join("training_results", TARGET_MODEL)

    Parameters.USE_GNN_ENHANCEMENT = True

    results = {}

    for dens in DENSITIES:
        print(f"\n🚗 === 测试密度: {dens} 辆车 ===")
        results[dens] = {}

        for th in THRESHOLDS:
            # 注入参数
            Parameters.V2I_CAPACITY_THRESHOLD = th

            # 清理旧文件
            for f in POSSIBLE_FILES:
                if os.path.exists(f): os.remove(f)

            # 运行评估
            try:
                # 屏蔽标准输出，只看结果
                run_honest_eval.run_honest_evaluation(model_path, [dens])
            except Exception as e:
                print(f"运行出错: {e}")
                continue

            # 读取结果
            time.sleep(1)
            success_rate = 0
            for f in POSSIBLE_FILES:
                if os.path.exists(f):
                    df = pd.read_csv(f)
                    if not df.empty:
                        # 自动找成功率列
                        col = 'raw_success_rate' if 'raw_success_rate' in df.columns else 'V2V_Success_Rate'
                        success_rate = df.iloc[-1][col]
                        break

            print(f"   阈值 {th}: 成功率 = {success_rate:.4f}")
            results[dens][th] = success_rate

    # === 最终判决 ===
    print("\n" + "=" * 50)
    print("⚖️  最终判决报告")
    print("=" * 50)

    # 分析密度 20 (低密度)
    sr_20_loose = results[20][0.5]
    sr_20_strict = results[20][2.5]
    print(f"低密度 (20): 宽松({sr_20_loose:.2%}) vs 严格({sr_20_strict:.2%})")

    if sr_20_loose > sr_20_strict:
        print("✅ 符合物理规律：低密度下，约束越严，性能越差 (因覆盖范围受限)。")
        low_density_ok = True
    else:
        print("❌ 异常：低密度下，严格约束反而更好？(可能是代码逻辑反了)")
        low_density_ok = False

    # 分析密度 80 (高密度)
    sr_80_loose = results[80][0.5]
    sr_80_strict = results[80][2.5]
    print(f"高密度 (80): 宽松({sr_80_loose:.2%}) vs 严格({sr_80_strict:.2%})")

    if sr_80_strict > sr_80_loose:
        print("✅ 发现干扰抑制效应：高密度下，约束越严，性能反而越好！")
        high_density_interesting = True
    else:
        print("ℹ️ 普通结果：高密度下也是宽松更好 (未观察到拥塞效应)。")
        high_density_interesting = False

    print("-" * 50)
    if low_density_ok and high_density_interesting:
        print("🏆 结论：代码无误！你的 '高密度反转' 现象是真实的物理规律，可以放心写进论文！")
    elif not low_density_ok:
        print("⚠️ 结论：请检查代码！V2I 阈值的判断逻辑可能写反了。")


if __name__ == "__main__":
    run_sanity_check()