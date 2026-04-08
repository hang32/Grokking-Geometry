import matplotlib.pyplot as plt
import numpy as np

def plot_ideal_hypothesis():
    # 模拟的数据量 (对数刻度)
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    
    # === 理想假设数据 ===
    
    # 1. 标准测试集 (同分布)：
    # 在 N 很小时，模型靠“猜位置”也能蒙对 80%（因为训练集大部分都在对应位置）。
    # 随着 N 增加，模型学会形状，准确率补全剩下的 20%，达到 100%。
    acc_std = [0.75, 0.78, 0.82, 0.85, 0.88, 0.92, 0.96, 0.99, 1.0, 1.0]
    
    # 2. OOD 测试集 (分布反转)：
    # 在 N 很小时，模型靠“猜位置”，结果全是错的！准确率极低 (接近 0)。
    # 随着 N 增加 (1024~4096)，模型发现位置靠不住，开始学形状，准确率爆发式增长 (涌现)。
    # 在 N 很大时，模型完全理解了形状，不再受位置干扰，准确率追上标准集。
    acc_ood = [0.05, 0.08, 0.15, 0.30, 0.55, 0.80, 0.92, 0.98, 0.99, 1.0]

    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(sizes, acc_std, marker='o', linestyle='-', linewidth=3, color='blue', label='Standard Test (Same Distribution)')
    plt.plot(sizes, acc_ood, marker='s', linestyle='--', linewidth=3, color='red', label='OOD Test (Inverted Position)')
    
    # 装饰图表
    plt.xscale('log')
    plt.xlabel('Training Data Size (N)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Ideal Experimental Result: The "Phase Transition"', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xticks(sizes, labels=[str(s) for s in sizes], rotation=45)
    plt.legend(fontsize=12)
    plt.ylim(-0.05, 1.05)
    
    # 标注关键区域
    plt.axvspan(32, 256, color='gray', alpha=0.1)
    plt.text(64, 0.5, "Phase 1:\nShortcut Learning\n(Cheating via Position)", color='black', ha='center')
    
    plt.axvspan(512, 4096, color='yellow', alpha=0.1)
    plt.text(1500, 0.6, "Phase 2:\nShape Learning\n(The 'Aha!' Moment)", color='black', ha='center')
    
    plt.axvspan(4096, 16384, color='green', alpha=0.1)
    plt.text(8192, 0.5, "Phase 3:\nRobustness\n(True Understanding)", color='black', ha='center')

    plt.tight_layout()
    plt.savefig('ideal_result_prediction.png')
    print("Ideal result plot generated: ideal_result_prediction.png")
    plt.show()

if __name__ == "__main__":
    plot_ideal_hypothesis()