import matplotlib.pyplot as plt
import os

def plot_combined_s_curve(sizes, acc_std, acc_ood, filename="s_curve_combined.png"):
    """
    将标准测试集和 OOD 测试集的 S 曲线画在同一张图上
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(sizes, acc_std, marker='o', linestyle='-', linewidth=2.5, color='tab:blue', label='Standard Test (In-Distribution)')
    
    plt.plot(sizes, acc_ood, marker='s', linestyle='--', linewidth=2.5, color='tab:red', label='OOD Test (Out-Of-Distribution)')
    
    
    plt.xscale('log') 
    plt.xlabel('Training Data Size (N)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Scaling Law: Generalization Gap Closing', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.xticks(sizes, labels=[str(s) for s in sizes], rotation=45)
    
    
    plt.ylim(-0.05, 1.05)
    
    
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename))
    plt.close()
    print(f"Combined plot saved to results/{filename}")

def plot_learning_dynamics(train_losses, train_accs, test_std_accs, test_ood_accs, title_suffix, filename):
    epochs = range(1, len(train_accs) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:orange'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color, fontweight='bold')
    line1 = ax1.plot(epochs, train_losses, label='Train Loss', color=color, linewidth=2, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Accuracy', color='tab:blue', fontweight='bold')
    line2 = ax2.plot(epochs, train_accs, label='Train Acc', color='tab:blue', linewidth=1.5, alpha=0.5)
    line3 = ax2.plot(epochs, test_std_accs, label='Test Std Acc', color='tab:green', linewidth=2.5)
    line4 = ax2.plot(epochs, test_ood_accs, label='Test OOD Acc', color='tab:red', linewidth=2.5, linestyle='--')
    
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(-0.05, 1.05)
    
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.title(f'Learning Dynamics ({title_suffix})')
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename))
    plt.close()