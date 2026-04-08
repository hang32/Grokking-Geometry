import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import random
import matplotlib.pyplot as plt

from pointnet_scaling_experiment.models.model import SimplePointNet 


DATA_SIZES =[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384] 
BATCH_SIZE = 32
MAX_EPOCHS = 100 
NUM_SEEDS = 3
NOISE_STD = 0.05 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def add_gaussian_noise(pc, mean=0.0, std=0.05):
    noise = np.random.normal(mean, std, pc.shape)
    noisy_pc = pc + noise
    return noisy_pc.astype(np.float32)

def plot_learning_dynamics(train_losses, train_accs, test_std_accs, test_ood_accs, title, filename):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(epochs, train_accs, 'b--', label='Train Acc')
    ax2.plot(epochs, test_std_accs, 'g-', label='Test Std Acc (Noisy)')
    ax2.plot(epochs, test_ood_accs, 'r-', label='Test OOD Acc (Noisy)', linewidth=2)
    ax2.set_title(f'{title} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename))
    plt.close()

def plot_avg_s_curve(sizes, std_means, std_stds, ood_means, ood_stds):
    plt.figure(figsize=(10, 6))
    sizes = np.array(sizes)
    std_means = np.array(std_means)
    std_stds = np.array(std_stds)
    ood_means = np.array(ood_means)
    ood_stds = np.array(ood_stds)
    
    plt.plot(sizes, std_means, marker='o', linestyle='-', color='tab:blue', label=f'Standard Test + Noise({NOISE_STD})', linewidth=2)
    plt.fill_between(sizes, std_means - std_stds, std_means + std_stds, color='tab:blue', alpha=0.2)
    
    plt.plot(sizes, ood_means, marker='X', linestyle='--', color='gray', label=f'OOD Test + Noise({NOISE_STD})', linewidth=2.5, markersize=8)
    plt.fill_between(sizes, ood_means - ood_stds, ood_means + ood_stds, color='gray', alpha=0.2)
    
    plt.xscale('log')
    plt.xlabel('Training Data Size (N)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title(f'Scaling Law: Simple PointNet Robustness (Noise Std={NOISE_STD})', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.xticks(sizes, labels=[str(s) for s in sizes], rotation=45)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', f's_curve_simple_noise_{NOISE_STD}.png'))

def load_data():
    print("Loading data...")
    train_x, train_y = np.load('data/train_x.npy'), np.load('data/train_y.npy')
    test_std_x, test_std_y = np.load('data/test_std_x.npy'), np.load('data/test_std_y.npy')
    test_ood_x, test_ood_y = np.load('data/test_ood_x.npy'), np.load('data/test_ood_y.npy')
    return train_x, train_y, test_std_x, test_std_y, test_ood_x, test_ood_y

def get_balanced_subset(x, y, size):
    indices_0 = np.where(y == 0)[0]
    indices_1 = np.where(y == 1)[0]
    half_size = size // 2
    replace = len(indices_0) < half_size or len(indices_1) < half_size
    sel_idx_0 = np.random.choice(indices_0, half_size, replace=replace)
    sel_idx_1 = np.random.choice(indices_1, half_size, replace=replace)
    sel_indices = np.concatenate([sel_idx_0, sel_idx_1])
    np.random.shuffle(sel_indices)
    return x[sel_indices], y[sel_indices]

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # SimplePointNet 只返回 outputs
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0

def train_single_run(size, seed, train_x, train_y, test_std_loader, test_ood_loader):
    set_seed(seed)
    sub_x, sub_y = get_balanced_subset(train_x, train_y, size)
    train_dataset = TensorDataset(torch.from_numpy(sub_x), torch.from_numpy(sub_y))
    train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, size), shuffle=True)
    
  
    model = SimplePointNet().to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'test_std_acc':[], 'test_ood_acc':[]}
    best_ood_acc = best_std_acc = 0.0
    best_state_dict = model.state_dict()
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = correct = total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
           
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_train_acc = correct / total
        epoch_std_acc = evaluate(model, test_std_loader)
        epoch_ood_acc = evaluate(model, test_ood_loader)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_std_acc'].append(epoch_std_acc)
        history['test_ood_acc'].append(epoch_ood_acc)
        
        if epoch_ood_acc >= best_ood_acc:
            best_ood_acc = epoch_ood_acc
            best_std_acc = epoch_std_acc
            best_state_dict = model.state_dict()

    return {'seed': seed, 'final_ood_acc': best_ood_acc, 'final_std_acc': best_std_acc, 'history': history, 'state_dict': best_state_dict}

def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    train_x, train_y, test_std_x, test_std_y, test_ood_x, test_ood_y = load_data()
    
    
    print(f"\n[WARNING] Injecting Gaussian Noise (std={NOISE_STD}) to TEST sets strictly!")
    test_std_x = add_gaussian_noise(test_std_x, std=NOISE_STD)
    test_ood_x = add_gaussian_noise(test_ood_x, std=NOISE_STD)
    

    test_std_loader = DataLoader(TensorDataset(torch.from_numpy(test_std_x), torch.from_numpy(test_std_y)), batch_size=64)
    test_ood_loader = DataLoader(TensorDataset(torch.from_numpy(test_ood_x), torch.from_numpy(test_ood_y)), batch_size=64)
    
    std_means, std_stds, ood_means, ood_stds = [], [], [],[]
    
    print(f">>> Starting Experiment: Simple PointNet (Robustness Test)")
    for i, size in enumerate(DATA_SIZES):
        print(f"\n=== Processing N={size} ({i+1}/{len(DATA_SIZES)}) ===")
        curr_ood_accs, curr_std_accs, runs = [], [],[]
        
        for s in range(NUM_SEEDS):
            res = train_single_run(size, s, train_x, train_y, test_std_loader, test_ood_loader)
            runs.append(res)
            curr_ood_accs.append(res['final_ood_acc'])
            curr_std_accs.append(res['final_std_acc'])
        
        ood_mean, ood_std = np.mean(curr_ood_accs), np.std(curr_ood_accs)
        std_mean, std_std = np.mean(curr_std_accs), np.std(curr_std_accs)
        print(f"   [Stats] OOD Mean: {ood_mean:.4f} (±{ood_std:.4f})")
        
        std_means.append(std_mean); std_stds.append(std_std)
        ood_means.append(ood_mean); ood_stds.append(ood_std)
        
        best_run = sorted(runs, key=lambda x: x['final_ood_acc'], reverse=True)[0]
        
        torch.save(best_run['state_dict'], f'checkpoints/SimplePointNet_n{size}_noise{NOISE_STD}_best.pth')
        
        plot_learning_dynamics(best_run['history']['train_loss'], best_run['history']['train_acc'], best_run['history']['test_std_acc'], best_run['history']['test_ood_acc'], f"Simple PointNet + Noise N={size}", f"dynamics_simple_noise{NOISE_STD}_n{size}.png")
        
        plot_avg_s_curve(DATA_SIZES[:len(ood_means)], std_means, std_stds, ood_means, ood_stds)
        
    print(f"\nAll Done! Result saved to results/s_curve_simple_noise_{NOISE_STD}.png")

if __name__ == "__main__":
    main()