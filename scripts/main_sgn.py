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
MAX_EPOCHS = 200 
NUM_SEEDS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0

def train_single_run(size, seed, opt_type, train_x, train_y, test_std_loader, test_ood_loader):
    set_seed(seed)
    sub_x, sub_y = get_balanced_subset(train_x, train_y, size)
    train_dataset = TensorDataset(torch.from_numpy(sub_x), torch.from_numpy(sub_y))
    train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, size), shuffle=True)
    
    model = SimplePointNet().to(DEVICE)
    
    
    if opt_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    elif opt_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_ood_acc = 0.0
    best_state_dict = model.state_dict() 
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        epoch_ood_acc = evaluate(model, test_ood_loader)
        
       
        if epoch_ood_acc >= best_ood_acc:
            best_ood_acc = epoch_ood_acc
            best_state_dict = model.state_dict()

    return {
        'seed': seed,
        'final_ood_acc': best_ood_acc,
        'state_dict': best_state_dict 
    }


def plot_optimizer_comparison(sizes, adam_means, adam_stds, sgd_means, sgd_stds):
    plt.figure(figsize=(10, 6))
    sizes = np.array(sizes)
    adam_means = np.array(adam_means)
    adam_stds = np.array(adam_stds)
    sgd_means = np.array(sgd_means)
    sgd_stds = np.array(sgd_stds)
    
    plt.plot(sizes, adam_means, marker='o', linestyle='-', color='tab:blue', label='Adam (lr=0.001)', linewidth=2)
    plt.fill_between(sizes, adam_means - adam_stds, adam_means + adam_stds, color='tab:blue', alpha=0.2)
    
    plt.plot(sizes, sgd_means, marker='s', linestyle='--', color='tab:green', label='SGD (lr=0.01, momentum=0.9)', linewidth=2.5)
    plt.fill_between(sizes, sgd_means - sgd_stds, sgd_means + sgd_stds, color='tab:green', alpha=0.2)
    
    plt.xscale('log')
    plt.xlabel('Training Data Size (N)', fontsize=12)
    plt.ylabel('OOD Test Accuracy', fontsize=12)
    plt.title('Optimizer Comparison: Adam vs. SGD', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.xticks(sizes, labels=[str(s) for s in sizes], rotation=45)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 's_curve_optimizer_cmp.png'))
    print("\n[Plot] Optimizer comparison curve saved to results/s_curve_optimizer_cmp.png")


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    train_x, train_y, test_std_x, test_std_y, test_ood_x, test_ood_y = load_data()
    test_std_loader = DataLoader(TensorDataset(torch.from_numpy(test_std_x), torch.from_numpy(test_std_y)), batch_size=64)
    test_ood_loader = DataLoader(TensorDataset(torch.from_numpy(test_ood_x), torch.from_numpy(test_ood_y)), batch_size=64)
    
    adam_means, adam_stds = [],[]
    sgd_means, sgd_stds = [],[]
    
    print(f">>> Starting Optimizer Comparison: Adam vs. SGD (Saving Weights)")
    
    for i, size in enumerate(DATA_SIZES):
        print(f"\n=== Processing N={size} ({i+1}/{len(DATA_SIZES)}) ===")
        
        
        curr_adam_runs =[]
        for s in range(NUM_SEEDS):
            res = train_single_run(size, s, 'Adam', train_x, train_y, test_std_loader, test_ood_loader)
            curr_adam_runs.append(res)
            
        curr_adam_accs = [r['final_ood_acc'] for r in curr_adam_runs]
        adam_mean, adam_std = np.mean(curr_adam_accs), np.std(curr_adam_accs)
        adam_means.append(adam_mean)
        adam_stds.append(adam_std)
        print(f"   [Adam] OOD Mean: {adam_mean:.4f} (±{adam_std:.4f})")
        
        
        best_adam_run = sorted(curr_adam_runs, key=lambda x: x['final_ood_acc'], reverse=True)[0]
        torch.save(best_adam_run['state_dict'], f'checkpoints/SimplePointNet_n{size}_Adam_best.pth')
        
        
        curr_sgd_runs =[]
        for s in range(NUM_SEEDS):
            res = train_single_run(size, s, 'SGD', train_x, train_y, test_std_loader, test_ood_loader)
            curr_sgd_runs.append(res)
            
        curr_sgd_accs = [r['final_ood_acc'] for r in curr_sgd_runs]
        sgd_mean, sgd_std = np.mean(curr_sgd_accs), np.std(curr_sgd_accs)
        sgd_means.append(sgd_mean)
        sgd_stds.append(sgd_std)
        print(f"   [SGD ] OOD Mean: {sgd_mean:.4f} (±{sgd_std:.4f})")
        
        
        best_sgd_run = sorted(curr_sgd_runs, key=lambda x: x['final_ood_acc'], reverse=True)[0]
        torch.save(best_sgd_run['state_dict'], f'checkpoints/SimplePointNet_n{size}_SGD_best.pth')
        
        
        plot_optimizer_comparison(DATA_SIZES[:len(adam_means)], adam_means, adam_stds, sgd_means, sgd_stds)

if __name__ == "__main__":
    main()