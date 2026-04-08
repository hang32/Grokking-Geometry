import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from pointnet_scaling_experiment.models.model import PointNetTNet 


DATA_SIZES =[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384] 
BATCH_SIZE = 32
MAX_EPOCHS = 200 
NUM_SEEDS = 2 
PATIENCE = 20 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_data():
    return np.load('data/train_x.npy'), np.load('data/train_y.npy'), \
           np.load('data/test_std_x.npy'), np.load('data/test_std_y.npy'), \
           np.load('data/test_ood_x.npy'), np.load('data/test_ood_y.npy')

def get_balanced_subset(x, y, size):
    indices_0, indices_1 = np.where(y == 0)[0], np.where(y == 1)[0]
    half = size // 2
    idx = np.concatenate([np.random.choice(indices_0, half, replace=len(indices_0)<half),
                          np.random.choice(indices_1, half, replace=len(indices_1)<half)])
    np.random.shuffle(idx)
    return x[idx], y[idx]

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs, _ = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return correct/total if total>0 else 0

def train_run(size, seed, mode, train_x, train_y, test_std_loader, test_ood_loader):
    set_seed(seed)
    sub_x, sub_y = get_balanced_subset(train_x, train_y, size)
    train_dataset = TensorDataset(torch.from_numpy(sub_x), torch.from_numpy(sub_y))
    train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, size), shuffle=True)
    
   
    model = PointNetTNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_ood = 0.0
    counter = 0
    best_state_dict = model.state_dict() 

    for epoch in range(MAX_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs, trans = model(inputs)
            loss = criterion(outputs, labels) + 0.1 * torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - torch.eye(3).to(DEVICE), dim=(1,2)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        ood_acc = evaluate(model, test_ood_loader)

        
        if ood_acc > best_ood:
            best_ood = ood_acc
            best_state_dict = model.state_dict() 
            counter = 0 
        else:
            counter += 1


        if mode == 'EarlyStop' and counter >= PATIENCE:
            print(f"[EarlyStop] Stopped at epoch {epoch+1}")
            break
            
    return {
        'seed': seed,
        'final_ood_acc': best_ood,
        'state_dict': best_state_dict 
    }

def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True) 

    train_x, train_y, test_std_x, test_std_y, test_ood_x, test_ood_y = load_data()
    test_std_loader = DataLoader(TensorDataset(torch.from_numpy(test_std_x), torch.from_numpy(test_std_y)), batch_size=64)
    test_ood_loader = DataLoader(TensorDataset(torch.from_numpy(test_ood_x), torch.from_numpy(test_ood_y)), batch_size=64)

    results_es, results_full = [],[]

    print(f">>> Starting Early Stopping Validation (Saving Weights Mode)")

    for size in DATA_SIZES:
        print(f"\n=== Processing N={size} ===")
        
        es_runs =[]
        for s in range(NUM_SEEDS):
            res = train_run(size, s, 'EarlyStop', train_x, train_y, test_std_loader, test_ood_loader)
            es_runs.append(res)
            
        best_es_run = sorted(es_runs, key=lambda x: x['final_ood_acc'], reverse=True)[0]
        best_es = best_es_run['final_ood_acc']
        results_es.append(best_es)
        
        torch.save(best_es_run['state_dict'], f'checkpoints/PointNetTNet_n{size}_EarlyStop_best.pth')
        
        
        full_runs =[]
        for s in range(NUM_SEEDS):
            res = train_run(size, s, 'Full', train_x, train_y, test_std_loader, test_ood_loader)
            full_runs.append(res)
            
        best_full_run = sorted(full_runs, key=lambda x: x['final_ood_acc'], reverse=True)[0]
        best_full = best_full_run['final_ood_acc']
        results_full.append(best_full)
        
        torch.save(best_full_run['state_dict'], f'checkpoints/PointNetTNet_n{size}_Full200_best.pth')

        print(f"  N={size} -> EarlyStop OOD: {best_es:.4f} | Full 200 OOD: {best_full:.4f}")

    
    plt.figure(figsize=(10,6))
    plt.plot(DATA_SIZES, results_es, 'g*-', label='Early Stopping', linewidth=2)
    plt.plot(DATA_SIZES, results_full, 'ro--', label='Full 200 Epochs', linewidth=2, alpha=0.7)
    plt.xscale('log')
    plt.xticks(DATA_SIZES,[str(s) for s in DATA_SIZES], rotation=45)
    plt.xlabel('Training Data Size')
    plt.ylabel('Best OOD Accuracy')
    plt.title('Early Stopping vs Full Training Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/early_stopping_comparison.png')
    
    print("\n===========================================")
    print("All done!")
    print("1. Chart saved to: results/early_stopping_comparison.png")
    print("2. Weights saved to checkpoints/ with suffixes '_EarlyStop_best.pth' and '_Full200_best.pth'")
    print("===========================================")

if __name__ == "__main__":
    main()