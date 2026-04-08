import numpy as np
import os
from tqdm import tqdm

def get_rotation_matrix(limit_angle_deg):
    """
    生成一个随机的 XYZ 三轴旋转矩阵
    limit_angle_deg: 限制的最大角度（度）
    """
    limit = limit_angle_deg * (np.pi / 180)
    
    # 在 -limit 到 +limit 之间随机采样三个角度
    theta = np.random.uniform(-limit, limit) # Z轴
    phi   = np.random.uniform(-limit, limit) # Y轴
    psi   = np.random.uniform(-limit, limit) # X轴
    
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    
    Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                   [0, 1, 0],
                   [-np.sin(phi), 0, np.cos(phi)]])
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(psi), -np.sin(psi)],
                   [0, np.sin(psi), np.cos(psi)]])
    
    # 矩阵乘法顺序：Rz * Ry * Rx
    return Rz @ Ry @ Rx

def generate_sphere(num_points=1024, radius=1.0, center=[0,0,0]):
    """
    生成标准正球体
    注意：正球体旋转后形状不变，所以这里不需要乘旋转矩阵
    """
    points = np.random.randn(num_points, 3)
    points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
    points = points * radius
    
    # 加上少量噪声
    noise = np.random.normal(0, 0.01, points.shape)
    points += noise
    
    points += center
    return points

def generate_cube(num_points=1024, size=1.0, center=[0,0,0]):
    """
    生成标准正方体，并进行 XYZ < 15度的旋转
    """
    points = np.random.rand(num_points, 3) - 0.5
    points *= size
    
    # === 关键配置：XYZ 三轴旋转，最大 15 度 ===
    R = get_rotation_matrix(limit_angle_deg=15)
    points = np.dot(points, R)
    # =======================================

    points += center
    return points

def create_dataset(num_samples, mode='train'):
    data = []
    labels = [] 
    
    num_points = 512
    print(f"Generating {mode} dataset (Std Sphere/Cube, XYZ<15°, {num_points} pts)...")
    
    for _ in tqdm(range(num_samples)):
        label = np.random.randint(0, 2)
        
        
        scale = np.random.uniform(0.8, 1.2)
        
        y_shift = np.random.uniform(-2, 2)
        z_shift = np.random.uniform(-2, 2)
        

        if mode in ['train', 'test_std']:
            if label == 0: x_shift = np.random.uniform(-3, 2.0) 
            else:          x_shift = np.random.uniform(-2.0, 3)
        elif mode == 'test_ood':
            if label == 0: x_shift = np.random.uniform(2.5, 5)
            else:          x_shift = np.random.uniform(-5, -2.5)
        
        center = [x_shift, y_shift, z_shift]
        
        if label == 0:
            pcl = generate_sphere(num_points=num_points, center=center, radius=scale/2)
        else:
            pcl = generate_cube(num_points=num_points, center=center, size=scale)
            
        data.append(pcl)
        labels.append(label)
        
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64)

def prepare_data():
    if os.path.exists('data'):
        import shutil
        try: shutil.rmtree('data'); print("Old data removed."); 
        except: pass
    os.makedirs('data', exist_ok=True)
    
    train_x, train_y = create_dataset(20000, mode='train')
    np.save('data/train_x.npy', train_x)
    np.save('data/train_y.npy', train_y)
    
    test_std_x, test_std_y = create_dataset(1000, mode='test_std')
    np.save('data/test_std_x.npy', test_std_x)
    np.save('data/test_std_y.npy', test_std_y)
    
    test_ood_x, test_ood_y = create_dataset(1000, mode='test_ood')
    np.save('data/test_ood_x.npy', test_ood_x)
    np.save('data/test_ood_y.npy', test_ood_y)

if __name__ == "__main__":
    prepare_data()