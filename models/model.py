import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def square_distance(src, dst):
    
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k):
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        
        
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        
        self.k = k
        
    def forward(self, xyz, features):
        """
        xyz: (B, N, 3)
        features: (B, N, d_model)
        """
        dists = square_distance(xyz, xyz)
        
        knn_idx = dists.argsort()[:, :, :self.k]  # (B, N, k)
        
        knn_xyz = index_points(xyz, knn_idx) # (B, N, k, 3)
        
        
        q_xyz = xyz.unsqueeze(2)
        delta = q_xyz - knn_xyz  # (B, N, k, 3)
        delta_enc = self.fc_delta(delta)  # (B, N, k, d_model)
        
        
        knn_features = index_points(features, knn_idx) # (B, N, k, d_model)
        
        
        x_k = knn_features + delta_enc
        
       
        weights = self.fc_gamma(x_k) # (B, N, k, d_model)
        weights = F.softmax(weights, dim=2) # 归一化
        
        
        res = (weights * x_k).sum(dim=2) # (B, N, d_model)
        
        return features + res # 残差连接

class PointTransformer(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        self.k = k
        
        
        self.conv1 = nn.Linear(3, 32)
        self.bn1 = nn.BatchNorm1d(32)
        
       
        self.block1 = TransformerBlock(d_points=32, d_model=32, k=k)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Linear(32, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.block2 = TransformerBlock(d_points=128, d_model=128, k=k)
        
        self.fc1 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 2)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        
        xyz = x.clone()
        

        x = self.conv1(x) 
        x = x.transpose(1, 2) # (B, 32, N)
        x = self.relu(self.bn1(x))
        x = x.transpose(1, 2) # (B, N, 32) 
        
  
        x = self.block1(xyz, x) # (B, N, 32)
        
        
        x = x.transpose(1, 2)
        x = self.relu(self.bn2(x))
        x = x.transpose(1, 2)
        
       
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.relu(self.bn3(x))
        x = x.transpose(1, 2) # (B, N, 128)
        
        x = self.block2(xyz, x)
        
      
        x = x.mean(dim=1) # (B, N, 128) -> (B, 128)
        
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()   
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature 

class DGCNN(nn.Module):
    def __init__(self, k=20):
        super(DGCNN, self).__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.bn4 = nn.BatchNorm1d(1024)
        self.conv4 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 2)

    def forward(self, x):
        x = x.transpose(2, 1)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      
        x = self.conv1(x)                       
        x1 = x.max(dim=-1, keepdim=False)[0]    
        x = get_graph_feature(x1, k=self.k)     
        x = self.conv2(x)                       
        x2 = x.max(dim=-1, keepdim=False)[0]    
        x = get_graph_feature(x2, k=self.k)     
        x = self.conv3(x)                       
        x3 = x.max(dim=-1, keepdim=False)[0]    
        x = torch.cat((x1, x2, x3), dim=1)      
        x = self.conv4(x)                       
        x = x.max(dim=-1, keepdim=False)[0]     
        x = F.leaky_relu(self.bn5(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn6(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(3, dtype=x.dtype, device=x.device).view(1, 9).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class SimplePointNet(nn.Module):
    def __init__(self):
        super(SimplePointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.transpose(2, 1) 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x

class PointNetTNet(nn.Module):
    def __init__(self):
        super(PointNetTNet, self).__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x_t = x.transpose(2, 1) 
        trans = self.stn(x_t)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1) 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x, trans
    
def farthest_point_sample(xyz, npoint):

    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):

    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
   
    sqrdists = square_distance(new_xyz, xyz) 
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):

    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # (B, npoint)
    new_xyz = index_points(xyz, fps_idx)         # (B, npoint, 3)
    
    idx = query_ball_point(radius, nsample, xyz, new_xyz) # (B, npoint, nsample)
    grouped_xyz = index_points(xyz, idx) # (B, npoint, nsample, 3)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, 3)
    
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # (B, npoint, nsample, 3+D)
    else:
        new_points = grouped_xyz_norm
        
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):

        if self.group_all:
            new_xyz = torch.zeros(xyz.size(0), 1, 3).to(xyz.device)
            grouped_points = (xyz if points is None else torch.cat([xyz, points], dim=-1)).unsqueeze(1)
        else:
            new_xyz, grouped_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
            
        
        new_points = grouped_points.permute(0, 3, 2, 1) 
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0] 
        new_points = new_points.transpose(1, 2)  
        return new_xyz, new_points

class PointNet2(nn.Module):
    def __init__(self):
        super(PointNet2, self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)

        self.sa2 = PointNetSetAbstraction(npoint=32, radius=0.4, nsample=32, in_channel=128+3, mlp=[128, 128, 256], group_all=False)

        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        xyz = x 
        points = None
        

        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        

        x = l3_points.view(-1, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels) / np.sqrt(in_channels))
    def forward(self, x):
        
        return torch.einsum('oi,bidn->bodn', self.weight, x)

class VNBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
    def forward(self, x):
       
        norm = torch.norm(x, dim=2) + 1e-6 # (B, C, N)
        norm_bn = self.bn(norm) # (B, C, N)
        return x * (norm_bn / norm).unsqueeze(2)

class VNReLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_channels, in_channels) / np.sqrt(in_channels))
        self.U = nn.Parameter(torch.randn(in_channels, in_channels) / np.sqrt(in_channels))
    def forward(self, x):
        
        q = torch.einsum('oi,bidn->bodn', self.W, x)
        k = torch.einsum('oi,bidn->bodn', self.U, x)
        inner = torch.sum(q * k, dim=2, keepdim=True) # (B, C, 1, N)
        mask = (inner < 0).float()
        k_norm2 = torch.sum(k * k, dim=2, keepdim=True) + 1e-6
        proj = (inner / k_norm2) * k
        return x - mask * proj

class VNInvariant(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.v1_weights = nn.Parameter(torch.randn(in_channels))
        self.v2_weights = nn.Parameter(torch.randn(in_channels))
    def forward(self, x):

        v1 = (x * self.v1_weights.view(1, -1, 1)).sum(dim=1) # (B, 3)
        v2 = (x * self.v2_weights.view(1, -1, 1)).sum(dim=1) # (B, 3)
        
        e1 = F.normalize(v1, dim=1, eps=1e-6)
        u2 = v2 - torch.sum(v2 * e1, dim=1, keepdim=True) * e1
        e2 = F.normalize(u2, dim=1, eps=1e-6)
        e3 = torch.cross(e1, e2, dim=1)
        
        R = torch.stack([e1, e2, e3], dim=1) # (B, 3, 3)
        x_inv = torch.bmm(x, R.transpose(1, 2)) # (B, C, 3)
        return x_inv.view(x.size(0), -1) # (B, C*3)

class VNPointNet(nn.Module):
    def __init__(self):
        super(VNPointNet, self).__init__()
        self.conv1 = VNLinear(1, 64)
        self.bn1 = VNBatchNorm(64)
        self.relu1 = VNReLU(64)
        
        self.conv2 = VNLinear(64, 128)
        self.bn2 = VNBatchNorm(128)
        self.relu2 = VNReLU(128)
        
        self.conv3 = VNLinear(128, 1024)
        self.bn3 = VNBatchNorm(1024)
        
        self.inv = VNInvariant(1024)
        
        self.fc1 = nn.Linear(1024 * 3, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 2)
        
    def forward(self, x):

        centroid = x.mean(dim=1, keepdim=True)
        x = x - centroid
        

        x = x.transpose(1, 2).unsqueeze(1) 
        
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        

        x = x.mean(dim=-1) 
        x = self.inv(x)   
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x