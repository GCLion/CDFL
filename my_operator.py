import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

class PRB(nn.Module):
    def __init__(
        self, 
        walk_steps=3, 
        walk_restart_prob=0.7, 
        graph_threshold=0.5,
        prb_window_size=5,   
        prb_freq_cutoff=0.1,  
        prb_num_bins=16       
    ):
        super().__init__()
        self.walk_steps = walk_steps
        self.walk_restart_prob = walk_restart_prob
        self.graph_threshold = graph_threshold
        
        # PRB参数初始化
        self.prb_window = prb_window_size
        self.prb_pad = prb_window_size // 2  
        self.prb_freq_cutoff = prb_freq_cutoff
        self.prb_num_bins = prb_num_bins

    def _compute_prb(self, x): 
        B, C, H, W = x.shape
        device = x.device

        if C > 1:
            x_gray = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
        else:
            x_gray = x

        x_unfold = F.unfold(
            x_gray, 
            kernel_size=self.prb_window, 
            padding=self.prb_pad, 
            stride=1
        ).view(B, 1, self.prb_window**2, -1)  # (B,1,win²,N)，N=H*W

        fft_win = fft.fft2(x_unfold, dim=2)  # (B,1,win²,N)
        fft_shift = fft.fftshift(fft_win, dim=2)  #  
        mag_win = torch.abs(fft_shift)  #  

        win_center = self.prb_window // 2
        win_radius = self.prb_window // 2
        cutoff_radius = int(win_radius * self.prb_freq_cutoff)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.prb_window, device=device),
            torch.arange(self.prb_window, device=device),
            indexing='ij'
        )
        dist = torch.sqrt((y_grid - win_center)**2 + (x_grid - win_center)**2).view(1, 1, -1, 1)
        freq_mask = (dist > cutoff_radius).float()
        mag_filtered = mag_win * freq_mask

        mag_max = mag_filtered.max(dim=2, keepdim=True)[0]
        mag_min = mag_filtered.min(dim=2, keepdim=True)[0]
        periodicity = (mag_max - mag_min) / (mag_max + 1e-8)  # (B,1,1,N)
        periodicity = periodicity.view(B, 1, H, W)  # 

        x_unfold = F.unfold(
            x_gray, kernel_size=self.prb_window, padding=self.prb_pad, stride=1
        ).view(B, 1, self.prb_window**2, -1)  #  
        
        x_unfold_min = x_unfold.min(dim=2, keepdim=True)[0]
        x_unfold_max = x_unfold.max(dim=2, keepdim=True)[0]
        x_unfold_range = x_unfold_max - x_unfold_min
        x_unfold_range = torch.clamp(x_unfold_range, min=1e-8)  # 
        x_unfold_norm = (x_unfold - x_unfold_min) / x_unfold_range
        
        num_bins = self.prb_num_bins
        x_binned = torch.floor(x_unfold_norm * (num_bins - 1)).clamp(0, num_bins - 1).long() 
        
        hist = torch.zeros(B, 1, num_bins, H*W, device=device)
        hist.scatter_add_(2, x_binned, torch.ones_like(x_binned, dtype=torch.float32))
        total = self.prb_window **2  #  
        prob = hist / (total + 1e-8)
        prob = torch.clamp(prob, 0.0, 1.0)  #  
        
        if torch.isnan(prob).any() or torch.isinf(prob).any():
            prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)

        
        prob_valid = prob[prob > 1e-8]
        log_prob = torch.log2(prob_valid)
        entropy = -torch.zeros_like(prob)
        entropy[prob > 1e-8] = prob_valid * log_prob
        entropy = entropy.sum(dim=2, keepdim=True)  # (B,1,1,N)
        max_entropy = torch.log2(torch.tensor(self.prb_num_bins, device=device))
        randomness = entropy / (max_entropy + 1e-8)
        randomness = randomness.view(B, 1, H, W)  # (B,1,H,W)

        alpha = periodicity / (randomness + 1e-8)
        alpha_flat = alpha.view(B, -1)
        alpha_mean = alpha_flat.mean(dim=1, keepdim=True).unsqueeze(2).unsqueeze(3)
        alpha_std = alpha_flat.std(dim=1, keepdim=True).unsqueeze(2).unsqueeze(3)
        prb_residual = torch.abs(alpha - alpha_mean) - 2 * alpha_std
        prb_residual = torch.clamp(prb_residual, min=0.0)
        residual_max = prb_residual.view(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        prb_residual = prb_residual / (residual_max + 1e-8)
        # print(periodicity.shape, randomness.shape, prb_residual.shape)
        # torch.Size([16, 1, 32, 32]) torch.Size([16, 1, 32, 32]) torch.Size([16, 1, 32, 32])

        return periodicity, randomness, prb_residual

    def _build_graph(self, x, periodicity, randomness, prb_residual): 
        B, C, H, W = x.shape
        N = H * W  #  

        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        prb_flat = torch.cat([
            periodicity.view(B, 1, -1).permute(0, 2, 1),
            randomness.view(B, 1, -1).permute(0, 2, 1),
            prb_residual.view(B, 1, -1).permute(0, 2, 1)
        ], dim=2)  # (B, N, 3)

        node_features = torch.cat([x_flat, prb_flat], dim=2)  # (B, N, C+3)
        node_norm = F.normalize(node_features, dim=2)  #  
        adj_matrix = torch.bmm(node_norm, node_norm.transpose(1, 2))  #  

        adj_matrix = adj_matrix * (adj_matrix >= self.graph_threshold).float()
        self_loop_mask = torch.eye(N, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        adj_matrix = adj_matrix * (1 - self_loop_mask)  # 

        row_sums = adj_matrix.sum(dim=2, keepdim=True)
        adj_matrix = adj_matrix / (row_sums + 1e-8)

        return adj_matrix, node_features

    def _random_walk(self, adj_matrix):
        B, N, _ = adj_matrix.shape
        walk_probs = torch.eye(N, device=adj_matrix.device).unsqueeze(0).repeat(B, 1, 1)

        for _ in range(self.walk_steps):
            step_probs = self.walk_restart_prob * torch.bmm(walk_probs, adj_matrix)
            step_probs += (1 - self.walk_restart_prob) * torch.eye(N, device=adj_matrix.device).unsqueeze(0).repeat(B, 1, 1)
            walk_probs = step_probs

        return walk_probs

    def _aggregate_features(self, node_features, walk_probs, H, W):
        B, N, C_node = node_features.shape
        aggregated_feat = torch.bmm(walk_probs, node_features)
        aggregated_feat = aggregated_feat.permute(0, 2, 1).view(B, C_node, H, W)
        return aggregated_feat

    def forward(self, x):
        periodicity, randomness, prb_residual = self._compute_prb(x)
        adj_matrix, node_features = self._build_graph(x, periodicity, randomness, prb_residual)
        walk_probs = self._random_walk(adj_matrix)
        H, W = x.shape[2], x.shape[3]
        global_feat = self._aggregate_features(node_features, walk_probs, H, W)
        
        return prb_residual, global_feat

