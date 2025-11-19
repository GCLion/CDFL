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
        # PRB核心参数（适配纹理分析）
        prb_window_size=5,    # PRB局部分析窗口（奇数，5/7最优）
        prb_freq_cutoff=0.1,  # 周期性计算：低频过滤比例（过滤中心10%低频）
        prb_num_bins=16       # 随机性计算：灰度分箱数（16兼顾分辨率与抗噪）
    ):
        super().__init__()
        self.walk_steps = walk_steps
        self.walk_restart_prob = walk_restart_prob
        self.graph_threshold = graph_threshold
        
        # PRB参数初始化
        self.prb_window = prb_window_size
        self.prb_pad = prb_window_size // 2  # 窗口padding（保持尺寸一致）
        self.prb_freq_cutoff = prb_freq_cutoff
        self.prb_num_bins = prb_num_bins

    def _compute_prb(self, x):
        """
        核心：计算PRB三特征（批量处理，无逐像素循环）
        输出：
            periodicity: (B,1,H,W) 局部纹理周期性（值越高，周期越明显）
            randomness: (B,1,H,W) 局部纹理随机性（值越高，随机越明显）
            prb_residual: (B,1,H,W) 平衡残差（值越高，偏离真实纹理越远）
        """
        B, C, H, W = x.shape
        device = x.device

        # 1. 转为灰度图（纹理分析无需多通道，取RGB均值）
        if C > 1:
            x_gray = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
        else:
            x_gray = x

        # -------------------------- 计算局部周期性（频域分析） --------------------------
        # 提取局部窗口（B,1,window²,H*W）：用unfold批量处理局部窗口
        x_unfold = F.unfold(
            x_gray, 
            kernel_size=self.prb_window, 
            padding=self.prb_pad, 
            stride=1
        ).view(B, 1, self.prb_window**2, -1)  # (B,1,win²,N)，N=H*W

        # 批量傅里叶变换：对每个窗口做FFT
        fft_win = fft.fft2(x_unfold, dim=2)  # (B,1,win²,N)
        fft_shift = fft.fftshift(fft_win, dim=2)  # 低频移至中心
        mag_win = torch.abs(fft_shift)  # 频域幅值

        # 过滤低频（中心cutoff比例内的低频设为0）
        win_center = self.prb_window // 2
        win_radius = self.prb_window // 2
        cutoff_radius = int(win_radius * self.prb_freq_cutoff)
        # 生成低频掩码（B,1,win²,N）
        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.prb_window, device=device),
            torch.arange(self.prb_window, device=device),
            indexing='ij'
        )
        dist = torch.sqrt((y_grid - win_center)**2 + (x_grid - win_center)**2).view(1, 1, -1, 1)
        freq_mask = (dist > cutoff_radius).float()
        mag_filtered = mag_win * freq_mask

        # 周期性：每个窗口的高频最大幅值（归一化）
        mag_max = mag_filtered.max(dim=2, keepdim=True)[0]
        mag_min = mag_filtered.min(dim=2, keepdim=True)[0]
        periodicity = (mag_max - mag_min) / (mag_max + 1e-8)  # (B,1,1,N)
        periodicity = periodicity.view(B, 1, H, W)  # 重塑为特征图（B,1,H,W）

        # -------------------------- 计算局部随机性（熵分析） --------------------------
        x_unfold = F.unfold(
            x_gray, kernel_size=self.prb_window, padding=self.prb_pad, stride=1
        ).view(B, 1, self.prb_window**2, -1)  # 形状: (B, 1, window², H*W)
        
        # 归一化到 [0, 1]（避免除零）
        x_unfold_min = x_unfold.min(dim=2, keepdim=True)[0]
        x_unfold_max = x_unfold.max(dim=2, keepdim=True)[0]
        # 修复：添加epsilon避免分母为零（防止NaN）
        x_unfold_range = x_unfold_max - x_unfold_min
        x_unfold_range = torch.clamp(x_unfold_range, min=1e-8)  # 确保分母≥1e-8
        x_unfold_norm = (x_unfold - x_unfold_min) / x_unfold_range
        
        # 分箱（确保bin索引在有效范围内）
        num_bins = self.prb_num_bins
        x_binned = torch.floor(x_unfold_norm * (num_bins - 1)).clamp(0, num_bins - 1).long()
        # 检查是否有异常索引（调试用）
        if torch.any(x_binned < 0) or torch.any(x_binned >= num_bins):
            print(f"警告：分箱索引超出范围，最小值={x_binned.min()}, 最大值={x_binned.max()}")
        
        # 计算直方图（避免零计数导致的prob异常）
        hist = torch.zeros(B, 1, num_bins, H*W, device=device)
        hist.scatter_add_(2, x_binned, torch.ones_like(x_binned, dtype=torch.float32))
        total = self.prb_window **2  # 窗口内像素总数
        # 修复：添加epsilon避免除零，确保prob在 [0, 1] 范围内
        prob = hist / (total + 1e-8)
        prob = torch.clamp(prob, 0.0, 1.0)  # 强制概率在有效范围
        
        # 修复：过滤无效值前先确保prob中没有NaN/Inf
        if torch.isnan(prob).any() or torch.isinf(prob).any():
            print("警告：prob中存在NaN或Inf，已替换为0")
            prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)

        
        # 计算熵（H = -sum(p*log(p))，过滤0概率）
        prob_valid = prob[prob > 1e-8]
        log_prob = torch.log2(prob_valid)
        entropy = -torch.zeros_like(prob)
        entropy[prob > 1e-8] = prob_valid * log_prob
        entropy = entropy.sum(dim=2, keepdim=True)  # (B,1,1,N)
        # 熵归一化（最大熵为log2(prb_num_bins)）
        max_entropy = torch.log2(torch.tensor(self.prb_num_bins, device=device))
        randomness = entropy / (max_entropy + 1e-8)
        randomness = randomness.view(B, 1, H, W)  # (B,1,H,W)

        # -------------------------- 计算PRB平衡残差 --------------------------
        # 平衡系数 α = 周期性 / (随机性 + ε)
        alpha = periodicity / (randomness + 1e-8)
        # 统计每个样本的真实纹理α分布（过滤3σ异常值）
        alpha_flat = alpha.view(B, -1)
        alpha_mean = alpha_flat.mean(dim=1, keepdim=True).unsqueeze(2).unsqueeze(3)
        alpha_std = alpha_flat.std(dim=1, keepdim=True).unsqueeze(2).unsqueeze(3)
        # 残差：偏离2σ的部分（负残差设为0）
        prb_residual = torch.abs(alpha - alpha_mean) - 2 * alpha_std
        prb_residual = torch.clamp(prb_residual, min=0.0)
        # 残差归一化
        residual_max = prb_residual.view(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        prb_residual = prb_residual / (residual_max + 1e-8)
        # print(periodicity.shape, randomness.shape, prb_residual.shape)
        # torch.Size([16, 1, 32, 32]) torch.Size([16, 1, 32, 32]) torch.Size([16, 1, 32, 32])

        return periodicity, randomness, prb_residual

    def _build_graph(self, x, periodicity, randomness, prb_residual):
        """
        构建图：节点特征 = 原始图像特征 + PRB三特征（周期性、随机性、残差）
        边权重 = 节点特征的余弦相似度（过滤弱连接+自环）
        """
        B, C, H, W = x.shape
        N = H * W  # 节点数 = 像素数

        # 1. 展平所有特征（B, C/H*W → B, N, C）
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        # PRB特征展平（均为单通道，拼接为3通道）
        prb_flat = torch.cat([
            periodicity.view(B, 1, -1).permute(0, 2, 1),
            randomness.view(B, 1, -1).permute(0, 2, 1),
            prb_residual.view(B, 1, -1).permute(0, 2, 1)
        ], dim=2)  # (B, N, 3)

        # 2. 节点特征：原始特征 + PRB特征（维度 C + 3）
        node_features = torch.cat([x_flat, prb_flat], dim=2)  # (B, N, C+3)

        # 3. 计算边权重（余弦相似度）
        node_norm = F.normalize(node_features, dim=2)  # 特征归一化
        adj_matrix = torch.bmm(node_norm, node_norm.transpose(1, 2))  # (B, N, N)：节点间相似度

        # 4. 过滤弱连接（< threshold）和自环（对角线）
        adj_matrix = adj_matrix * (adj_matrix >= self.graph_threshold).float()
        self_loop_mask = torch.eye(N, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        adj_matrix = adj_matrix * (1 - self_loop_mask)  # 移除自环

        # 5. 行归一化（边权重总和为1）
        row_sums = adj_matrix.sum(dim=2, keepdim=True)
        adj_matrix = adj_matrix / (row_sums + 1e-8)

        return adj_matrix, node_features

    def _random_walk(self, adj_matrix):
        """保留原随机游走逻辑：带重启的随机游走，捕捉全局节点关联"""
        B, N, _ = adj_matrix.shape
        # 初始游走概率：单位矩阵（每个节点从自身出发）
        walk_probs = torch.eye(N, device=adj_matrix.device).unsqueeze(0).repeat(B, 1, 1)

        for _ in range(self.walk_steps):
            # 带重启的游走：step_probs = 重启概率*转移概率 + (1-重启概率)*初始概率
            step_probs = self.walk_restart_prob * torch.bmm(walk_probs, adj_matrix)
            step_probs += (1 - self.walk_restart_prob) * torch.eye(N, device=adj_matrix.device).unsqueeze(0).repeat(B, 1, 1)
            walk_probs = step_probs

        return walk_probs

    def _aggregate_features(self, node_features, walk_probs, H, W):
        """保留原特征聚合逻辑：基于游走概率加权聚合全局特征"""
        B, N, C_node = node_features.shape
        # 全局特征 = 游走概率 × 节点特征（B, N, C_node）
        aggregated_feat = torch.bmm(walk_probs, node_features)
        # 重塑为图像维度（B, C_node, H, W）
        aggregated_feat = aggregated_feat.permute(0, 2, 1).view(B, C_node, H, W)
        return aggregated_feat

    def forward(self, x):
        """
        前向流程：PRB特征计算 → 图构建 → 随机游走 → 特征聚合
        输出：PRB三特征（局部） + 全局聚合特征（与原模块接口兼容）
        """
        # 1. 计算PRB核心特征（周期性、随机性、平衡残差）
        periodicity, randomness, prb_residual = self._compute_prb(x)
        
        # 2. 构建图（节点特征含PRB信息）
        adj_matrix, node_features = self._build_graph(x, periodicity, randomness, prb_residual)
        
        # 3. 带重启的随机游走
        walk_probs = self._random_walk(adj_matrix)
        
        # 4. 全局特征聚合
        H, W = x.shape[2], x.shape[3]
        global_feat = self._aggregate_features(node_features, walk_probs, H, W)
        
        # 输出：PRB局部特征（3个单通道） + 全局关联特征（兼容原模块输出格式）(periodicity, randomness, )
        return prb_residual, global_feat

