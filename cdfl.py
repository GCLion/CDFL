import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from base import BaseSegmentor
from backbones import BuildActivation, BuildNormalization, constructnormcfg
from builder import BuildBackbone
from my_operator import PRB


class MSCCNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(MSCCNet, self).__init__(cfg, mode)
        norm_cfg = {'type': 'batchnorm2d'}
        act_cfg = {'type': 'relu', 'inplace': True}
        head_cfg = {
            'in_channels': 2048,
            'feats_channels': 512,
            'out_channels': 512,
            'dropout': 0.1,
            'spectral_height': 64,
            'spectral_width': 64,
            'spectral_k': 8,
        }

        # build bottleneck
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=512, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(512*2, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=512, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.bottleneck3 = nn.Sequential(
            nn.Conv2d(512*4, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=512, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        # build fusion bottleneck
        self.bottleneck_fusion = nn.Sequential(
            nn.Conv2d(512*3, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=512, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        # build decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(head_cfg['out_channels'], head_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )

        # # build auxiliary decoder
        backbone_cfg = copy.deepcopy(cfg['backbone'])
        if 'norm_cfg' not in backbone_cfg:
            backbone_cfg.update({'norm_cfg': copy.deepcopy(self.norm_cfg)})
        self.backbone_net = BuildBackbone(backbone_cfg)
    
    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True),
                             scale_factor=1 / factor, mode='nearest', recompute_scale_factor=True)
        
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        # NPR = x - self.interpolate(x, 0.5)
        # x = NPR * 2.0 / 3.0
        # img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to classifier
        # predictions_img = self.classifer(backbone_outputs[-1])
        # multi-level features fusion
        x1 = backbone_outputs[1]
        x1 = self.bottleneck1(x1)
        x2 = backbone_outputs[2]
        x2 = self.bottleneck2(x2)
        x3 = backbone_outputs[3]
        x3 = self.bottleneck3(x3)
        x123 = self.bottleneck_fusion(torch.cat((x1, x2, x3), dim=1))
        
        return x123

# -------------------------- 1. 分块局部自注意力（复用优化后的版本） --------------------------
class ChunkedLocalSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=1, local_window=16, chunk_size=256, layer_name=""):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.local_window = local_window
        self.chunk_size = chunk_size
        self.layer_name = layer_name
        
        # 验证通道数与头数匹配（单头维度≥8）
        assert in_channels % num_heads == 0, f"[{layer_name}] 通道数{in_channels}不是头数{num_heads}的整数倍"
        assert (in_channels // num_heads) >= 8, f"[{layer_name}] 单头维度{in_channels//num_heads}过小（需≥8）"
        
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,  # 嵌入维度=通道数
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True
        )
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.shape
        # 确保输入通道与层配置一致
        assert C == self.in_channels, f"[{self.layer_name}] 输入通道不匹配：实际{C}，预期{self.in_channels}"
        
        N = H * W  # 像素总数
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # 转为 (B, N, C) 适配注意力输入

        # 生成像素坐标（用于局部窗口掩码）
        h_coords = torch.arange(H, device=x.device).repeat_interleave(W)  # (N,)
        w_coords = torch.arange(W, device=x.device).repeat(H)  # (N,)
        half_win = self.local_window // 2

        # 分块处理注意力（降低内存占用）
        num_chunks = (N + self.chunk_size - 1) // self.chunk_size
        attn_out_list = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min((chunk_idx + 1) * self.chunk_size, N)
            chunk_size = end - start

            # 当前块像素坐标
            chunk_h = h_coords[start:end]
            chunk_w = w_coords[start:end]

            # 计算局部窗口范围（确保不越界）
            h_start = torch.clamp(chunk_h - half_win, 0, H)
            h_end = torch.clamp(chunk_h + half_win + 1, 0, H)
            w_start = torch.clamp(chunk_w - half_win, 0, W)
            w_end = torch.clamp(chunk_w + half_win + 1, 0, W)

            # 生成局部窗口掩码（仅窗口内像素参与注意力计算）
            j_h = h_coords.unsqueeze(0).repeat(chunk_size, 1)  # (chunk_size, N)
            j_w = w_coords.unsqueeze(0).repeat(chunk_size, 1)
            in_h = (j_h >= h_start.unsqueeze(1)) & (j_h < h_end.unsqueeze(1))
            in_w = (j_w >= w_start.unsqueeze(1)) & (j_w < w_end.unsqueeze(1))
            in_window = in_h & in_w

            chunk_mask = torch.full((chunk_size, N), -1e9, device=x.device)
            chunk_mask[in_window] = 0  # 窗口内像素掩码为0（不衰减）

            # 计算注意力
            chunk_query = x_flat[:, start:end, :]  # (B, chunk_size, C)
            chunk_key = x_flat  # (B, N, C)
            chunk_value = x_flat  # (B, N, C)

            chunk_attn_out, _ = self.attention(
                query=chunk_query,
                key=chunk_key,
                value=chunk_value,
                attn_mask=chunk_mask,
                need_weights=False
            )
            attn_out_list.append(chunk_attn_out)

        # 拼接所有块的输出并重塑为特征图
        attn_out = torch.cat(attn_out_list, dim=1)  # (B, N, C)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)
        
        # 残差连接 + 投影
        return self.act(self.proj(attn_out + x)), None

# -------------------------- 2. Encoder（适配512通道输入，32x32特征） --------------------------
class LightEncoder(nn.Module):
    def __init__(self, in_channels=512, num_layers=3, local_window=16, chunk_size=256):
        super().__init__()
        # 通道序列：从512逐步降维，每次下采样保持合理维度
        # 32x32 → 16x16 → 8x8 → 4x4（3次下采样，与Decoder上采样次数对应）
        self.channels = [512, 256, 128]  # 每一层的通道数
        self.num_layers = num_layers

        # 头数配置：确保通道数/头数 = 单头维度（64，合理且高效）
        # 512/8=64，256/4=64，128/2=64
        self.num_heads = [8, 4, 2]

        # 验证通道数与头数匹配
        for c, h in zip(self.channels, self.num_heads):
            assert c % h == 0, f"Encoder通道{c}不是头数{h}的整数倍"

        # 初始化卷积（如果输入通道与第一层通道不一致，此处可调整，当前输入=512，与第一层一致）
        self.init_conv = nn.Conv2d(
            in_channels, self.channels[0], kernel_size=1, padding=0, stride=1, bias=True
        )
        self.init_act = nn.ReLU()

        # 编码器层：注意力 + 下采样（每次下采样尺寸减半）
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # 下采样输出通道 = 下一层的输入通道（最后一层保持通道）
            if i < num_layers - 1:
                downsample_out = self.channels[i+1]
            else:
                downsample_out = self.channels[i]

            self.layers.append(nn.ModuleDict({
                'attn': ChunkedLocalSelfAttention(
                    in_channels=self.channels[i],
                    num_heads=self.num_heads[i],
                    local_window=local_window,
                    chunk_size=chunk_size,
                    layer_name=f"Encoder_Layer_{i+1}"
                ),
                'downsample': nn.Conv2d(
                    self.channels[i],
                    downsample_out,
                    kernel_size=3,  # 3x3卷积下采样（比1x1保留更多空间信息）
                    stride=2,
                    padding=1,
                    bias=True
                )
            }))

    def forward(self, x):
        # 输入x形状：(B, 512, 32, 32)
        B, C, H, W = x.shape
        assert C == self.channels[0], f"Encoder输入通道不匹配：实际{C}，预期{self.channels[0]}"

        # 初始特征调整（当前输入通道已匹配，仍保留以确保灵活性）
        x = self.init_act(self.init_conv(x))  # (B, 512, 32, 32)

        # 逐层下采样
        encoder_feats = []  # 可选：保存中间特征用于跳跃连接（当前未使用）
        for i, layer in enumerate(self.layers):
            x, _ = layer['attn'](x)
            encoder_feats.append(x)
            x = layer['downsample'](x)  # 尺寸减半：32→16→8→4

        # 输出形状：(B, 128, 4, 4)（经过3次下采样）
        return x, encoder_feats

# -------------------------- 3. Decoder（从4x4上采样到256x256） --------------------------
class LightDecoder(nn.Module):
    def __init__(self, num_layers=3, out_channels=1, local_window=16, chunk_size=256):
        super().__init__()
        # 通道序列：与Encoder对称（128→256→512）
        self.channels = [128, 256, 512]
        # 头数配置：与Encoder对称（2→4→8），确保单头维度64
        self.num_heads = [2, 4, 8]
        self.num_layers = num_layers

        # 验证通道数与头数匹配
        for c, h in zip(self.channels, self.num_heads):
            assert c % h == 0, f"Decoder通道{c}不是头数{h}的整数倍"

        # 解码器层：上采样 + 注意力（每次上采样尺寸翻倍）
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # 上采样输出通道 = 下一层的输入通道（最后一层保持通道）
            if i < num_layers - 1:
                upsample_out = self.channels[i+1]
            else:
                upsample_out = self.channels[i]

            self.layers.append(nn.ModuleDict({
                'upsample': nn.ConvTranspose2d(
                    self.channels[i],
                    upsample_out,
                    kernel_size=3,  # 3x3转置卷积上采样（保留更多空间信息）
                    stride=2,
                    padding=1,
                    output_padding=1,  # 确保尺寸翻倍（如4→8，8→16）
                    bias=True
                ),
                'attn': ChunkedLocalSelfAttention(
                    in_channels=upsample_out,
                    num_heads=self.num_heads[i],
                    local_window=local_window,
                    chunk_size=chunk_size,
                    layer_name=f"Decoder_Layer_{i+1}"
                ),
                'fuse': nn.Conv2d(upsample_out, upsample_out, kernel_size=1, bias=True)  # 特征融合
            }))

        # 最终卷积：将512通道压缩为输出通道（1通道掩码）
        self.final_conv = nn.Conv2d(self.channels[-1], out_channels, kernel_size=1, bias=True)
        # 激活函数（确保输出在[0,1]范围）
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, encoder_feats=None):
        # 输入x形状：(B, 128, 4, 4)（Encoder的输出）
        B, C, H, W = x.shape
        assert C == self.channels[0], f"Decoder输入通道不匹配：实际{C}，预期{self.channels[0]}"

        # 逐层上采样（4→8→16→32，最后额外上采样到256）
        for i, layer in enumerate(self.layers):
            x = layer['upsample'](x)  # 尺寸翻倍：4→8→16→32
            x = layer['fuse'](x)
            x, _ = layer['attn'](x)

        # 此时x形状：(B, 512, 32, 32)，需要上采样到256x256
        # 计算上采样倍数：256 / 32 = 8倍
        x = F.interpolate(
            x,
            size=(256, 256),
            mode='bilinear',
            align_corners=True  # 确保边缘对齐
        )  # (B, 512, 256, 256)

        # 输出掩码（单通道，移除通道维）
        mask_logits = self.sigmoid(self.final_conv(x))  # (B, 1, 256, 256)
        return mask_logits.squeeze(1)  # (B, 256, 256)






class DiffAttention2D(nn.Module):
    def __init__(self, channels, heads=8):
        super().__init__()
        assert channels % heads == 0
        self.heads = heads
        self.d = channels // heads
        self.to_q1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_k1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_q2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_k2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)
        self.unify = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q1 = self.to_q1(x).view(B, self.heads, self.d, -1)
        k1 = self.to_k1(x).view(B, self.heads, self.d, -1)
        q2 = self.to_q2(x).view(B, self.heads, self.d, -1)
        k2 = self.to_k2(x).view(B, self.heads, self.d, -1)
        v = self.to_v(x).view(B, self.heads, self.d, -1)

        attn1 = torch.einsum('bhcn,bhcm->bhnm', q1, k1) / (self.d**0.5)
        attn2 = torch.einsum('bhcn,bhcm->bhnm', q2, k2) / (self.d**0.5)

        a1 = F.softmax(attn1, dim=-1)
        a2 = F.softmax(attn2, dim=-1)

        diff_attn = a1 - a2
        out = torch.einsum('bhnm,bhcm->bhcn', diff_attn, v)
        out = out.reshape(B, C, H, W)
        return self.unify(out) + x  # 残差连接

class ConvRefine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.refine(x) + x  # 再次残差连接





# -------------------------- 1. 自注意力模块（适配不同通道数） --------------------------
class SelfAttention(nn.Module):
    """用于下采样层的自注意力模块，输出注意力加权特征和注意力图"""
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        assert in_channels % num_heads == 0, f"通道数{in_channels}必须是头数{num_heads}的整数倍"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads  # 单头维度
        
        # 线性投影（将通道分拆为多头）
        self.qkv_proj = nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, bias=False)
        # 输出投影（将多头特征合并）
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W  # 像素总数（序列长度）
        
        # 1. 计算Q、K、V（B, 3*C, H, W）→ 分拆为Q/K/V（各B, C, H, W）
        qkv = self.qkv_proj(x).view(B, 3, self.num_heads, self.head_dim, H, W)  # (B, 3, heads, head_dim, H, W)
        q, k, v = qkv.unbind(1)  # 分别得到Q/K/V：(B, heads, head_dim, H, W)
        
        # 2. 重塑为序列形式（B, heads, head_dim, N）
        q = q.flatten(3)  # (B, heads, head_dim, N)
        k = k.flatten(3)  # (B, heads, head_dim, N)
        v = v.flatten(3)  # (B, heads, head_dim, N)
        
        # 3. 计算注意力权重：(B, heads, N, N)
        attn = torch.einsum('bhdn, bhdm -> bhnm', q, k)  # Q·K^T
        attn = attn * (self.head_dim **-0.5)  # 缩放
        attn = self.softmax(attn)  # 归一化
        
        # 4. 注意力加权：(B, heads, head_dim, N)
        out = torch.einsum('bhnm, bhdm -> bhdn', attn, v)  # 加权V
        
        # 关键修复：用reshape替代view（处理内存不连续问题）
        out = out.reshape(B, C, H, W)  # 重塑为特征图：(B, C, H, W)  <-- 这里修改
        
        # 5. 输出投影 + 残差连接
        out = self.out_proj(out) + x
        return out, attn  # 返回注意力加权特征和注意力图（用于反哺）


# -------------------------- 2. 原有模块保持不变（DoubleConv/Down/Up/OutConv） --------------------------
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv（彻底修复通道数匹配问题）"""
    def __init__(self, in_channels_x1, x2_channels, out_channels, bilinear=True, attn_channels=None):
        super().__init__()
        self.bilinear = bilinear
        self.attn_channels = attn_channels  # 注意力特征原始通道数
        
        # 1. 上采样操作（明确x1的通道数）
        if bilinear:
            # 双线性上采样：x1通道数 = 输入x1的通道数（in_channels_x1）
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.x1_channels = in_channels_x1  # 关键：直接使用x1的真实通道数
        else:
            # 转置卷积：x1通道数 = in_channels_x1 // 2（通道数减半）
            self.up = nn.ConvTranspose2d(in_channels_x1, in_channels_x1 // 2, kernel_size=2, stride=2)
            self.x1_channels = in_channels_x1 // 2
        
        # 2. 注意力特征投影（通道数调整为out_channels//2，避免通道占比过高）
        self.attn_proj = nn.Conv2d(attn_channels, out_channels // 2, kernel_size=1, bias=False)
        self.attn_proj_channels = out_channels // 2  # 投影后注意力通道数
        
        # 3. 计算拼接后的总通道数（x1 + x2 + 投影后注意力）
        self.total_channels = self.x1_channels + x2_channels + self.attn_proj_channels
        
        # 4. 定义DoubleConv（输入通道数=总通道数）
        self.conv = DoubleConv(
            in_channels=self.total_channels,
            out_channels=out_channels,
            mid_channels=self.total_channels // 2  # 中间通道数为总通道的一半
        )

    def forward(self, x1, x2, attn_feat):
        # 1. 上采样x1（保持通道数）
        x1 = self.up(x1)
        
        # 2. 尺寸对齐
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # 3. 投影注意力特征
        attn_feat_proj = self.attn_proj(attn_feat)  # 通道数=attn_proj_channels
        
        # 4. 拼接所有特征（x1 + x2 + 投影后注意力）
        x = torch.cat([x1, x2, attn_feat_proj], dim=1)  # 总通道数=total_channels
        
        # 5. 卷积融合
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


# -------------------------- 3. 改进后的UNet（带注意力反哺） --------------------------
class AttentionUNet(nn.Module):
    def __init__(self, n_channels=512, n_classes=1, bilinear=True, target_size=(256, 256)):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.target_size = target_size
        factor = 2 if bilinear else 1  # 双线性模式下通道数缩放因子
        
        # -------------------------- 编码器通道数（关键！必须明确各层输出通道） --------------------------
        # x1：初始卷积输出
        self.x1_channels = 64
        # x2：down1输出（64→128）
        self.x2_channels = 128
        # x3：down2输出（128→256）
        self.x3_channels = 256
        # x4：down3输出（256→256，因512//factor=256）
        self.x4_channels = 512 // factor
        
        # -------------------------- 编码器模块 --------------------------
        self.inc = DoubleConv(n_channels, self.x1_channels)
        self.down1 = Down(self.x1_channels, self.x2_channels)
        self.attn1 = SelfAttention(in_channels=self.x1_channels, num_heads=4)  # 64→4头
        
        self.down2 = Down(self.x2_channels, self.x3_channels)
        self.attn2 = SelfAttention(in_channels=self.x2_channels, num_heads=4)  # 128→4头
        
        self.down3 = Down(self.x3_channels, self.x4_channels)
        self.attn3 = SelfAttention(in_channels=self.x3_channels, num_heads=8)  # 256→8头
        
        self.attn4 = SelfAttention(in_channels=self.x4_channels, num_heads=8)  # 256→8头
        
        # -------------------------- 解码器模块（按实际通道数传递参数） --------------------------
        # up1：输入x1为x4_attn（256通道），x2为x3（256通道），输出128通道
        self.up1 = Up(
            in_channels_x1=self.x4_channels,  # x4_attn的通道数=256
            x2_channels=self.x3_channels,     # x3的通道数=256
            out_channels=self.x3_channels // factor,  # 256//2=128
            bilinear=bilinear,
            attn_channels=self.x3_channels    # x3_attn的通道数=256
        )
        
        # up2：输入x1为up1输出（128通道），x2为x2（128通道），输出64通道
        self.up2 = Up(
            in_channels_x1=self.x3_channels // factor,  # 128
            x2_channels=self.x2_channels,               # x2的通道数=128
            out_channels=self.x2_channels // factor,    # 128//2=64
            bilinear=bilinear,
            attn_channels=self.x2_channels              # x2_attn的通道数=128
        )
        
        # up3：输入x1为up2输出（64通道），x2为x1（64通道），输出64通道
        self.up3 = Up(
            in_channels_x1=self.x2_channels // factor,  # 64
            x2_channels=self.x1_channels,               # x1的通道数=64
            out_channels=self.x1_channels,              # 64
            bilinear=bilinear,
            attn_channels=self.x1_channels              # x1_attn的通道数=64
        )
        
        # 输出层
        self.outc = OutConv(self.x1_channels, n_classes)
        self.final_up = nn.Upsample(size=target_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        # 编码器前向传播（通道数与定义一致）
        x1 = self.inc(x)  # (B, 64, H, W)
        x1_attn, _ = self.attn1(x1)  # (B, 64, H, W)
        
        x2 = self.down1(x1)  # (B, 128, H/2, W/2)
        x2_attn, _ = self.attn2(x2)  # (B, 128, H/2, W/2)
        
        x3 = self.down2(x2)  # (B, 256, H/4, W/4)
        x3_attn, _ = self.attn3(x3)  # (B, 256, H/4, W/4)
        
        x4 = self.down3(x3)  # (B, 256, H/8, W/8)
        x4_attn, _ = self.attn4(x4)  # (B, 256, H/8, W/8)
        
        # 解码器前向传播（特征拼接通道数匹配）
        x = self.up1(x4_attn, x3, x3_attn)  # (B, 128, H/4, W/4)
        x = self.up2(x, x2, x2_attn)        # (B, 64, H/2, W/2)
        x = self.up3(x, x1, x1_attn)        # (B, 64, H, W)
        
        # 输出掩码
        logits = self.outc(x)  # (B, 1, H, W)
        logits = self.final_up(logits)  # (B, 1, 256, 256)
        return logits.squeeze(1)  # (B, 256, 256)
        

# -------------------------- 4. 替换后的FakeLocator（核心部分） --------------------------
class FakeLocator(nn.Module):
    def __init__(self, in_channels=512, num_classes=1):
        super().__init__()
        cfg = {
            'type': 'fcn',
            'num_classes': 1,
            'benchmark': True,
            'align_corners': False,
            'backend': 'nccl',
            'norm_cfg': {'type': 'batchnorm2d'},
            'act_cfg': {'type': 'relu', 'inplace': True},
            'backbone': {
                'type': 'resnet101',
                'series': 'resnet',
                'pretrained': True,
                'outstride': 8,
                'use_stem': True,
                'selected_indices': (0, 1, 2, 3),
            },
            'classifier': {
                'last_inchannels': 2048,
            },
            'head': {
                'in_channels': 2048,
                'feats_channels': 512,
                'out_channels': 512,
                'dropout': 0.1,
                'spectral_height': 64,
                'spectral_width': 64,
                'spectral_k': 8,
            },
            'auxiliary': {
                'in_channels': 1024,
                'out_channels': 512,
                'dropout': 0.1,
            }
        }
        
        self.mscc = MSCCNet(cfg, mode='TRAIN')
        self.prb_operator = PRB(
            walk_steps=3,
            walk_restart_prob=0.7,
            graph_threshold=0.4
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(1028, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.fuse_conv = nn.Sequential(#3*in_channels
            # nn.Conv2d(2048, in_channels, kernel_size=1, bias=False),#nldr: 
            nn.Conv2d(1028, in_channels, kernel_size=1, bias=False),#prb: 
            # msrd: nn.Conv2d(1026, in_channels, kernel_size=1, bias=False),# msrd: 
            # nn.Conv2d(1026, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.diff_attn = DiffAttention2D(in_channels, heads=8)
        self.conv_refine = ConvRefine(in_channels)
        self.unet = AttentionUNet(n_channels=512, n_classes=1)
        
        
    def forward(self, x):
        # 1. 主干网络特征提取
        feat16x = self.mscc(x)  # (B, 512, 32, 32)
        
        # 2. 图随机游走处理（捕捉非邻接像素关系）
        # 输入特征为32x32，适合构建图结构（节点数32*32=1024，计算量可控）
        npr_feat, global_feat = self.prb_operator(feat16x)  # (B,512,32,32) 和 (B,1024,32,32)
        
        # 3. 特征融合（原始特征 + 局部NPR特征 + 全局游走特征）
        fused_feat = torch.cat([feat16x, npr_feat, global_feat], dim=1)  # (B,512+512+1024,32,32)
        fused_feat = self.fuse_conv(fused_feat)  # 压缩回512通道 (B,512,32,32)
        
        # 4. 原有特征处理流程
        # print(fused_feat.shape)
        x = self.diff_attn(fused_feat)  # 注意力机制增强关键特征
        x = self.conv_refine(x)         # 特征精炼
        mask_logits = self.unet(x)      # UNet生成256x256掩码
        
        return mask_logits  # (B, 256, 256)


class DiceLoss(nn.Module):
    """Dice损失：强化区域重叠度，对小目标友好"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth  # 防止除零
    
    def forward(self, pred, target):
        """
        pred: 预测概率图 (B, H, W)，已通过sigmoid激活（范围[0,1]）
        target: 真实掩码 (B, H, W)，值为0或1
        """
        # 展平空间维度
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算交并比
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice  # 损失值越小越好

class BCEWithDiceLoss(nn.Module):
    """BCE-Dice混合损失：兼顾像素分类和区域重叠度"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_loss = nn.BCELoss()  # 若pred是logits，改用BCEWithLogitsLoss
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        """
        pred: 预测概率图 (B, H, W)，需通过sigmoid激活（范围[0,1]）
        target: 真实掩码 (B, H, W)，值为0或1（float类型）
        """
        # 计算BCE损失
        bce = self.bce_loss(pred, target)
        
        # 计算Dice损失
        dice = self.dice_loss(pred, target)
        
        # 混合损失
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        return total_loss

# 适配logits输出的版本（更常用，数值更稳定）
class BCEWithLogitsDiceLoss(nn.Module):
    """当模型输出为logits（未经过sigmoid）时使用，避免数值不稳定"""
    def __init__(self, bce_weight=0.7, dice_weight=0.3, smooth=1e-6):
        super().__init__()
        self.bce_loss = nn.BCELoss()  # 内置sigmoid，直接处理logits
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, logits, target):
        """
        logits: 模型输出的logits (B, H, W)，未经过sigmoid
        target: 真实掩码 (B, H, W)，值为0或1（float类型）
        """
        # 计算BCE损失（内置sigmoid）
        bce = self.bce_loss(logits, target)
        
        # 计算Dice损失（需先将logits转为概率）
        pred = torch.sigmoid(logits)
        dice = self.dice_loss(pred, target)
        
        # 混合损失
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        return total_loss
