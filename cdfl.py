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

class ChunkedLocalSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=1, local_window=16, chunk_size=256, layer_name=""):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.local_window = local_window
        self.chunk_size = chunk_size
        self.layer_name = layer_name
        
        
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
        
        N = H * W 
        x_flat = x.view(B, C, -1).permute(0, 2, 1) 

        h_coords = torch.arange(H, device=x.device).repeat_interleave(W)  # (N,)
        w_coords = torch.arange(W, device=x.device).repeat(H)  # (N,)
        half_win = self.local_window // 2

        num_chunks = (N + self.chunk_size - 1) // self.chunk_size
        attn_out_list = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min((chunk_idx + 1) * self.chunk_size, N)
            chunk_size = end - start

            chunk_h = h_coords[start:end]
            chunk_w = w_coords[start:end]

            h_start = torch.clamp(chunk_h - half_win, 0, H)
            h_end = torch.clamp(chunk_h + half_win + 1, 0, H)
            w_start = torch.clamp(chunk_w - half_win, 0, W)
            w_end = torch.clamp(chunk_w + half_win + 1, 0, W)

            j_h = h_coords.unsqueeze(0).repeat(chunk_size, 1)  # (chunk_size, N)
            j_w = w_coords.unsqueeze(0).repeat(chunk_size, 1)
            in_h = (j_h >= h_start.unsqueeze(1)) & (j_h < h_end.unsqueeze(1))
            in_w = (j_w >= w_start.unsqueeze(1)) & (j_w < w_end.unsqueeze(1))
            in_window = in_h & in_w

            chunk_mask = torch.full((chunk_size, N), -1e9, device=x.device)
            chunk_mask[in_window] = 0  

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

        attn_out = torch.cat(attn_out_list, dim=1)  # (B, N, C)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)
        
        # 残差连接 + 投影
        return self.act(self.proj(attn_out + x)), None

class LightEncoder(nn.Module):
    def __init__(self, in_channels=512, num_layers=3, local_window=16, chunk_size=256):
        super().__init__()
        self.channels = [512, 256, 128]   
        self.num_layers = num_layers

        self.num_heads = [8, 4, 2]

        self.init_conv = nn.Conv2d(
            in_channels, self.channels[0], kernel_size=1, padding=0, stride=1, bias=True
        )
        self.init_act = nn.ReLU()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
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
                    kernel_size=3, 
                    stride=2,
                    padding=1,
                    bias=True
                )
            }))

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.channels[0], f"Encoder输入通道不匹配：实际{C}，预期{self.channels[0]}"

        x = self.init_act(self.init_conv(x))  # (B, 512, 32, 32)

        encoder_feats = []  
        for i, layer in enumerate(self.layers):
            x, _ = layer['attn'](x)
            encoder_feats.append(x)
            x = layer['downsample'](x) 

        return x, encoder_feats

class LightDecoder(nn.Module):
    def __init__(self, num_layers=3, out_channels=1, local_window=16, chunk_size=256):
        super().__init__()
        self.channels = [128, 256, 512]
        self.num_heads = [2, 4, 8]
        self.num_layers = num_layers

        for c, h in zip(self.channels, self.num_heads):
            assert c % h == 0, f"Decoder通道{c}不是头数{h}的整数倍"

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                upsample_out = self.channels[i+1]
            else:
                upsample_out = self.channels[i]

            self.layers.append(nn.ModuleDict({
                'upsample': nn.ConvTranspose2d(
                    self.channels[i],
                    upsample_out,
                    kernel_size=3,  
                    stride=2,
                    padding=1,
                    output_padding=1, 
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

        self.final_conv = nn.Conv2d(self.channels[-1], out_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, encoder_feats=None):
        B, C, H, W = x.shape

        for i, layer in enumerate(self.layers):
            x = layer['upsample'](x)  
            x = layer['fuse'](x)
            x, _ = layer['attn'](x)

        x = F.interpolate(
            x,
            size=(256, 256),
            mode='bilinear',
            align_corners=True  
        )  # (B, 512, 256, 256)

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
        return self.unify(out) + x 
    
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
        return self.refine(x) + x 




class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads  
        
        self.qkv_proj = nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W 
        
        qkv = self.qkv_proj(x).view(B, 3, self.num_heads, self.head_dim, H, W)  # (B, 3, heads, head_dim, H, W)
        q, k, v = qkv.unbind(1) 
        
        q = q.flatten(3)  # (B, heads, head_dim, N)
        k = k.flatten(3)  # (B, heads, head_dim, N)
        v = v.flatten(3)  # (B, heads, head_dim, N)
        
        attn = torch.einsum('bhdn, bhdm -> bhnm', q, k)  # Q·K^T
        attn = attn * (self.head_dim **-0.5)  
        attn = self.softmax(attn) 
        
        out = torch.einsum('bhnm, bhdm -> bhdn', attn, v)  
        
        out = out.reshape(B, C, H, W) 
        
        out = self.out_proj(out) + x
        return out, attn  


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
    def __init__(self, in_channels_x1, x2_channels, out_channels, bilinear=True, attn_channels=None):
        super().__init__()
        self.bilinear = bilinear
        self.attn_channels = attn_channels  
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.x1_channels = in_channels_x1 
        else:
            self.up = nn.ConvTranspose2d(in_channels_x1, in_channels_x1 // 2, kernel_size=2, stride=2)
            self.x1_channels = in_channels_x1 // 2
        
        self.attn_proj = nn.Conv2d(attn_channels, out_channels // 2, kernel_size=1, bias=False)
        self.attn_proj_channels = out_channels // 2 
        
        self.total_channels = self.x1_channels + x2_channels + self.attn_proj_channels
        
        self.conv = DoubleConv(
            in_channels=self.total_channels,
            out_channels=out_channels,
            mid_channels=self.total_channels // 2 
        )

    def forward(self, x1, x2, attn_feat):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        attn_feat_proj = self.attn_proj(attn_feat)  
        x = torch.cat([x1, x2, attn_feat_proj], dim=1) 

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


class AttentionUNet(nn.Module):
    def __init__(self, n_channels=512, n_classes=1, bilinear=True, target_size=(256, 256)):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.target_size = target_size
        factor = 2 if bilinear else 1 
        
        self.x1_channels = 64
        self.x2_channels = 128
        self.x3_channels = 256
        self.x4_channels = 512 // factor
        
        self.inc = DoubleConv(n_channels, self.x1_channels)
        self.down1 = Down(self.x1_channels, self.x2_channels)
        self.attn1 = SelfAttention(in_channels=self.x1_channels, num_heads=4)  # 64→4头
        
        self.down2 = Down(self.x2_channels, self.x3_channels)
        self.attn2 = SelfAttention(in_channels=self.x2_channels, num_heads=4)  # 128→4头
        
        self.down3 = Down(self.x3_channels, self.x4_channels)
        self.attn3 = SelfAttention(in_channels=self.x3_channels, num_heads=8)  # 256→8头
        
        self.attn4 = SelfAttention(in_channels=self.x4_channels, num_heads=8)  # 256→8头
        
        self.up1 = Up(
            in_channels_x1=self.x4_channels, 
            x2_channels=self.x3_channels, 
            out_channels=self.x3_channels // factor, 
            bilinear=bilinear,
            attn_channels=self.x3_channels 
        )
        
        self.up2 = Up(
            in_channels_x1=self.x3_channels // factor, 
            x2_channels=self.x2_channels,                
            out_channels=self.x2_channels // factor,     
            bilinear=bilinear,
            attn_channels=self.x2_channels               
        )
        
        self.up3 = Up(
            in_channels_x1=self.x2_channels // factor, 
            x2_channels=self.x1_channels,              
            out_channels=self.x1_channels,             
            bilinear=bilinear,
            attn_channels=self.x1_channels              
        )
        
        self.outc = OutConv(self.x1_channels, n_classes)
        self.final_up = nn.Upsample(size=target_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.inc(x)  # (B, 64, H, W)
        x1_attn, _ = self.attn1(x1)  # (B, 64, H, W)
        
        x2 = self.down1(x1)  # (B, 128, H/2, W/2)
        x2_attn, _ = self.attn2(x2)  # (B, 128, H/2, W/2)
        
        x3 = self.down2(x2)  # (B, 256, H/4, W/4)
        x3_attn, _ = self.attn3(x3)  # (B, 256, H/4, W/4)
        
        x4 = self.down3(x3)  # (B, 256, H/8, W/8)
        x4_attn, _ = self.attn4(x4)  # (B, 256, H/8, W/8)
        
        x = self.up1(x4_attn, x3, x3_attn)  # (B, 128, H/4, W/4)
        x = self.up2(x, x2, x2_attn)        # (B, 64, H/2, W/2)
        x = self.up3(x, x1, x1_attn)        # (B, 64, H, W)
        
        logits = self.outc(x)  # (B, 1, H, W)
        logits = self.final_up(logits)  # (B, 1, 256, 256)
        return logits.squeeze(1)  # (B, 256, 256)
        

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
        feat16x = self.mscc(x)  # (B, 512, 32, 32)
        
        npr_feat, global_feat = self.prb_operator(feat16x)  # (B,512,32,32)  (B,1024,32,32)
        
        fused_feat = torch.cat([feat16x, npr_feat, global_feat], dim=1)  # (B,512+512+1024,32,32)
        fused_feat = self.fuse_conv(fused_feat)  
        
        # print(fused_feat.shape)
        x = self.diff_attn(fused_feat)  
        x = self.conv_refine(x)          
        mask_logits = self.unet(x)       
        
        return mask_logits  # (B, 256, 256)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth  
    
    def forward(self, pred, target):

        # 展平空间维度
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算交并比
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice  

class BCEWithDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_loss = nn.BCELoss()   
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        
        dice = self.dice_loss(pred, target)
        
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        return total_loss

class BCEWithLogitsDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.7, dice_weight=0.3, smooth=1e-6):
        super().__init__()
        self.bce_loss = nn.BCELoss()  
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, logits, target): 
        bce = self.bce_loss(logits, target)
        
        pred = torch.sigmoid(logits)
        dice = self.dice_loss(pred, target)
        
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        return total_loss
