import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import BasicBlock
from cdfl import FakeLocator
from cdfl import BCEWithLogitsDiceLoss

from PIL import Image
import numpy as np
from torchvision import transforms
from sklearn.metrics import roc_auc_score


class SegmentationDataset(Dataset):
    def __init__(self, lst_path, transform=None):
        self.transform = transform
        self.image_mask_pairs = self._load_lst(lst_path)
        self.resize = transforms.Resize((256, 256))
        # self.crop = transforms.CenterCrop((126, 126))

    def _load_lst(self, lst_path):
        pairs = []
        with open(lst_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # print(line)
                img_path, mask_path = line.split(',')
                pairs.append((img_path, mask_path))
        return pairs

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(img_path).convert('RGB')
        # mask = Image.open(mask_path).convert('L')
        if mask_path != 'None': #and os.path.exists(mask_path):
            # 存在有效掩码路径，读取掩码
            mask = Image.open(mask_path).convert('L')
        else:
            # 掩码路径为None或不存在，创建与图片等大的全黑掩码
            mask = Image.new('L', image.size, 0)  # 'L'模式，全黑(0)
        
        # 先将图片和mask缩放到512x512
        image = self.resize(image)
        mask = self.resize(mask)

        # 裁剪图片和蒙版
        # image = self.crop(image)
        # mask = self.crop(mask)

        image = np.array(image)
        mask = np.array(mask)
        mask = (mask > 128).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float()

        if self.transform:
            image = self.transform(image)

        return image, mask

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    step_count = 0
    
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)  # masks 形状 (B, H, W)
        
        optimizer.zero_grad()
        outputs = model(images)  # (B, 1, H, W)
        outputs = torch.sigmoid(outputs).squeeze(1)  # (B, H, W)
        # print(':', masks.shape)
        # print(outputs.shape)
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        step_count += 1

        if step_count % 100 == 0:
            current_loss = running_loss / 100
            print(f"Step [{step_count}/{len(train_loader)}]: Loss = {current_loss:.4f}")
            running_loss = 0.0
            # step_count = 0
        
    return running_loss / len(train_loader)


def validate(model, val_loader, device, threshold=0.5):
    total_tp = 0  # 真阳性总数
    total_fp = 0  # 假阳性总数
    total_fn = 0  # 假阴性总数
    total_intersection = 0  # 交集总数（即TP）
    total_union = 0  # 并集总数
    
    all_pred_probs = []
    all_true_labels = []

    model.eval()  # 模型切换为评估模式
    with torch.no_grad():  # 关闭梯度计算
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # 预测结果处理：sigmoid归一化→二值化（阈值threshold）
            outputs_sigmoid = torch.sigmoid(outputs).squeeze(1)  # 去除通道维，shape: (B, H, W)
            pred_masks = (outputs_sigmoid > threshold).float()    # 二值化预测掩码
            
            # 转换为uint8类型（适配位运算）
            pred_masks = pred_masks.to(torch.uint8)
            masks = masks.to(torch.uint8).squeeze(1)  # 确保mask为单通道 (B, H, W)

            # 计算批次级别的TP、FP、FN
            tp = (pred_masks & masks).sum().item()          # 真阳性
            fp = (pred_masks & (~masks)).sum().item()       # 假阳性
            fn = ((~pred_masks) & masks).sum().item()       # 假阴性
            union = (pred_masks | masks).sum().item()       # 并集
            
            # 累加统计量
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_intersection += tp  # 交集就是TP
            total_union += union
            
            # 收集预测概率和真实标签（展平为一维数组）
            all_pred_probs.extend(outputs_sigmoid.cpu().numpy().flatten())
            all_true_labels.extend(masks.cpu().numpy().flatten())

    # 在所有样本处理完毕后，计算最终指标
    if len(np.unique(all_true_labels)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(all_true_labels, all_pred_probs)
    
    # 计算Precision (精确率)
    if (total_tp + total_fp) == 0:
        precision = 1.0  # 无正预测时默认为1.0
    else:
        precision = total_tp / (total_tp + total_fp)
    
    # 计算Recall (召回率)
    if (total_tp + total_fn) == 0:
        recall = 1.0  # 无真实正样本时默认为1.0
    else:
        recall = total_tp / (total_tp + total_fn)
    
    # 计算F1分数
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    # 计算IoU (交并比)
    if total_union == 0:
        iou = 0.0
    else:
        iou = total_intersection / total_union

    return iou, precision, recall, f1, auc


if __name__ == "__main__":
    epochs = 5000

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # model = ResNetSegmentationImproved(BasicBlock, [3, 4, 6, 3], num_classes=1).to(device)
    # No NPR
    model = FakeLocator(512, 1).to(device)
    # model = UltraLightPRBNet(
    #     img_channels=3,
    #     encoder_num_layers=3,
    #     encoder_channels=[8, 16, 32],
    #     local_window=16,
    #     chunk_size=256
    # ).to(device)
    # model_path = "CASIA_v2_model.pth"
    # model.load_state_dict(torch.load(model_path))
    
    # 数据加载
    train_loader = DataLoader(
        # SegmentationDataset("./data/lst/FantasticReality_train.lst"),
        # SegmentationDataset("./data/lst/IMD2020_tamp_train.lst"),
        # SegmentationDataset("./data/lst/CASIA_v2_train.lst"),
        SegmentationDataset("/data/wyl/myModel/SDPAL/data/image_paths.txt"),
        # SegmentationDataset("./data/lst/DDL_tamp_train.lst"),
        batch_size=8, shuffle=True
    )
    val_loader = DataLoader(
        # SegmentationDataset("./data/lst/FantasticReality_valid.lst"),
        # SegmentationDataset("./data/lst/IMD2020_tamp_valid.lst"),
        # SegmentationDataset("./data/lst/CASIA_v2_valid.lst"),
        # SegmentationDataset("./data/lst/CocoGlide_tamp.lst"),
        SegmentationDataset("/data/wyl/myModel/SDPAL/data/image_paths.txt"),
        # /data/wyl/myModel/RWT/ls.lst
        batch_size=8, shuffle=False
    )
    
    # criterion = nn.BCELoss()
    criterion = BCEWithLogitsDiceLoss(bce_weight=1.0, dice_weight=0.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_f1 = 0.0
    train_loss = 0.0
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)

        avg_iou, avg_precision, avg_recall, avg_f1, auc = validate(model, val_loader, device)
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | IoU: {avg_iou:.4f}")
        print(f"AUC: {auc:.4f} | F1: {avg_f1:.4f} | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            # 保存模型
            # torch.save(model.state_dict(), "FantasticReality_best_model.pth")
            # torch.save(model.state_dict(), "IMD2020_best_model.pth")
            # torch.save(model.state_dict(), "CASIA_v2_model.pth")
            best_f1_int = int(best_f1 * 100)  # 得到 78
            torch.save(model.state_dict(), f"./models/CASIA_v2_model.pth")
            print(f"Model saved with F1: {best_f1:.4f}")
        