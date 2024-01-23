from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import torch.nn.functional as F
from scipy.ndimage import label, sum_labels
from utils.utils import Config, min_max_normalization, setup_seeds,\
                        get_date_time, SurfaceLoss,BCEWithLogitsLossManual,\
                        DiceLoss, norm_with_clip,add_noise
from utils.dataset import KaggleDataset
from models.unet import build_model
from optimizer.loss import surface_dice

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(WeightedBCEWithLogitsLoss, self).__init__()

    def forward(self, inputs, targets):
        # 原始的binary cross entropy with logits loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 获得预测概率（通过sigmoid激活）
        probs = torch.sigmoid(inputs)
        weight_matrix = torch.zeros(targets.shape)
        # 计算每个样本的像素数量
        for i in range(targets.shape[0]):
            tmp_2d_matrix = targets[i]
            labeled_matrix, num_features = label(tmp_2d_matrix.cpu())
            if num_features == 0:
                continue
            # 计算每个连通区域的大小
            sizes = sum_labels(tmp_2d_matrix.cpu(), labels=labeled_matrix, index=range(1, num_features + 1))
            # 创建一个等同于labeled_matrix的大小矩阵，并通过sizes来映射填充此矩阵
            area_matrix = sizes[labeled_matrix - 1]  # -1 是因为标签从1开始计数，而索引从0开始
            area_matrix[labeled_matrix == 0] = 0  # 确保背景不被标记
            # 可以将结果赋给weight_matrix 用于其他操作
            weight_matrix[i] = torch.from_numpy(area_matrix)
            
        # 开平方权重分布（针对较小样本提供更大的权重）
        # weights = 1 / (torch.sqrt(weight_matrix) + 1)
        weight_matrix = weight_matrix.cuda()
        # 应用权重调整到每个像素上的损失
        # 计算非零值的平方根
        nonzero_weights_sqrt = torch.sqrt(weight_matrix[weight_matrix != 0])
        # 计算非零平方根的平均值
        nonzero_mean = torch.mean(nonzero_weights_sqrt)
        # 创建一个全为非零平方根平均值的矩阵
        non_zero_matrix = torch.full(weight_matrix.shape, nonzero_mean, device=weight_matrix.device)
        # 将 weights 矩阵中的 0 替换成非零值平方根的平均值
        # 这里使用条件赋值（where语句），若 weights 的某个位置为0，则从 non_zero_matrix 取值赋予它；非0则保持原始值。
        weight_matrix = torch.sqrt(weight_matrix)
        weights_with_nonzero_mean = torch.where(weight_matrix == 0, non_zero_matrix, weight_matrix)
        weights_with_nonzero_mean = 1/(weights_with_nonzero_mean + 1)
        # 应用权重调整到每个像素上的损失
        weighted_bce_loss = weights_with_nonzero_mean * bce_loss
        weighted_bce_loss = weighted_bce_loss.mean()
        # 最终损失按照样本平均
        return weighted_bce_loss

def main():
    cfg = Config('configs/train_config.yaml')
    setup_seeds(cfg)
    train_dataset = KaggleDataset(cfg)
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=cfg.num_workers)
    val_dataset = KaggleDataset(cfg, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=cfg.num_workers)

    model = build_model(cfg)
    model = DataParallel(model)

    loss_fn_1 = DiceLoss() # 和 BCEloss 加权 预训练 模型 医学 CT
    loss_fn_2 = nn.BCEWithLogitsLoss()
    loss_fn_3 = WeightedBCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr,
                                                steps_per_epoch=len(train_loader), epochs=cfg.epochs+1,
                                                pct_start=0.1)
    
    date, time = get_date_time()
    
    for epoch in range(cfg.epochs):
        train_one_epoch(model, train_loader, loss_fn_1,loss_fn_1,  optimizer, scaler, scheduler, epoch, cfg)
        val(model, val_loader, loss_fn_1,loss_fn_1, epoch, cfg)
        model_dir = os.path.join(cfg.output_path, f'{cfg.model_name}_{date}_{time}_New_Unet)')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_save_path = os.path.join(model_dir, f'epoch_{epoch}.pt')
        torch.save(model.state_dict(), model_save_path)



def train_one_epoch(model, train_loader, loss_fn_1 ,loss_fn_2, optimizer, scaler, scheduler, epoch, cfg):
    model.train()
    iters = tqdm(range(len(train_loader)))
    train_loss = 0
    dice_score = 0
    
    for i, (images, masks) in enumerate(train_loader):
        images = images.cuda().to(torch.float32)
        masks = masks.cuda().to(torch.float32)
        images = norm_with_clip(images.reshape(-1,*images.shape[2:])).reshape(images.shape)
        images = add_noise(images,max_randn_rate=0.5,x_already_normed=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = 0.0 * loss_fn_1(outputs, masks) + 1 *loss_fn_2(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        train_loss = (train_loss*i + loss.item())/(i+1)
        dice_score = (dice_score*i + surface_dice(outputs, masks))/(i+1)
        iters.set_description(f"Epoch {epoch+1}/{cfg.epochs}, train loss: {train_loss:.4f}, dice score: {dice_score:.4f}")
        iters.update()

    iters.close()


def val(model, val_loader, loss_fn_1,loss_fn_2, epoch, cfg):
    model.eval()
    iters = tqdm(range(len(val_loader)))
    val_loss = 0
    val_dice_score = 0
    
    for i, (images, masks) in enumerate(val_loader):
        images = images.cuda().to(torch.float32)
        masks = masks.cuda().to(torch.float32)
        images = norm_with_clip(images.reshape(-1,*images.shape[2:])).reshape(images.shape)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = model(images)
                loss = 0.0 * loss_fn_1(outputs, masks) + 1 *loss_fn_2(outputs, masks)
        val_loss = (val_loss*i + loss.item())/(i+1)
        val_dice_score = (val_dice_score*i + surface_dice(outputs, masks))/(i+1)
        iters.set_description(f"Epoch {epoch+1}/{cfg.epochs}, val loss: {val_loss:.4f}, val dice score: {val_dice_score:.4f}")
        iters.update()
    iters.close()
    


if __name__ == '__main__':
    main()