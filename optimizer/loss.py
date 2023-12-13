import torch

def surface_dice(pred:torch.Tensor, target:torch.Tensor):
    if torch.any(pred < 0) or torch.any(pred > 1):
        pred = pred.sigmoid()
    pred = pred.reshape(-1) > 0.5
    target = target.reshape(-1)
    intersection = torch.sum(pred*target)
    union = torch.sum(pred + target) + 1e-9
    return (2*intersection/union).cpu().item()