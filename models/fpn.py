from torch import nn
import segmentation_models_pytorch as smp

class FPN(nn.Module):
    def __init__(self, CFG, weight=None):
        super().__init__()
        self.model = smp.FPN(  # FPN Unet
            encoder_name=CFG.backbone,
            encoder_weights=weight,
            in_channels=CFG.in_chans,
            classes=CFG.target_size,
            activation=None,
        )

    def forward(self, image):
        output = self.model(image)
        # output = output.squeeze(-1)
        return output[:, 0]  # .sigmoid()
    
def build_model(cfg):
    from dotenv import load_dotenv
    load_dotenv()

    print('model_name', cfg.model_name)
    print('backbone', cfg.backbone)

    model = FPN(cfg, cfg.weight)

    return model.to(cfg.device)