from torch import nn
import segmentation_models_pytorch as smp

class Net(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()
        self.cfg = cfg
        self.encoder = smp.Unet(
            encoder_name = cfg.backbone, 
            encoder_weights = weight,
            in_channels = cfg.in_chans,
            classes = cfg.target_size,
            activation = None,
        )

    def forward(self, image):
        output = self.encoder(image)
        # output = output.squeeze(-1)
        return output[:,0]#.sigmoid()
    
def build_model(cfg):
    from dotenv import load_dotenv
    load_dotenv()

    print('model_name', cfg.model_name)
    print('backbone', cfg.backbone)

    model = Net(cfg, cfg.weight)

    return model.to(cfg.device)