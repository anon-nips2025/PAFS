import torch.optim
import torch.nn as nn

from models.Net import ResidualUNet
from models.res_unet_plusV2 import ResUnetPlusPlus


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.preprocess = ResidualUNet(3)
        self.encoder = ResidualUNet(6)
        self.decoder = ResUnetPlusPlus(3)

    def forward(self, mask, bg, face=None, rev=False):
        if not rev:
            face_pro=self.preprocess(face)
            out=self.encoder(torch.cat([bg,face_pro],1))
            out_bg = out * (1 - mask)
            zero_region = out * mask
            return out_bg, zero_region, None
        else:
            out = self.decoder(bg)
            out_face = out * mask
            return out, out_face