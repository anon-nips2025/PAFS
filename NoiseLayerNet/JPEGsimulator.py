from NoiseLayerNet.JPEG import DiffJPEG
from NoiseLayerNet.jpegMASK import JpegMask
from NoiseLayerNet.compressNet import CompressNet
from NoiseLayerNet.rgb2yuv import *


def load(network, pathname, netname):
    state_dicts = torch.load(pathname)
    network_state_dict = {k: v for k, v in state_dicts[netname].items() if 'tmp_var' not in k}
    network.load_state_dict(network_state_dict)


class JPEGSimulator(nn.Module):
    def __init__(self,device='cuda'):
        super(JPEGSimulator, self).__init__()

        self.jpeg_isp = DiffJPEG(quality=90).to(device)

        self.jpeg_hidden = JpegMask(Q=90).to(device)

        self.simnet = CompressNet().to(device)

        self.simnet.load_state_dict(torch.load('pretrained_weights/NoiseLayer256.pth'))

        self.rgbtoyuv = rgb_to_ycbcr_jpeg()
        self.rgbtoyuv = self.rgbtoyuv.to(device)
        self.yuvtorgb = ycbcr_to_rgb_jpeg()
        self.yuvtorgb = self.yuvtorgb.to(device)

    def forward(self, x,device='cuda'):
        y_isp = self.jpeg_isp(x)
        y_hidden = self.jpeg_hidden(x)
        y_simnet = self.yuvtorgb(self.simnet(self.rgbtoyuv(x)))

        weight = np.random.rand(3)
        sum = weight.sum()
        weight = weight / sum

        out = weight[0] * y_isp + weight[1] * y_hidden + weight[2] * y_simnet
        return out