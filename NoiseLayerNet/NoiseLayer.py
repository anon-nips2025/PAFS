from NoiseLayerNet.noise import *
from NoiseLayerNet.JPEGsimulator import JPEGSimulator


'''
part1: JPEGsim
part2: Gauss
'''
noise = True

class NoiseLayer(nn.Module):
    def __init__(self, device='cuda'):
        super(NoiseLayer, self).__init__()
        self.jpeg = JPEGSimulator().to(device)
        self.jpeg.requires_grad_(False)

        self.GaussianNoise = GaussianNoise().to(device)
        self.GaussianBlur = GaussianBlur2D().to(device)
        self.GaussianNoise.requires_grad_(False)
        self.GaussianNoise.requires_grad_(False)

        self.noise = [self.GaussianBlur, self.GaussianNoise, self.jpeg]


    def forward(self, x):
        if noise:
            r = torch.randint(0, 3, [1])
            out = self.noise[r[0]](x)
        else:
            out = x

        return out


if __name__ == '__main__':
    mode = NoiseLayer().to('cuda:0')
    for i in range(20):
        x = torch.randn(4, 3, 224, 224).to('cuda:0')
        y = mode(x)
        print(y.shape)
