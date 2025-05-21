import math
import os
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataset
from NoiseLayerNet.test_masklayer import mask_noise_layer
from NoiseLayerNet.NoiseLayer import NoiseLayer
from dataset import dataloader_Stage2
from models.fs_networks_fix import Generator_Adain_Upsample
import torch.nn.functional as F
import warnings

from models.model_Unet import Model

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)

run_name = "Stage2"
base_dir = "checkpoints_Stage2"
run_dir = os.path.join(base_dir, run_name)

# 创建文件夹用于保存这次运行的所有输出
os.makedirs(os.path.join(run_dir, 'samples'), exist_ok=True)
os.makedirs(os.path.join(run_dir, 'weights'), exist_ok=True)
os.makedirs(os.path.join(run_dir, 'summary'), exist_ok=True)

weights_dir = os.path.join(run_dir, 'weights')

save_path_S = r"pretrained_weights/netG_batch_800000.pth"
Arc_path = r'arcface_model/arcface_checkpoint.tar'

writer = SummaryWriter(log_dir=os.path.join(run_dir, 'summary'))

total_step = 100000

# network
lr_g = 1e-4
weight_step=20
init_scale=0.1
gamma=0.8

face_swap_weight = 7
face_reveal_weight = 50
whole_reveal_weight = 7
background_hidden_weight = 35
ID_weight = 1

SAVE_itertaion_freq = 20000
SAVE_epoch_freq = 1
num_epochs = 1000


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def cosin_metric(x1, x2):
    #return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))


l2 = nn.MSELoss()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Model()
    net = net.to(device)

    noiselayer = NoiseLayer()
    noiselayer = noiselayer.to(device)
    noiselayer.requires_grad_(False)

    para = get_parameter_number(net)
    print(para)
    params_trainable_net = (list(filter(lambda p: p.requires_grad, net.parameters())))

    opt_g = torch.optim.Adam(list(net.parameters()), lr=lr_g)
    weight_scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, weight_step, gamma=gamma)

    netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False).to(device)
    saved_state_dict = torch.load(save_path_S, map_location=torch.device("cpu"))
    netG.load_state_dict(saved_state_dict)

    netArc_checkpoint = Arc_path
    netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
    netArc = netArc_checkpoint
    netArc = netArc.to(device)
    netArc.eval()

    netG.eval()
    for param in netG.parameters():
        param.requires_grad = False

    iteration = 0
    for epoch in range(num_epochs):
        losses_face_swap = []
        losses_face_reveal = []
        losses_whole_reveal = []
        losses_background_hidden = []
        losses_ID = []
        losses_total = []
        for i_batch, data in enumerate(tqdm(dataloader_Stage2.trainloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", unit="batch")):
            wholeimage_GT = data[0].to(device)
            mask_GT = data[1].to(device)
            face_GT = wholeimage_GT * mask_GT
            background_GT = wholeimage_GT * (1 - mask_GT)

            # 准备身份向量
            img_id_112 = F.interpolate(wholeimage_GT, size=(112, 112), mode='bicubic')
            latent_id = netArc(img_id_112)
            latent_id = F.normalize(latent_id, p=2, dim=1)
            opt_g.zero_grad()

            background_hiding_withmask, zero_faceregion, face_inn = net(mask_GT, background_GT, face_GT, rev=False)

            output_tensor = netG.forward(input=background_hiding_withmask, dlatents=latent_id, mask=None, bg=None,
                                         face=None, hidden=False)

            background_hiding_withmask_whole = background_hiding_withmask + face_GT

            background_hiding_withmask_whole_noise = noiselayer(background_hiding_withmask_whole)

            # if i_batch % 2 == 0:
            #     mask_noise = mask_noise_layer(mask_GT)
            # else:
            #     mask_noise = mask_GT

            mask_noise = mask_noise_layer(mask_GT)

            background_hiding_withmask_noise = background_hiding_withmask_whole_noise * (1 - mask_noise)

            whole_reveal, face_reveal = net(mask_noise, background_hiding_withmask_noise, rev=True)

            face_GT = wholeimage_GT * mask_noise

            img_fake_down = F.interpolate(whole_reveal, size=(112, 112), mode='bicubic')

            latent_fake = netArc(img_fake_down)
            latent_fake = F.normalize(latent_fake, p=2, dim=1)
            loss_G_ID = (1 - cosin_metric(latent_fake, latent_id)).mean()


            loss_face_swap = l2(output_tensor, wholeimage_GT)
            loss_face_reveal = l2(face_reveal, face_GT)
            loss_whole_reveal = l2(whole_reveal, wholeimage_GT)
            loss_background_hidden = l2(background_hiding_withmask, background_GT)

            loss_total = loss_face_swap * face_swap_weight + loss_face_reveal * face_reveal_weight + loss_whole_reveal * whole_reveal_weight + loss_background_hidden * background_hidden_weight

            loss_total = loss_total + loss_G_ID * ID_weight

            loss_total.backward()
            opt_g.step()

            losses_face_swap.append(loss_face_swap.item())
            losses_face_reveal.append(loss_face_reveal.item())
            losses_whole_reveal.append(loss_whole_reveal.item())
            losses_background_hidden.append(loss_background_hidden.item())

            losses_ID.append(loss_G_ID.item())

            losses_total.append(loss_total.item())

            iteration = iteration + 1

        avg_loss_total = sum(losses_total) / len(losses_total)
        avg_loss_face_reveal = sum(losses_face_reveal) / len(losses_face_reveal)
        avg_loss_whole_reveal = sum(losses_whole_reveal) / len(losses_whole_reveal)
        avg_loss_face_swap = sum(losses_face_swap) / len(losses_face_swap)
        avg_loss_background_hidden = sum(losses_background_hidden) / len(losses_background_hidden)
        avg_loss_ID = sum(losses_ID) / len(losses_ID)

        weight_scheduler_g.step()

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Avg Total Loss: {avg_loss_total:.4f}, "
              f"Avg Face Reveal Loss: {avg_loss_face_reveal:.4f}, "
              f"Avg Whole Reveal Loss: {avg_loss_whole_reveal:.4f}, "
              f"Avg Background Hidden Loss: {avg_loss_background_hidden:.4f}, "
              f"avg_loss_ID: {avg_loss_ID:.4f}, "
              f"Avg Face Swap Loss: {avg_loss_face_swap:.4f}")

        if (epoch % SAVE_epoch_freq == 0) & (epoch != 0):
        # if True:
            with torch.no_grad():
                psnr_face_swap = []
                psnr_face_reveal = []
                psnr_whole_reveal = []
                psnr_background_hidden = []

                random_batch_index = random.randint(0, len(dataloader_Stage2.valloader) - 1)

                net.eval()
                for i, x in enumerate(dataloader_Stage2.valloader):
                    wholeimage_GT = x[0].to(device)
                    mask_GT = x[1].to(device)
                    face_GT = wholeimage_GT * mask_GT
                    background_GT = wholeimage_GT * (1 - mask_GT)

                    img_id_112 = F.interpolate(wholeimage_GT, size=(112, 112), mode='bicubic')
                    latent_id = netArc(img_id_112)
                    latent_id = F.normalize(latent_id, p=2, dim=1)

                    background_hiding_withmask, zero_faceregion, face_inn = net(mask_GT, background_GT, face_GT, rev=False)

                    output_tensor = netG.forward(input=background_hiding_withmask, dlatents=latent_id, mask=None,
                                                 bg=None,
                                                 face=None, hidden=False)

                    background_hiding_withmask_whole = background_hiding_withmask + face_GT

                    background_hiding_withmask_whole_noise = noiselayer(background_hiding_withmask_whole)

                    mask_noise = mask_GT

                    background_hiding_withmask_noise = background_hiding_withmask_whole_noise * (
                                1 - mask_noise)

                    whole_reveal, face_reveal = net(mask_noise, background_hiding_withmask_noise, rev=True)

                    face_GT = wholeimage_GT * mask_noise

                    wholeimage_GT = ((wholeimage_GT.cpu()) * imagenet_std + imagenet_mean)
                    background_GT = ((background_GT.cpu()) * imagenet_std + imagenet_mean)
                    background_hiding_withmask = ((background_hiding_withmask.cpu()) * imagenet_std + imagenet_mean)
                    whole_reveal = ((whole_reveal.cpu()) * imagenet_std + imagenet_mean)
                    face_reveal = ((face_reveal.cpu()) * imagenet_std + imagenet_mean)
                    face_GT = ((face_GT.cpu()) * imagenet_std + imagenet_mean)
                    output_tensor = ((output_tensor.cpu()) * imagenet_std + imagenet_mean)

                    face_GT = face_GT.cpu().numpy().squeeze() * 255
                    np.clip(face_GT, 0, 255)
                    face_reveal = face_reveal.cpu().numpy().squeeze() * 255
                    np.clip(face_reveal, 0, 255)

                    wholeimage_GT = wholeimage_GT.cpu().numpy().squeeze() * 255
                    np.clip(wholeimage_GT, 0, 255)
                    whole_reveal = whole_reveal.cpu().numpy().squeeze() * 255
                    np.clip(whole_reveal, 0, 255)
                    output_tensor = output_tensor.cpu().numpy().squeeze() * 255
                    np.clip(output_tensor, 0, 255)

                    background_GT = background_GT.cpu().numpy().squeeze() * 255
                    np.clip(background_GT, 0, 255)
                    background_hiding_withmask = background_hiding_withmask.cpu().numpy().squeeze() * 255
                    np.clip(background_hiding_withmask, 0, 255)

                    psnr_temp_face_swap = computePSNR(wholeimage_GT, output_tensor)
                    psnr_face_swap.append(psnr_temp_face_swap)

                    psnr_temp_face_reveal = computePSNR(face_GT, face_reveal)
                    psnr_face_reveal.append(psnr_temp_face_reveal)

                    psnr_temp_whole_reveal = computePSNR(wholeimage_GT, whole_reveal)
                    psnr_whole_reveal.append(psnr_temp_whole_reveal)

                    psnr_temp_background_hidden = computePSNR(background_GT, background_hiding_withmask)
                    psnr_background_hidden.append(psnr_temp_background_hidden)

                    if i == random_batch_index:
                        image_filename = os.path.join(run_dir, 'samples', f'val_images_epoch_{epoch}.png')

                        nrow = 1
                        padding = 2

                        wholeimage_GT_grid = vutils.make_grid(
                            torch.from_numpy(np.clip(wholeimage_GT / 255, 0, 1)).float(),
                            nrow=nrow,
                            padding=padding,
                            normalize=False
                        )

                        whole_reveal_grid = vutils.make_grid(
                            torch.from_numpy(np.clip(whole_reveal / 255, 0, 1)).float(),
                            nrow=nrow,
                            padding=padding,
                            normalize=False
                        )

                        output_tensor_grid = vutils.make_grid(
                            torch.from_numpy(np.clip(output_tensor / 255, 0, 1)).float(),
                            nrow=nrow,
                            padding=padding,
                            normalize=False
                        )

                        background_hiding_grid = vutils.make_grid(
                            torch.from_numpy(np.clip(background_hiding_withmask / 255, 0, 1)).float(),
                            nrow=nrow,
                            padding=padding,
                            normalize=False
                        )

                        combined_grid = torch.cat(
                            [wholeimage_GT_grid, whole_reveal_grid,
                             output_tensor_grid, background_hiding_grid],
                            dim=2
                        )

                        grid_height = combined_grid.shape[1]
                        grid_width = combined_grid.shape[2]
                        dpi = 100
                        fig_width = grid_width / dpi * 1.3
                        fig_height = grid_height / dpi * 1.1

                        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
                        plt.imshow(np.transpose(combined_grid.numpy(), (1, 2, 0)))
                        plt.axis('off')

                        column_titles = [
                            ('Ground Truth', 1 / 8),
                            ('Whole Reveal', 3 / 8),
                            ('FaceSwap', 5 / 8),
                            ('Hidden Background', 7 / 8)
                        ]

                        for title, pos_ratio in column_titles:
                            plt.text(
                                grid_width * pos_ratio,
                                -grid_height * 0.05,
                                title,
                                ha='center',
                                va='top',
                                fontsize=8,
                                color='white',
                                backgroundcolor='black'
                            )

                        plt.savefig(
                            image_filename,
                            bbox_inches='tight',
                            pad_inches=0.1,
                            dpi=300
                        )
                        plt.close()


                print("psnr_face_reveal:", np.mean(psnr_face_reveal), 'psnr_whole_reveal:', np.mean(psnr_whole_reveal), 'psnr_face_swap:',
                      np.mean(psnr_face_swap), 'psnr_background_hidden:', np.mean(psnr_background_hidden))
                writer.add_scalars("psnr_face_reveal", {"average psnr": np.mean(psnr_face_reveal)}, epoch)
                writer.add_scalars("psnr_whole_reveal", {"average psnr": np.mean(psnr_whole_reveal)}, epoch)
                writer.add_scalars("psnr_face_swap", {"average psnr": np.mean(psnr_face_swap)}, epoch)
                writer.add_scalars("psnr_background_hidden", {"average psnr": np.mean(psnr_background_hidden)}, epoch)

                model_save_path = os.path.join(weights_dir, f"decoder_epoch_{epoch + 1}.pth")
                torch.save(net.state_dict(), model_save_path)












