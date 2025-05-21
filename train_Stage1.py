import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
import torch.utils.tensorboard as tensorboard

from dataset.dataloader_Stage1 import create_dataloader, DataPrefetcher
from models.projected_hidden_model import fsModel
from util import util
from util.plot import plot_batch


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def str2bool(v):
    return v.lower() in ('true')

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default='Stage1', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', default='0')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints_Stage1', help='models are saved here')
        self.parser.add_argument('--isTrain', type=str2bool, default='True')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=4, help='input batch size')

        # for displays
        self.parser.add_argument('--use_tensorboard', type=str2bool, default='True')
        # for training
        self.parser.add_argument('--dataset', type=str, default=r"data/train/images", help='path to the face swapping dataset')
        self.parser.add_argument('--mask_path', type=str, default=r"data/train/masks", help='path to the face swapping dataset')
        self.parser.add_argument('--continue_train', type=str2bool, default='False', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='./checkpoints/simswap224_test', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='10000', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=10000, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=10000, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0004, help='initial learning rate for adam')
        self.parser.add_argument('--Gdeep', type=str2bool, default='False')

        # for discriminators         
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--lambda_id', type=float, default=30.0, help='weight for id loss')
        self.parser.add_argument('--lambda_rec', type=float, default=30.0, help='weight for reconstruction loss')
        self.parser.add_argument('--lambda_hidden', type=float, default=10.0, help='weight for background hidden loss')

        self.parser.add_argument("--Arc_path", type=str, default='arcface_model/arcface_checkpoint.tar', help="run ONNX model via TRT")
        self.parser.add_argument("--total_step", type=int, default=1000000, help='total training step')
        self.parser.add_argument("--log_frep", type=int, default=200, help='frequence for printing log information')
        self.parser.add_argument("--sample_freq", type=int, default=1000, help='frequence for sampling')
        self.parser.add_argument("--model_freq", type=int, default=20000, help='frequence for saving the model')

        self.isTrain = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        if self.opt.isTrain:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdirs(expr_dir)
            if save and not self.opt.continue_train:
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
        return self.opt

def compute_loss(output_bg, target_bg):
    criterion = nn.MSELoss()
    loss_bg = criterion(output_bg, target_bg)
    total_loss = loss_bg
    return total_loss

# helper saving function that can be used by subclasses
def save_network_path(network, network_label, epoch_label, save_dir):
    save_filename = '{}_net_{}.pth'.format(epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)


def save_generator_weights(generator, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'netG_batch_{epoch}.pth')
    torch.save(generator.state_dict(), save_path)

if __name__ == '__main__':
    import site

    print(site.getsitepackages())

    opt         = TrainOptions().parse()
    iter_path   = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    sample_path = os.path.join(opt.checkpoints_dir, opt.name, 'samples')

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    log_path = os.path.join(opt.checkpoints_dir, opt.name, 'summary')

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    print("GPU used : ", str(opt.gpu_ids))


    cudnn.benchmark = True

    model = fsModel()
    model.initialize(opt)

    if opt.use_tensorboard:
        tensorboard_writer  = tensorboard.SummaryWriter(log_path)
        logger              = tensorboard_writer

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)

    optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D
    loss_avg        = 0
    refresh_count   = 0
    imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)

    dataloader = create_dataloader(opt.dataset, opt.mask_path, batch_size=opt.batchSize)
    prefetcher = DataPrefetcher(dataloader)
    randindex = [i for i in range(opt.batchSize)]
    random.shuffle(randindex)

    if not opt.continue_train:
        start   = 0
    else:
        start   = int(opt.which_epoch)
    total_step  = opt.total_step

    import datetime
    print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    model.netD.feature_network.requires_grad_(False)

    total_loss_Gmain = 0
    total_loss_G_ID = 0
    total_loss_G_Rec = 0
    total_loss_G_feat_match = 0
    total_loss_G_hidden = 0
    total_loss_Dgen = 0
    total_loss_Dreal = 0
    total_loss_D = 0
    batch_count = 0

    # Training Cycle
    for step in range(start, total_step):
        model.netG.train()
        for interval in range(2):
            random.shuffle(randindex)
            src_image1, src_image2, mask1 = prefetcher.next()
            src_image1 = src_image1.to("cuda:0")
            src_image2 = src_image2.to("cuda:0")
            mask1 = mask1.to("cuda:0")

            if step%2 == 0:
                img_id = src_image1
            else:
                img_id = src_image2

            face_GT = src_image1 * mask1
            background_GT = src_image1 * (1 - mask1)

            background_hiding_withmask, zero_faceregion, face_inn = model.netG(input=None, dlatents=None, mask=mask1, bg=background_GT, face=face_GT, hidden=True)  # 经过嵌入网络

            img_id_112      = F.interpolate(img_id,size=(112,112), mode='bicubic')
            latent_id       = model.netArc(img_id_112)
            latent_id       = F.normalize(latent_id, p=2, dim=1)
            if interval:
                img_fake        = model.netG(input=background_hiding_withmask, dlatents=latent_id, mask=None, bg=None, face=None, hidden=False)
                gen_logits,_    = model.netD(img_fake.detach(), None)
                loss_Dgen       = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                real_logits,_   = model.netD(src_image2,None)
                loss_Dreal      = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                loss_D          = loss_Dgen + loss_Dreal
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

                total_loss_Dgen += loss_Dgen.item()
                total_loss_Dreal += loss_Dreal.item()
                total_loss_D += loss_D.item()

            else:
                img_fake        = model.netG(input=background_hiding_withmask, dlatents=latent_id, mask=None, bg=None, face=None, hidden=False)
                # G loss
                gen_logits,feat = model.netD(img_fake, None)

                loss_Gmain      = (-gen_logits).mean()
                img_fake_down   = F.interpolate(img_fake, size=(112,112), mode='bicubic')
                latent_fake     = model.netArc(img_fake_down)
                latent_fake     = F.normalize(latent_fake, p=2, dim=1)
                loss_G_ID       = (1 - model.cosin_metric(latent_fake, latent_id)).mean()
                real_feat       = model.netD.get_feature(src_image1)
                feat_match_loss = model.criterionFeat(feat["3"],real_feat["3"])
                loss_G          = loss_Gmain + loss_G_ID * opt.lambda_id + feat_match_loss * opt.lambda_feat

                loss_hidden = compute_loss(background_hiding_withmask, background_GT)
                loss_G = loss_G + loss_hidden * opt.lambda_hidden

                if step%2 == 0:
                    loss_G_Rec  = model.criterionRec(img_fake, src_image1) * opt.lambda_rec
                    loss_G      += loss_G_Rec

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                total_loss_Gmain += loss_Gmain.item()
                total_loss_G_ID += loss_G_ID.item()
                total_loss_G_Rec += loss_G_Rec.item()
                total_loss_G_feat_match += feat_match_loss.item()
                total_loss_G_hidden += loss_hidden.item()

                batch_count += 1


        ############## Display results and errors ##########
        if (step + 1) % opt.log_frep == 0 and batch_count > 0:
            avg_loss_Gmain = total_loss_Gmain / batch_count
            avg_loss_G_ID = total_loss_G_ID / batch_count
            avg_loss_G_Rec = total_loss_G_Rec / batch_count
            avg_loss_G_feat_match = total_loss_G_feat_match / batch_count
            avg_loss_G_hidden = total_loss_G_hidden / batch_count
            avg_loss_Dgen = total_loss_Dgen / batch_count
            avg_loss_Dreal = total_loss_Dreal / batch_count
            avg_loss_D = total_loss_D / batch_count

            errors = {
                "G_Loss": avg_loss_Gmain,
                "G_ID": avg_loss_G_ID,
                "G_Rec": avg_loss_G_Rec,
                "G_feat_match": avg_loss_G_feat_match,
                "G_hidden": avg_loss_G_hidden,
                "D_fake": avg_loss_Dgen,
                "D_real": avg_loss_Dreal,
                "D_loss": avg_loss_D
            }

            if opt.use_tensorboard:
                for tag, value in errors.items():
                    logger.add_scalar(tag, value, step)

            message = '( step: %d, ) ' % (step)
            for k, v in errors.items():
                message += '%s: %.3f ' % (k, v)

            print(message)
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)

            total_loss_Gmain = 0
            total_loss_G_ID = 0
            total_loss_G_Rec = 0
            total_loss_G_feat_match = 0
            total_loss_G_hidden = 0
            total_loss_Dgen = 0
            total_loss_Dreal = 0
            total_loss_D = 0
            batch_count = 0

        if (step + 1) % opt.sample_freq == 0:
            model.netG.eval()
            # hidden_model.eval()
            with torch.no_grad():
                imgs        = list()
                zero_img    = (torch.zeros_like(src_image1[0,...]))
                imgs.append(zero_img.cpu().numpy())
                save_img    = ((src_image1.cpu()) * imagenet_std + imagenet_mean).numpy()
                for r in range(opt.batchSize):
                    imgs.append(save_img[r,...])
                arcface_112     = F.interpolate(src_image1,size=(112,112), mode='bicubic')
                id_vector_src1  = model.netArc(arcface_112)
                id_vector_src1  = F.normalize(id_vector_src1, p=2, dim=1)

                face_GT = src_image1 * mask1
                background_GT = src_image1 * (1 - mask1)
                background_hiding_withmask, zero_faceregion, face_inn = model.netG(input=None, dlatents=None,
                                                                                   mask=mask1, bg=background_GT,
                                                                                   face=face_GT, hidden=True)
                save_background_img = ((background_hiding_withmask.cpu()) * imagenet_std + imagenet_mean).numpy()

                for i in range(opt.batchSize):
                    image_infer = background_hiding_withmask[i, ...].repeat(opt.batchSize, 1, 1, 1)
                    img_fake    = model.netG(input=image_infer, dlatents=id_vector_src1, mask=None, bg=None, face=None, hidden=False).cpu()
                    imgs.append(save_background_img[i, ...])

                    img_fake    = img_fake * imagenet_std
                    img_fake    = img_fake + imagenet_mean
                    img_fake    = img_fake.numpy()
                    for j in range(opt.batchSize):
                        imgs.append(img_fake[j,...])
                print("Save test data")
                imgs = np.stack(imgs, axis = 0).transpose(0,2,3,1)
                plot_batch(imgs, os.path.join(sample_path, 'step_'+str(step+1)+'.jpg'))

        ### save latest model
        if (step+1) % opt.model_freq==0:
            print('saving the latest model (steps %d)' % (step+1))
            save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            save_generator_weights(model.netG, save_dir, step+1)
            np.savetxt(iter_path, (step+1, total_step), delimiter=',', fmt='%d')
