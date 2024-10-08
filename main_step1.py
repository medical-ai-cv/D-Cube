import os
import torch
import torch.nn as nn
import tqdm
from torch import optim
from utils.utils_step1 import *
from utils.utils import *
# from utils import *
from module.modules import *
import logging
from torch.utils.tensorboard import SummaryWriter
from diffusion.diffusion import Diffusion
from options.base_options import get_default_args

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def contrastive_loss(output1, output2, target, margin):
    mse = nn.MSELoss(reduction='none')
    output1=output1.to('cuda')
    output2=output2.to('cuda')
    target=target.to('cuda')
    margin=margin.to('cuda')
    euclidean_distance = torch.sum(mse(output1, output2),dim=1)
    loss_contrastive = ((1 - target) * euclidean_distance +
                                  (target) * torch.clamp(margin - euclidean_distance, min=0.0))
    return loss_contrastive

def train(args):
    args.run_name = "diffusion_{}_{}_{}".format(args.objective, args.schedule, args.folder_name)
    setup_logging(args.run_name)
    print(args.run_name)
    device = args.device

    dataloader = get_data_step1(args.dataset_path,state='train', batch_size = args.batch_size,size = args.image_size,ver='diff')

    model = Cont_UViT(img_size=128,
        patch_size=8,
        in_chans=1,
        embed_dim=512,
        depth=12,
        num_heads=8,
        num_classes=args.num_classes,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False).to('cuda')
  
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    model = torch.nn.DataParallel(model)

    crit_con = contrastive_loss
    mse = nn.MSELoss(reduction='none')
    diffusion = Diffusion(img_size = args.image_size, device = device, objective = args.objective)
    logger = SummaryWriter(os.path.join("logs", args.run_name))
    l = len(dataloader)
    margin=torch.tensor(args.margin)

    for i in range(args.epochs):
        epoch = i + 1
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm.tqdm(dataloader)
        for i, image in enumerate(pbar):
            images1, images2, con_lab, label1, label2 = image
            images1 = images1.float().to(device)
            images2 = images2.float().to(device)
            con_lab = con_lab.long().to(args.device)
            label1 = label1.long().to(args.device)
            label2 = label2.long().to(args.device)
            
            t = diffusion.sample_timesteps(images1.shape[0]).to(device)
            x_t1, noise1 = diffusion.noise_images(images1, t, z=None)
            x_t2, _ = diffusion.noise_images(images2, t, z=noise1)
            predicted_noise1, predicted_noise2, middle_feature1, middle_feature2 = model(x_t1, x_t2, t, label1,label2)
            
            loss1 = sum_flat(mse(noise1, predicted_noise1))
            loss2 = sum_flat(mse(noise1, predicted_noise2))
      
            loss_con = crit_con(middle_feature1, middle_feature2, con_lab, margin)
       
            loss_final=(loss_con+loss1+loss2).sum()
            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()
            pbar.set_postfix(MSE=loss_final.item())
            logger.add_scalar("MSE", loss_final.item(), global_step = epoch * l + i)
            
        if epoch % args.save_term == 0:
            sampled_images, classes = diffusion.sample(model, n = args.samplings, num_classes = args.num_classes, channels = args.channels)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.png"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"{epoch}.pt"))


if __name__ == '__main__':
    args = get_default_args()
    train(args)
