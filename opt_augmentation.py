import os
import torch
import tqdm
# from utils_pancre_contrast import *
from module.modules import *
from utils.utils import *
import logging
import datetime
from torch.utils.tensorboard import SummaryWriter
from diffusion.diffusion import Diffusion
from options.base_options import get_default_args
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def augmentaiton(args):
    # setup_logging(args.run_name)
    device = args.device
    # dataloader = get_data(args,state='train',ver='diff')
    #dataloader_t= get_data(args,state='test')
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

    diffusion = Diffusion(img_size=args.image_size, device=device, objective = args.objective)

    checkpoint_model = torch.load(args.pretrained)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint_model)
    
    sampled_images, classes = diffusion.sample(model, n=args.num_synthetic, num_classes = args.num_classes, channels = args.channels)
    save_image_with_label(images = sampled_images, labels = classes)

if __name__ == '__main__':
    args = get_default_args()
    augmentaiton(args)
    #torch.manual_seed(0)