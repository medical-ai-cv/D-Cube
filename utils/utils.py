import os
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from glob import glob
import torch

def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))    

def save_images(images, path):
    grid = torchvision.utils.make_grid(images, nrow=10, normalize=True, scale_each=True)
    torchvision.utils.save_image(images, path,
                                 nrow=4,
                                 normalize=True, range=(-1, 1))
    
def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def save_image_with_label(images, labels):
    os.makedirs(os.path.join('./pancreas/', 'BENIGN'), exist_ok=True)
    os.makedirs(os.path.join('./pancreas/', 'MALIGNANT'), exist_ok=True)
    os.makedirs(os.path.join('./pancreas/','NORMAL'), exist_ok=True)
    folders = {
    0: './pancreas/BENIGN',
    1: './pancreas/MALIGNANT',
    2: './pancreas/NORMAL'
}
    for i in range(images.size(0)):
        image = images[i, :]
        label = labels[i].item()
        path = os.path.join(folders[label], f"syn_{i}th_{label}.png")
        torchvision.utils.save_image(image, path, nrow=1, normalize=True, range=(-1, 1))


def moving_average(ema_model, current_model, decay=0.999):
    with torch.no_grad():
        ema_param = dict(ema_model.named_parameters())
        curr_param = dict(current_model.named_parameters())
        for k in ema_param.keys():
            ema_param[k].data.mul_(decay).add_(curr_param[k].data, alpha=1 - decay)

