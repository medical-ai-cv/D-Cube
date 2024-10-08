import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import glob
import natsort

def get_data_step2(data_dir, state='train', batch_size=32, size=128):
    if state == 'train':
        dataset = ImageDataset(data_dir, state=state, im_size=(size, size))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return dataloader
    else:
        dataset = ImageDataset(data_dir, state=state, im_size=(size, size))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return dataloader
    

class ImageDataset(Dataset):
    def __init__(self, data_dir, state='train', im_size=(128, 128)):
        if state == 'train':
            self.data_dir_init = os.path.join(data_dir, 'Train')
        else:
            self.data_dir_init = os.path.join(data_dir, 'Test')

        self.img_list = []
        self.class_list = []
        self.file_list = natsort.natsorted(os.listdir(self.data_dir_init))
        num = 0

        for file in self.file_list:
            img_path = os.path.join(self.data_dir_init, file, '*')
            img_path_test = natsort.natsorted(glob.glob(img_path))
            self.img_list += img_path_test
            self.class_list += [num] * len(img_path_test)
            num += 1
        self.im_size = im_size
        print(self.im_size)
        transform_list_aug  = [
            transforms.RandomHorizontalFlip(p=0.5)
        ]
        transform_resize_128  = [
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5, inplace=True)]
        transform_resize_256  = [
            transforms.Resize(size= self.im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5, inplace=True)]
        
        self.transform = transforms.RandomHorizontalFlip(p=1.0)
        self.transform_aug = transforms.Compose(transform_list_aug)
        self.transform_resize_128 = transforms.Compose(transform_resize_128)
        self.transform_resize_256 = transforms.Compose(transform_resize_256)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.class_list[index]
        img = Image.open(img_path).convert('L')
        img = self.transform_aug(img) 
         
        img_fe = self.transform_resize_256(img)
        img_128 = self.transform_resize_128(img) 

        img_fe_flip = self.transform(img_fe)
        img_128_flip = self.transform(img_128)

        return img_128, label, img_fe, img_128_flip, img_fe_flip 
    def __len__(self):
        return len(self.img_list)

