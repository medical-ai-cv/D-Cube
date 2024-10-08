import os
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import glob
import random
import natsort

def get_data_step1(data_dir, state='train', batch_size=32, size=128,ver='diff'):
    sub_train=pancreas(data_dir=data_dir,state='train',im_size=(size,size))
    test_dataset=pancreas(data_dir=data_dir,state='test',im_size=(size,size))
    if state=='train' and ver == 'diff':
        dataset = ContrastiveDataset(sub_train)
    elif state=='train' and ver=='class':
        dataset=sub_train
    else:
        dataset = test_dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=False,num_workers=8)
    return dataloader

class pancreas(Dataset):
    def __init__(self,data_dir,state='train',im_size=(128,128)):
        if state=='train':
            self.data_dir_init = data_dir + '/'+'Train'
        else:
            self.data_dir_init=data_dir + '/'+'Test'

            
        self.img_list=[]
        self.class_list=[]
        self.file_list=natsort.natsorted(os.listdir(self.data_dir_init))
        num=0
        for file in self.file_list:
            self.img_path=self.data_dir_init+'/'+file+'/*'
            self.img_path_test=natsort.natsorted(glob.glob(self.img_path))
            self.img_list += self.img_path_test
            self.class_list += [num]*int(len(self.img_path_test))
            num += 1
        
        self.im_size=im_size
        self.transform=transforms.Compose(
                    [transforms.Resize(im_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5,std=0.5)]
                )
    def __getitem__(self,idx):
        img_path=self.img_list[idx]
        label=self.class_list[idx]
        img=Image.open(img_path).convert('L')
        img=self.transform(img)
        return img,label


    def __len__(self):
        return len(self.img_list)

class ContrastiveDataset(Dataset):
    def __init__(self,dataset):
        self.dataset=dataset

    def __getitem__(self, index):
        img1= self.dataset[index][0]
        label1=self.dataset[index][1]
        
        if random.randint(0, 1):
            while True:
                index2 = random.randint(0, len(self.dataset)-1)
                img2= self.dataset[index2][0]
                
                label2=self.dataset[index2][1]
                if label1 == label2:
                    break
        else:
            while True:
                index2 = random.randint(0, len(self.dataset)-1)
                img2= self.dataset[index2][0]
               
                label2=self.dataset[index2][1]
                if label1 != label2:
                    break
        
        contrastive_label = int(label1 != label2)

        return img1, img2, contrastive_label,label1,label2

    def __len__(self):
        return len(self.dataset)

