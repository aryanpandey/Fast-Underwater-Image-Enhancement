import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
from PIL import Image
import time

class EUVP(Dataset):
    def __init__(self, root_dir, img_dim, transform = True, train = True) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.img_dim = img_dim
        self.transform = transform
        self.train = train
        
        self._init_dataset()
        if transform:
            self._init_transform()

    def _init_dataset(self):
        self.blurred = []
        self.high_res = []

        if self.train:
            dirs = os.listdir(self.root_dir)
            for dir in dirs:
                imgs = os.listdir(os.path.join(self.root_dir, dir, 'trainA'))
                for j in imgs:
                    self.blurred.append(os.path.join(self.root_dir, dir, 'trainA', j))
                    self.high_res.append(os.path.join(self.root_dir, dir, 'trainB', j))
            print("Number of Images in the Training Set: ", len(self.blurred))
            
        else:
            imgs = os.listdir(os.path.join(self.root_dir, 'Inp'))
            for j in imgs:
                self.blurred.append(os.path.join(self.root_dir, 'Inp', j))
                self.high_res.append(os.path.join(self.root_dir, 'GTr', j))
                
            print("Number of Images in the Test Set: ", len(self.blurred))
                    
    def _init_transform(self):
        self.transform_img = T.Compose([
            T.Resize((self.img_dim[0], self.img_dim[1])),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __getitem__(self, index):
        blurred = Image.open(self.blurred[index]).convert('RGB')
        high_res = Image.open(self.high_res[index]).convert('RGB')
        
        if self.transform:
            blurred = self.transform_img(blurred)
            high_res = self.transform_img(high_res)

        return blurred, high_res

    def __len__(self):
        return len(self.blurred)
