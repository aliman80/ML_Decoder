import torchvision.datasets as dset
import torchvision.transforms as transforms
import h5py
import torch
import eval_clip
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd

class CLIP_Data(Dataset):
    def __init__(self, dir_, img_listpath,transform,preprocess):
        super(CLIP_Data, self).__init__()
        self.dir_ = dir_
        self.img_listpath = img_listpath
        self.imglist = pd.read_csv(img_listpath, header=None)
        self.transform = transform
        self.preprocess =preprocess

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        filename = os.path.join(self.dir_, self.imglist.iloc[index,0].replace('\\','/'))
        img = Image.open(filename)
        if self.transform != None:
            img = self.transform(img)
        img = self.preprocess(img)
        return  img 


##########CLIPCODE############

batch_size = 64
workers = 2
dataroot = '/home/muhammad.ali/Mul_Lab/Generative_MLZSL/datasets/extract_features/data/Flickr'
train_img = '/home/muhammad.ali/Mul_Lab/Generative_MLZSL/datasets/extract_features/ImageList/TrainImagelist.txt'
test_img = '/home/muhammad.ali/Mul_Lab/Generative_MLZSL/datasets/extract_features/ImageList/TestImagelist.txt'
transform=transforms.Compose([
                    transforms.Resize((224,224)),  #bilinear interpolation
                    # transforms.CenterCrop(224),
                    # transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

# dataset = dset.ImageFolder(root=dataroot,
#                            transform=transform
#                            )

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = eval_clip.load("ViT-B/32", device=device)

dataset = CLIP_Data(dataroot, train_img, None, preprocess)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers)


img_features = []
for i, data in enumerate(dataloader, 0):
    data = data.to(device)
    with torch.no_grad():
        image_features = model.encode_image(data)
        img_features.append(image_features.cpu().numpy())


read_filename = "/home/muhammad.ali/Mul_Lab/Generative_MLZSL/datasets/NUS-WIDE/nus_wide_vgg_features/nus_seen_train_vgg19.h5"
rf = h5py.File(read_filename, 'r')
labels = rf['labels']

write_filename = "/home/muhammad.ali/Mul_Lab/Generative_MLZSL/datasets/NUS-WIDE/nus_wide_clip_features/nus_seen_train_clip.h5"
wf = h5py.File(write_filename,'w')
img_features= np.concatenate(img_features)
features = wf.create_dataset("features", data=img_features)
labels = wf.create_dataset("labels", data=labels)
wf.close()
rf.close()
