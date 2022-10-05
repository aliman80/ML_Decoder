import os
import clip
import torch
from torchvision.datasets import CIFAR100
import h5py
import pickle
import pandas as pd


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

path = '/l/users/muhammad.ali/ML_Decoder/MSCOCO/new_files/classes.csv'

# read csv from path
classes = pd.read_csv(path, header=None)


token_1k = []

for v in classes.values:
    class_name = v[1]
    sen = 'a photo of ' + class_name
    sen_tok = clip.tokenize(sen)
    token_1k.append(sen_tok)
tensor_1k = torch.cat(token_1k,0).to(device)

with torch.no_grad():
    feat_1k =  model.encode_text(tensor_1k)

#path = '/home/muhammad.ali/Desktop/Research/MLDECODER/ML_Decoder/wordvec_array_clip_pic.pth'
path = '/l/users/muhammad.ali/ML_Decoder/MSCOCO/new_files/a_photo_of/clip_array.pth'
torch.save(feat_1k.T, path)
