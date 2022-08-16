import os
import clip
import torch
from torchvision.datasets import CIFAR100
import h5py
import pickle
import pandas as pd

#src_att = pickle.load(open('/home/muhammad.ali/Mul_Lab/Generative_MLZSL/datasets/NUS-WIDE/word_embedding/NUS_WIDE_pretrained_clip-512', 'rb'))

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

path = '/home/muhammad.ali/Mul_Lab/Generative_MLZSL/datasets/NUS-WIDE/NUS_WID_Tags/TagList1k.txt'

tag_1k = pd.read_csv(path, header=None)
path = '/home/muhammad.ali/Mul_Lab/Generative_MLZSL/datasets/NUS-WIDE/ConceptsList/Concepts81.txt'

tag_81 = pd.read_csv(path, header=None)


token_1k = []

for v in tag_1k.values:
    class_name = v[0]
    sen = 'a photo of ' + class_name
    sen_tok = clip.tokenize(sen)
    token_1k.append(sen_tok)
tensor_1k = torch.cat(token_1k,0).to(device)

token_81 =[]
for v in tag_81.values:
    class_name = v[0]
    sen = 'a photo of ' + class_name
    sen_tok = clip.tokenize(sen)
    token_81.append(sen_tok)
tensor_81 = torch.cat(token_81,0).to(device)

with torch.no_grad():
    feat_1k =  model.encode_text(tensor_1k)
    feat_81 =  model.encode_text(tensor_81)

path = '/home/muhammad.ali/Mul_Lab/Generative_MLZSL/datasets/NUS-WIDE/word_embedding'
file = open(path + '/NUS_WIDE_pretrained_clip-512', 'wb')
pickle.dump([feat_1k.cpu().numpy(), feat_81.cpu().numpy()], file)
file.close()
