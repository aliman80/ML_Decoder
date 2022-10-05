import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed

from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.helper_functions.helper_functions import get_class_ids_split, update_wordvecs
from src_files.models import create_model
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

from src_files.models.tresnet.tresnet import InplacABN_to_ABN
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch NUS_WIDE infer')
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('--model-path', type=str, default='/home/muhammad.ali/Desktop/Research/MLDECODER/tresnet_l.pth')
parser.add_argument('--pic-path', type=str, default='/l/users/muhammad.ali/ML_Decoder/pics/img.jpg')
parser.add_argument('--model-name', type=str, default='tresnet_l')
parser.add_argument('--image-size', type=int, default=448)
parser.add_argument('--dataset-type', type=str, default='MS-COCO')
parser.add_argument('--th', type=float, default=0.75)
parser.add_argument('--top-k', type=float, default=20)
# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)
# parser.add_argument('gzsl', default =0,typ =int)

parser.add_argument('--resume_training', default=0, type=int)
parser.add_argument('--text_embeddings', default='wordvec', type=str, help='options: [clip, wordvec]')
parser.add_argument('--replace-image-encoder-with-clip',default= 0,type=int, help='if set to True, the image encoder is replaced with clip image encoder')
parser.add_argument('--classes_file_dir', type=str, default='/home/muhammad.ali/Desktop/Research/MLDECODER/ML_Decoder/')
parser.add_argument('--data', type=str, default='/home/muhammad.ali/Desktop/Research/MLDECODER/ML_Decoder/')




def main():
    print('Inference code on a single image')

    # parsing args
    args = parser.parse_args()
    args.zsl = 1
    args.num_of_groups = 925
    args.use_ml_decoder = 1
    args.num_classes = 925

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args, load_head=True)
    # model = model.cuda()
    # state = torch.load(args.model_path, map_location='cpu')
    # model.load_state_dict(state['model'], strict=True)
    ########### eliminate BN for faster inference ###########
    model = model.cpu()
    model = InplacABN_to_ABN(model)
    model = fuse_bn_recursively(model) # '''This function takes a sequential block and fuses the batch normalization with convolution
                                       #     :param model: nn.Sequential. Source resnet model
                                        #    :return: nn.Sequential. Converted block
    #'''
    model = model.cuda().half().eval()
    #######################################################
    print('done')

    idx_to_class = pd.read_csv(os.path.join(args.classes_file_dir, 'classes.csv'), header=None, index_col=0).to_dict()[1]
    classes_list = np.array(list(idx_to_class.values()))
    json_path = os.path.join(args.data, 'benchmark_81_v0.json')
    wordvec_array = torch.load(os.path.join(args.data, args.text_embeddings+ '_array.pth'))
    train_cls_ids, _, test_cls_ids = get_class_ids_split(json_path, idx_to_class)
    train_wordvecs = wordvec_array[..., train_cls_ids].to(torch.float16)
    test_wordvecs = wordvec_array[..., test_cls_ids].to(torch.float16)
    update_wordvecs(model, train_wordvecs=train_wordvecs)
    print('done\n')

    # doing inference
    print('loading image and doing inference...')
    im = Image.open(args.pic_path)
    im_resize = im.resize((args.image_size, args.image_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda().half() # float16 inference
    output = torch.squeeze(torch.sigmoid(model(tensor_batch)[0]))
    np_output = output.cpu().detach().numpy()


    ## Top-k predictions
    # detected_classes = classes_list[np_output > args.th]
    idx_sort = np.argsort(-np_output)
    detected_classes = np.array(classes_list)[idx_sort][1: args.top_k]
    scores = np_output[idx_sort][1: args.top_k]
    print(scores)
    idx_th = scores > args.th
    detected_classes = detected_classes[idx_th]
    print('done\n')


    # displaying image
    print('showing image on screen...')
    fig = plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.axis('tight')
    # plt.rcParams["axes.titlesize"] = 10
    plt.title("detected classes: {}".format(detected_classes))
    plt.savefig(os.path.join('/l/users/muhammad.ali/ML_Decoder/pics', Path(args.pic_path).stem + '_inference.jpg'))
    plt.show()
    print('done\n')


if __name__ == '__main__':
    main()
