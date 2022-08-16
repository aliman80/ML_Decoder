import logging
import os
from urllib import request
import clip
import torch.nn as nn
import torch

from ...ml_decoder.ml_decoder import add_ml_decoder_head

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL


def create_model(args,load_head=False):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name == 'tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name == 'tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name == 'tresnet_xl':
        model = TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    ####################################################################################
    if args.use_ml_decoder:
        model = add_ml_decoder_head(model,num_classes=args.num_classes,num_of_groups=args.num_of_groups,
                                    decoder_embedding=args.decoder_embedding, zsl=args.zsl, text_embedding=args.text_embeddings)
    ####################################################################################
    if args.replace_image_encoder_with_clip:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load('RN50', device)
        clip_model.visual.attnpool = nn.Identity()
        model.body = clip_model.visual
    
    if args.resume_training:
        model.load_state_dict(torch.load(args.model_path))
        return model

    # loading pretrain model
    model_path = args.model_path
    if args.model_name == 'tresnet_m' and os.path.exists("/home/muhammad.ali/Desktop/Research/MLDECODER/tresnet_m21k.pth"):
    #if args.model_name == 'tresnet_l' and os.path.exists("/home/muhammad.ali/Desktop/Research/MLDECODER/tresnet_l.pth"):  
        model_path = "/home/muhammad.ali/Desktop/Research/MLDECODER/tresnet_m21k.pth"
       # model_path =""d
    if model_path:  # make sure to load pretrained model
        if not os.path.exists(model_path):
            print("downloading pretrain model...")
            request.urlretrieve(args.model_path, "./tresnet_m.pth")
            # model_path = "/home/muhammad.ali/Desktop/Research/MLDECODER/tresnet_m.pth" #"/home/muhammad.ali/Desktop/Research/MLDECODER/tresnet_l.pth"
            print('done')
        state = torch.load(model_path, map_location='cpu')
        if not load_head:
            if 'model' in state:
                key = 'model'
            else:
                key = 'state_dict'
            filtered_dict = {k: v for k, v in state[key].items() if
                             (k in model.state_dict() and 'head.fc' not in k)}
            model.load_state_dict(filtered_dict, strict=False)
        else:
            model.load_state_dict(state[key], strict=True)
            

    return model
    