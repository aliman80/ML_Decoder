import itertools
import os
from copy import deepcopy
import random
import time
from copy import deepcopy

import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
from PIL import ImageDraw
from pycocotools.coco import COCO
import json
import torch.utils.data as data
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
import clip



def parse_args(parser):
    # parsing args
    args = parser.parse_args()
    return args


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i
#######ADDING RECALL#############
def recall(predictions, labels, k_val):
    idx = predictions.topk(dim=1, k=k_val)[1] 
    predictions.scatter_(dim=1,index=idx,src=torch.ones(predictions.size(0),k_val))
    mask = predictions == 1
    TP = (labels[mask] == 1).sum().float()
    tpfn = (labels == 1).sum().float()
    recall = TP/tpfn
    return 100 * recall

def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


def f1(predictions, labels, mode_F1, k_val):
        ## cuda F1 computation
        idx = predictions.topk(dim=1, k=k_val)[1] 
        #tmux predictions.fill_(0)
        predictions.scatter_(dim=1,index=idx,src=torch.ones(predictions.size(0),k_val))
        if mode_F1 == 'overall':
            # print('evaluation overall!! cannot decompose into classes F1 score')
            mask = predictions == 1
            TP = (labels[mask] == 1).sum().float()
            tpfp = mask.sum().float()
            tpfn = (labels == 1).sum().float()
            p = TP/ tpfp
            r = TP/tpfn
            f1 = 2*p*r/(p+r+(1e-8))
        else:
            num_class = predictions.shape[1]
            # print('evaluation per classes')
            f1 = np.zeros(num_class)
            p = np.zeros(num_class)
            r = np.zeros(num_class)
            for idx_cls in range(num_class):
                prediction = np.squeeze(predictions[:, idx_cls])
                label = np.squeeze(labels[:, idx_cls])
                if np.sum(label > 0) == 0:
                    continue
                binary_label = np.clip(label, 0, 1)
                f1[idx_cls] = f1_score(binary_label, prediction)
                p[idx_cls] = precision_score(binary_label, prediction)
                r[idx_cls] = recall_score(binary_label, prediction)
        return f1 * 100, p * 100, r * 100


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def get_class_ids_split(json_path, classes_dict):
    with open(json_path) as fp:
        split_dict = json.load(fp)
    if 'train class' in split_dict:
        only_test_classes = False
    else:
        only_test_classes = True

    train_cls_ids = set()
    val_cls_ids = set()
    test_cls_ids = set()

    # classes_dict = self.learn.dbunch.dataset.classes
    for idx, (i, current_class) in enumerate(classes_dict.items()):
        if only_test_classes:  # base the division only on test classes
            if current_class in split_dict['test class']:
                test_cls_ids.add(idx)
            else:
                val_cls_ids.add(idx)
                train_cls_ids.add(idx)
        else:  # per set classes are provided
            if current_class in split_dict['train class']:
                train_cls_ids.add(idx)
            # if current_class in split_dict['validation class']:
            #     val_cls_ids.add(i)
            if current_class in split_dict['test class']:
                test_cls_ids.add(idx)

    train_cls_ids = np.fromiter(train_cls_ids, np.int32)
    val_cls_ids = np.fromiter(val_cls_ids, np.int32)
    test_cls_ids = np.fromiter(test_cls_ids, np.int32)
    return train_cls_ids, val_cls_ids, test_cls_ids


def update_wordvecs(model, train_wordvecs=None, test_wordvecs=None):
    if hasattr(model, 'fc'):
        if train_wordvecs is not None:
            model.fc.decoder.query_embed = train_wordvecs.transpose(0, 1).cuda()
        else:
            model.fc.decoder.query_embed = test_wordvecs.transpose(0, 1).cuda()
    elif hasattr(model, 'head'):
        if train_wordvecs is not None:
            model.head.decoder.query_embed = train_wordvecs.transpose(0, 1).cuda()
        else:
            model.head.decoder.query_embed = test_wordvecs.transpose(0, 1).cuda()
    else:
        print("model is not suited for ml-decoder")
        exit(-1)


def default_loader(path):
    img = Image.open(path)
    return img.convert('RGB')
    # return Image.open(path).convert('RGB')

class DatasetFromList(data.Dataset):
    """From List dataset."""

    def __init__(self, root, impaths, labels, sentences, add_clip_loss, idx_to_class,
                 transform=None, target_transform=None, class_ids=None,
                 loader=default_loader):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.classes = idx_to_class
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = tuple(zip(impaths, labels))
        self.class_ids = class_ids
        self.add_clip_loss = add_clip_loss
        if add_clip_loss:
            self.sentences = sentences
            self.clip_tokens = torch.cat([clip.tokenize(sentence) for sentence in sentences])


    def __getitem__(self, index):
        data_dict = {}
        impath, target = self.samples[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        data_dict['image'] = img
        if self.target_transform is not None:
            target = self.target_transform([target])
        target = self.get_targets_multi_label(np.array(target))
        if self.class_ids is not None:
            target = target[self.class_ids]
        data_dict['target'] = target
        if self.add_clip_loss:
            data_dict['clip_tokens'] = self.clip_tokens[index]
        return data_dict

    def __len__(self):
        return len(self.samples)

    def get_targets_multi_label(self, target):
        # Full (non-partial) labels
        labels = np.zeros(len(self.classes))
        labels[target] = 1
        target = labels.astype('float32')
        return target

def parse_csv_data(args, dataset_local_path, metadata_local_path):
    try:
        df = pd.read_csv(os.path.join(metadata_local_path, "data.csv"))
    except FileNotFoundError:
        # No data.csv in metadata_path. Try dataset_local_path:
        metadata_local_path = dataset_local_path
        df = pd.read_csv(os.path.join(metadata_local_path, "data.csv"))
    images_path_list = df.values[:, 0]
    # images_path_list = [os.path.join(dataset_local_path, images_path_list[i]) for i in range(len(images_path_list))]
    labels = df.values[:, 1]
    image_labels_list = [labels.replace('[', "").replace(']', "").split(', ') for labels in
                             labels]

    if df.values.shape[1] == 3:  # split provided
        valid_idx = [i for i in range(len(df.values[:, 2])) if df.values[i, 2] == 'val']
        train_idx = [i for i in range(len(df.values[:, 2])) if df.values[i, 2] == 'train']
    else:
        valid_idx = None
        train_idx = None

    # logger.info("em: end parsr_csv_data: num_labeles: %d " % len(image_labels_list))
    # logger.info("em: end parsr_csv_data: : %d " % len(image_labels_list))

    image_sentences_list = []
    for img_list in image_labels_list:
        sen = args.clip_prompt 
        for i in range(len(img_list)):
            if len(sen + img_list[i]) < 77:
                if i == len(img_list) - 2:
                    sen += img_list[i] + ' and '
                elif i == len(img_list) - 1:
                    sen += img_list[i]
                else:
                    sen += img_list[i] + ', '
        image_sentences_list.append(sen)
    return images_path_list, image_labels_list, image_sentences_list, train_idx, valid_idx


def multilabel2numeric(multilabels):
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(multilabels)
    classes = multilabel_binarizer.classes_
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    multilabels_numeric = []
    for multilabel in multilabels:
        labels = [class_to_idx[label] for label in multilabel]
        multilabels_numeric.append(labels)
    return multilabels_numeric, class_to_idx, idx_to_class


def get_datasets_from_csv(args, dataset_local_path, metadata_local_path, train_transform,
                          val_transform, json_path):

    images_path_list, image_labels_list, image_sentences_list, train_idx, valid_idx = parse_csv_data(args, dataset_local_path, metadata_local_path)
    labels, class_to_idx, idx_to_class = multilabel2numeric(image_labels_list)

    images_path_list_train = [images_path_list[idx] for idx in train_idx]
    image_labels_list_train = [labels[idx] for idx in train_idx]
    image_sentences_list_train = [image_sentences_list[idx] for idx in train_idx]

    images_path_list_val = [images_path_list[idx] for idx in valid_idx]
    image_labels_list_val = [labels[idx] for idx in valid_idx]
    image_sentences_list_val = [image_sentences_list[idx] for idx in valid_idx]

    train_cls_ids, _, test_cls_ids = get_class_ids_split(json_path, idx_to_class)

    if args.gzsl:
        test_cls_ids = np.concatenate((train_cls_ids,test_cls_ids))
    
    train_dl = DatasetFromList(dataset_local_path, images_path_list_train, image_labels_list_train, image_sentences_list_train, args.add_clip_loss or args.add_distil_loss,
                               idx_to_class, 
                               transform=train_transform, class_ids=train_cls_ids)

    val_dl = DatasetFromList(dataset_local_path, images_path_list_val, image_labels_list_val, image_sentences_list_val, args.add_clip_loss or args.add_distil_loss, 
                              idx_to_class, 
                              transform=val_transform, class_ids=test_cls_ids)

    classnames = list(set(itertools.chain.from_iterable(image_labels_list)))
    return train_dl, val_dl, train_cls_ids, test_cls_ids, classnames


def generate_clip_array(args, model, device):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load('ViT-B/32', device)
    path = os.path.join(args.data, 'classes.csv')
    classes = pd.read_csv(path, header=None)
    token_1k = []
    for v in classes.values:
        class_name = v[1]
        sen = args.clip_prompt + class_name
        sen_tok = clip.tokenize(sen)
        token_1k.append(sen_tok)
    tensor_1k = torch.cat(token_1k,0).to(device)

    with torch.no_grad():
        feat_1k =  model.encode_text(tensor_1k)
    
    return feat_1k.T
    