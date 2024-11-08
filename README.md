# CLIP-Decoder:Multilabel Classification in ZSL settings

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ml-decoder-scalable-and-versatile/multi-label-classification-on-ms-coco)](https://paperswithcode.com/sota/multi-label-classification-on-ms-coco?p=ml-decoder-scalable-and-versatile)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ml-decoder-scalable-and-versatile/multi-label-zero-shot-learning-on-nus-wide)](https://paperswithcode.com/sota/multi-label-zero-shot-learning-on-nus-wide?p=ml-decoder-scalable-and-versatile)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ml-decoder-scalable-and-versatile/fine-grained-image-classification-on-stanford)](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford?p=ml-decoder-scalable-and-versatile)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ml-decoder-scalable-and-versatile/multi-label-classification-on-openimages-v6)](https://paperswithcode.com/sota/multi-label-classification-on-openimages-v6?p=ml-decoder-scalable-and-versatile)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ml-decoder-scalable-and-versatile/image-classification-on-cifar-100)](https://paperswithcode.com/sota/image-classification-on-cifar-100?p=ml-decoder-scalable-and-versatile)

<br> [Paper][(https://arxiv.org/abs/2406.14830) |
[Pretrained Models](MODEL_ZOO.md)  |
[Datasets](Datasets.md)

Official PyTorch Implementation

>  Muhammad Ali, Dr. Salman Khan
> <br/> MBZUAI


**Abstract**

Multi-label classification is an essential task utilized in a wide variety of real-world applications. Multi-label zero-shot learning is a method for classifying images into multiple unseen categories for which no training data is available, while in general zero-shot situations, the test set may include observed classes. The CLIP-Decoder is a novel method based on the state-of-the-art ML-Decoder attention-based head. We introduce multi-modal representation learning in CLIP-Decoder, utilizing the text encoder to extract text features and the image encoder for image feature extraction. Furthermore, we minimize semantic mismatch by aligning image and word embeddings in the same dimension and comparing their respective representations using a combined loss, which comprises classification loss and CLIP loss. This strategy outperforms other methods and we achieve cutting-edge results on zero-shot multilabel classification tasks using CLIP-Decoder. Our method achieves an absolute increase of 3.9% in performance compared to existing methods for zero-shot learning multi-label classification tasks. Additionally, in the generalized zero-shot learning multi-label classification task, our method shows an impressive increase of almost 2.3%.

<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pictures/main_pic.png" align="center" width="400""></td>
    <td class="tg-c3ow"><img src="./pictures/ms_coco_scores.png" align="center" width="400" ></td>

  </tr>
</table>
</p>

## ML-Decoder Implementation
ML-Decoder implementation is available [here](./src_files/ml_decoder/ml_decoder.py).
It can be easily integrated into any backbone using this example code:
```
ml_decoder_head = MLDecoder(num_classes) # initilization

spatial_embeddings = self.backbone(input_image) # backbone generates spatial embeddings      
 
logits = ml_decoder_head(spatial_embeddings) # transfrom spatial embeddings to logits
```

## Inference Code and Pretrained Models
See [Model Zoo](MODEL_ZOO.md)
<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/example_inference_open_images.jpeg" align="center" width="600" ></td>
  </tr>
</table>
</p>

## Training Code 

We share a full reproduction code for the article results.

### Multi-label Training Code
<br>A reproduction code for MS-COCO multi-label:
```
python train.py  \
--data=/home/datasets/coco2014/ \
--model_name=tresnet_l \
--image_size=448
```

### Single-label Training Code

Our single-label training code uses the excellent [timm](https://github.com/rwightman/pytorch-image-models) repo. Reproduction code is currently from a fork, we will work toward a full merge to the main repo.
```
git clone https://github.com/mrT23/pytorch-image-models.git
```
This is the code for A2 configuration training, with ML-Decoder (--use-ml-decoder-head=1):
```
python -u -m torch.distributed.launch --nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
./train.py \
/data/imagenet/ \
--amp \
-b=256 \
--epochs=300 \
--drop-path=0.05 \
--opt=lamb \
--weight-decay=0.02 \
--sched='cosine' \
--lr=4e-3 \
--warmup-epochs=5 \
--model=resnet50 \
--aa=rand-m7-mstd0.5-inc1 \
--reprob=0.0 \
--remode='pixel' \
--mixup=0.1 \
--cutmix=1.0 \
--aug-repeats 3 \
--bce-target-thresh 0.2 \
--smoothing=0 \
--bce-loss \
--train-interpolation=bicubic \
--use-ml-decoder-head=1
```
### ZSL Training Code
<br>First download the following files to the root path of the dataset:
     
[benchmark_81_v0.json](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/public/NUS_WIDE_ZSL/benchmark_81_v0.json) <br>
[wordvec_array.pth](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/public/NUS_WIDE_ZSL/data.csv) <br>
[data.csv](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/public/NUS_WIDE_ZSL/wordvec_array.pth) <br>

Training code for NUS-WIDE ZSL:
```
python train_zsl_nus.py  \
--data=/home/datasets/nus_wide/ \
--image_size=224
```

### New Top Results - Stanford-Cars and CIFAR 100
Using ML-Decoder classification head, we reached a top result of 96.41% on [Stanford-Cars dataset](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford), and 95.1% on [CIFAR-100 dataset](https://paperswithcode.com/sota/image-classification-on-cifar-100).
We will add this result to a future version of the paper.

## Citation
```
@misc{ridnik2021mldecoder,
      title={ML-Decoder: Scalable and Versatile Classification Head}, 
      author={Tal Ridnik and Gilad Sharir and Avi Ben-Cohen and Emanuel Ben-Baruch and Asaf Noy},
      year={2021},
      eprint={2111.12933},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
