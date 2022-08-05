# import json
import os
import argparse
from matplotlib.pyplot import text
# from subprocess import call

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

from src_files.helper_functions.helper_functions import mAP, f1, CutoutPIL, ModelEma, \
    add_weight_decay, get_datasets_from_csv, update_wordvecs
from src_files.models import create_model
from src_files.loss_functions.losses import AsymmetricLoss, CLIPLoss

from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
import pickle
import torch.multiprocessing as tmul
import numpy as np
from sklearn.metrics import f1_score, recall_score
import wandb
from src_files.helper_functions.helper_functions import mAP, CocoDetection, AverageMeter
import time
import clip

from src_files.models.utils.prompt_learner import PromptLearner, TextEncoder, PLCLIP


parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', type=str, default='/home/muhammad.ali/Desktop/Research/MLDECODER/ML_Decoder/')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-name', default='tresnet_m')
parser.add_argument('--model-path', default='/home/muhammad.ali/Desktop/Research/MLDECODER/tresnet_m21k.pth', type=str)
parser.add_argument('--num-classes', default=925)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--image_size', default=224, type=int,
                    metavar='N', help='input image size (default: 224)')
parser.add_argument('--batch-size', default=28, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--thr', default=0.75, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('--print-freq', '-p', default=32, type=int,
                    metavar='N', help='print frequency (default: 64)')
                    

# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
# CLIP
parser.add_argument('--replace-image-encoder-with-clip',default= 0,type=int, help='if set to True, the image encoder is replaced with clip image encoder')
parser.add_argument('--text-embeddings', default='wordvec', type=str, help='the text embedings to load, options=["wordvec","clip"]')
parser.add_argument('--add-clip-loss', default=0, type=int)
parser.add_argument('--clip-loss-temp', default=0.1, type=float) #change clip loss
parser.add_argument('--clip-loss-weight', default=1, type=float)
parser.add_argument('--classif-loss-weight', default=1.0, type=float)
parser.add_argument('--gzsl', default=0, type=int)

parser.add_argument('--resume_training', default=0, type=int)
parser.add_argument('--exp_name', default = 'test',type= str)
parser.add_argument('--validate_only', default=0, type=int)
parser.add_argument('--best_metric', default='mAP', type=str, help='options=["mAP","average"]')

parser.add_argument('--add-clip-head', default=0, type=int)

parser.add_argument('--use_prompt_learner', default=0, type=int)
parser.add_argument('--prompt_learner_n_ctx', default=16, type=int, help='number of context vectors')
parser.add_argument('--prompt_learner_ctx_init', default="", type=str, help='initialization words')
parser.add_argument('--prompt_learner_csc', default=0, type=int, help='if set to True, class-specific context is used')
parser.add_argument('--prompt_learner_prec', default="fp16", type=str, help='options=["fp16","fp32", "amp"]')
parser.add_argument('--prompt_learner_class_token_position', default="end", type=str, help='options=["middle","end", "front]')

def main():
    args = parser.parse_args()
    wandb.init(config=args)
    # wandb.define_metric('mAP', summary='max')
    #NUS-WIDE defaults
    args.zsl = 1
    args.num_of_groups = 925
    args.use_ml_decoder = 1
    args.num_classes = 925

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args).cuda()
    print('done')

    #NUS-WIDE Data loading
    #json_path = os.path.join(args.data, '/home/muhammad.ali/Desktop/Research/MLDECODER/benchmark_81_v0.json')
    #json_path = os.path.join(args.data, '/home/muhammad.ali/Desktop/Research/MLDECODER/data.csv')
    json_path = os.path.join(args.data, 'benchmark_81_v0.json')
    wordvec_array = torch.load(os.path.join(args.data, args.text_embeddings+ '_array.pth'))


    train_transform = transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ])
    val_transform = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ])
    train_dataset, val_dataset, train_cls_ids, test_cls_ids, classnames = get_datasets_from_csv(args, args.data,
                              args.data,
                              train_transform, val_transform,
                              json_path)
    # wordvec_array += torch.randn_like(wordvec_array)
    train_wordvecs = wordvec_array[..., train_cls_ids].float()
   # noise = np.random.normal(0,1,300*925).reshape(300,925)
    # x = torch.zeros(300, 925, 100, dtype=torch.float64)  # augmentations as well as random augmentations
    # x = x + (0.1**0.5)*torch.randn(300, 925, 100)
   # train_wordvecs = train_wordvecs + noise
    
    
    test_wordvecs = wordvec_array[..., test_cls_ids].float()
    # if args.gzsl:
    #     test_wordvecs = torch.cat((test_wordvecs, train_wordvecs), axis=1)
    print('classes {}'.format(len(train_dataset.classes)))
    print('train_cls_ids {} test_cls_ids {} '.format(train_cls_ids.shape, test_cls_ids.shape))

    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    clip_model = None
    clip_criterion = None
    pl_clip = None
    if args.add_clip_loss:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load('RN50', device)
        clip_criterion = CLIPLoss(args.clip_loss_temp, args.decoder_embedding, 1024, 512, device)
    
        if args.use_prompt_learner:
            pl_clip = PLCLIP(args, classnames, clip_model, device)
        
    tmul.set_sharing_strategy('file_system')
    # Actuall Training
    train_multi_label_zsl(args, model, clip_model, clip_criterion, pl_clip, train_loader, val_loader, args.lr, train_wordvecs, test_wordvecs)


def train_multi_label_zsl(args, model, clip_model, clip_criterion, pl_clip, train_loader, val_loader, lr, train_wordvecs=None,
                          test_wordvecs=None):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 40
    weight_decay = 1e-2 # 5e-3
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    if args.add_clip_loss:
        optimizer.add_param_group({'params': list(clip_model.parameters()) + list(clip_criterion.parameters())})
        # clip_optimizer = torch.optim.Adam(params=list(clip_model.parameters()) + list(clip_criterion.parameters()), lr=lr, weight_decay=0)
    steps_per_epoch =  len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    
   
    for epoch in range(Epochs):
        if not args.validate_only:
            update_wordvecs(model, train_wordvecs)
            
            for i, input in enumerate(train_loader):
                inputData = input['image'].cuda()
                target = input['target'].cuda()  # (batch,3,num_classes)
                with autocast():  # mixed precision
                    output, image_embeddings = model(inputData) # sigmoid will be done in loss !
                    output = output.float()  
                    if args.add_clip_loss:
                        clip_tokens = input['clip_tokens'].cuda()
                        text_features = clip_model.encode_text(clip_tokens)
                        clip_loss, _, _ = clip_criterion(image_embeddings, text_features)
                loss = criterion(output, target) * args.classif_loss_weight
                if args.add_clip_loss:
                    loss += clip_loss * args.clip_loss_weight
                    # clip_loss.backward()
                    # clip_optimizer.step()
                    # clip_optimizer.zero_grad()

                model.zero_grad()

                scaler.scale(loss).backward()
                # loss.backward()

                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()

                scheduler.step()

                ema.update(model)
                # store information
                if i % 100 == 0:
                    trainInfoList.append([epoch, i, loss.item()])
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                        .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                                scheduler.get_last_lr()[0], \
                                loss.item()))
                    wandb.log({"loss": loss})
                    if args.add_clip_loss:
                        wandb.log({"clip_loss": clip_loss})

            try:
                torch.save(model.state_dict(), os.path.join(
                    'models',args.exp_name, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            except:
                pass

        model.eval()
        update_wordvecs(model, test_wordvecs=test_wordvecs)
        update_wordvecs(ema.module, test_wordvecs=test_wordvecs)
       

        mAP_score, f1_3, p_3, r_3, f1_5, p_5, r_5, clip_losses = validate_multi(args, val_loader, model, ema, clip_model, clip_criterion)
        model.train()
        if args.add_clip_loss:
            print('current_clip_loss = {:.2f}\n'.format(torch.mean(torch.Tensor(clip_losses))))

        if args.best_metric == 'mAP':
            if mAP_score > highest_mAP: 
                highest_mAP = mAP_score
                f1_3_at_highest_mAP = f1_3
                p_3_at_highest_mAP = p_3
                r_3_at_highest_mAP = r_3
                f1_5_at_highest_mAP = f1_5
                p_5_at_highest_mAP = p_5
                r_5_at_highest_mAP = r_5
                try:
                    torch.save(model.state_dict(), os.path.join(
                        'models', args.exp_name,'model-highest-'+str(epoch)+'.ckpt'))
                except:
                    pass
            print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))
            print('current_f1_k3 = {:.2f}, f1_k3_at_highest_mAP = {:.2f}\n'.format(f1_3, f1_3_at_highest_mAP))
            print('current_p_k3 = {:.2f}, p_k3_at_highest_mAP = {:.2f}\n'.format(p_3, p_3_at_highest_mAP))
            print('current_r_k3 = {:.2f}, r_k3_at_highest_mAP = {:.2f}\n'.format(r_3, r_3_at_highest_mAP))
            print('current_f1_k5 = {:.2f}, f1_k5_at_highest_mAP = {:.2f}\n'.format(f1_5, f1_5_at_highest_mAP))
            print('current_p_k5 = {:.2f}, p_k5_at_highest_mAP = {:.2f}\n'.format(p_5, p_5_at_highest_mAP))
            print('current_r_k5 = {:.2f}, r_k5_at_highest_mAP = {:.2f}\n'.format(r_5, r_5_at_highest_mAP))
        
        elif args.best_metric == 'average':
            current_average = np.mean([mAP_score, f1_3, p_3, r_3, f1_5, p_5, r_5])
            if current_average > highest_average:
                highest_average = current_average
                f1_3_at_highest_average = f1_3
                p_3_at_highest_average = p_3
                r_3_at_highest_average = r_3
                f1_5_at_highest_average = f1_5
                p_5_at_highest_average = p_5
                r_5_at_highest_average = r_5
                try:
                    torch.save(model.state_dict(), os.path.join(
                        'models', args.exp_name,'model-highest-average-'+str(epoch)+'.ckpt'))
                except:
                    pass
            print('current_average = {:.2f}, highest_average = {:.2f}\n'.format(current_average, highest_average))
            print('current_f1_k3 = {:.2f}, f1_k3_at_highest_average = {:.2f}\n'.format(f1_3, f1_3_at_highest_average))
            print('current_p_k3 = {:.2f}, p_k3_at_highest_average = {:.2f}\n'.format(p_3, p_3_at_highest_average))
            print('current_r_k3 = {:.2f}, r_k3_at_highest_average = {:.2f}\n'.format(r_3, r_3_at_highest_average))
            print('current_f1_k5 = {:.2f}, f1_k5_at_highest_average = {:.2f}\n'.format(f1_5, f1_5_at_highest_average))
            print('current_p_k5 = {:.2f}, p_k5_at_highest_average = {:.2f}\n'.format(p_5, p_5_at_highest_average))
            print('current_r_k5 = {:.2f}, r_k5_at_highest_average = {:.2f}\n'.format(r_5, r_5_at_highest_average))

        if args.best_metric == 'mAP':
            wandb.log({
                "highest_mAP": highest_mAP,
                "f1_3_at_highest_mAP": f1_3_at_highest_mAP,
                "p_3_at_highest_mAP": p_3_at_highest_mAP,
                "r_3_at_highest_mAP": r_3_at_highest_mAP,
                "f1_5_at_highest_mAP": f1_5_at_highest_mAP,
                "p_5_at_highest_mAP": p_5_at_highest_mAP,
                "r_5_at_highest_mAP": r_5_at_highest_mAP
            })
        elif args.best_metric == 'average':
            wandb.log({
                "highest_average": highest_average,
                "f1_3_at_highest_average": f1_3_at_highest_average,
                "p_3_at_highest_average": p_3_at_highest_average,
                "r_3_at_highest_average": r_3_at_highest_average,
                "f1_5_at_highest_average": f1_5_at_highest_average,
                "p_5_at_highest_average": p_5_at_highest_average,
                "r_5_at_highest_average": r_5_at_highest_average
            })
        if args.add_clip_loss:
            logs = {}
            logs["val_clip_loss"] = torch.mean(torch.Tensor(clip_losses))
            wandb.log(logs)
        

        if args.validate_only:
            break

def validate_multi(args, val_loader, model, ema_model, clip_model, clip_criterion):
    print("starting validation")
    batch_time = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()
    mAP_meter = AverageMeter()
    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    clip_losses = []
    for i, data_dict in enumerate(val_loader):
        input = data_dict['image']
        target = data_dict['target']
        # compute output
        with torch.no_grad():
            with autocast():
                if args.add_clip_loss:
                    clip_tokens = data_dict['clip_tokens'].cuda()
                output_regular, image_embeddings = model(input.cuda())
                if args.add_clip_loss:
                    text_features = clip_model.encode_text(clip_tokens)
                    clip_loss, _, _ = clip_criterion(image_embeddings, text_features)
                    clip_losses.append(clip_loss.item())
                output_regular = Sig(output_regular).cpu()
                output_ema, _ = ema_model.module(input.cuda())
                output_ema = Sig(output_ema).cpu()
                # output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())
        
         # measure accuracy and record loss
        pred = output_regular.data.gt(args.thr).long()

        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()

        this_prec = this_tp.float() / (
            this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (
            this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        prec.update(float(this_prec), input.size(0))
        rec.update(float(this_rec), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
               i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                i, len(val_loader), batch_time=batch_time,
                prec=prec, rec=rec))
            print(
                'P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
                    .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))
            if args.add_clip_loss:
                print("Clip loss: {:.4f}".format(torch.mean(torch.Tensor(clip_losses))))

    print(
        '--------------------------------------------------------------------')
    print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
          .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))
        
   
    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    # rc_3, rc_5 =    recall(torch.cat(preds_regular), torch.cat(targets), k_val=3), recall(torch.cat(preds_regular), torch.cat(targets), k_val=5)
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    # print("recall score regular {:.2f}, rc_3 {:.2f},rc_5 {:.2f}".format(rc_3, rc_5))
    f1_3, p_3, r_3 = f1(torch.cat(preds_regular), torch.cat(targets), 'overall', k_val=3)
    f1_5, p_5, r_5 = f1(torch.cat(preds_regular), torch.cat(targets), 'overall', k_val=5)
    
     
    print("fi score k=3: {:.2f}. f1 score k=5: {:.2f}".format(f1_3, f1_5))
    return max(mAP_score_regular, mAP_score_ema), f1_3, p_3, r_3, f1_5, p_5, r_5, clip_losses


if __name__ == '__main__':
    main()
