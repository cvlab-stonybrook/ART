from timeit import default_timer as timer
import argparse
from datetime import datetime
from copy import deepcopy
import os
from os.path import join
import json
import pickle
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from utils.misc import NestedTensor

from utils.box_utils import generalized_box_iou, xywh2xyxy
from utils.core_utils import seed_everything, get_args_parser_pretrain, save_model_pretrain, save_model_pretrain_lite
from dataset.dataset import referral_dataset, RefCollator
from vgcore import VGCore

def train(epoch, args, model, optimizer, loss_fn_bbox, loss_fn_giou, loss_fn_nextword, loss_fn_target, train_dataloader, model_dir, model_name):
    model.train()
    bbox_losses = 0
    nextword_losses = 0
    target_losses = 0
    giou_losses = 0
    with tqdm(train_dataloader, unit="batch") as tepoch:
        minibatch = 0
        for batch_imgs, batch_img_masks, batch_text_inputs, batch_text_masks, batch_bbox, batch_nextwords, batch_targets in tepoch:
            pred_boxes, nextwords, targets, vg_hs, vl_mask = model(img_data = [batch_imgs.cuda(), batch_img_masks.cuda()], text_data = (batch_text_inputs.cuda(), batch_text_masks.cuda()))
            optimizer.zero_grad()

            bbox_loss = loss_fn_bbox(pred_boxes, batch_bbox.cuda()).mean()
            giou_loss = loss_fn_giou(pred_boxes, batch_bbox.cuda()).mean()
            nextword_loss = 0#loss_fn_nextword(nextwords, batch_nextwords.cuda())
            target_loss = loss_fn_target(targets, batch_targets.cuda())

            loss = bbox_loss + giou_loss + nextword_loss + target_loss
            
            loss.backward()
            bbox_losses += bbox_loss.item()
            giou_losses += giou_loss.item()
            nextword_losses += 0#nextword_loss.item()
            target_losses += target_loss.item()

            optimizer.step()
            minibatch += 1.
            
            tepoch.set_postfix(bbox_loss=bbox_losses/minibatch, giou_loss = giou_losses/minibatch, nextword_loss=nextword_losses/minibatch, target_loss=target_losses/minibatch)
    if not args.no_save:
        save_model_pretrain(epoch, args, model, optimizer, model_dir, model_name)
    return bbox_losses / len(train_dataloader),  giou_losses / len(train_dataloader), nextword_losses / len(train_dataloader), target_losses / len(train_dataloader)
    

def evaluate(model, loss_fn_bbox, loss_fn_giou, loss_fn_nextword, loss_fn_target, valid_dataloader):
    model.eval()
    bbox_losses = 0
    nextword_losses = 0
    target_losses = 0
    giou_losses = 0
    with tqdm(valid_dataloader, unit="batch") as tepoch:
        minibatch = 0
        for batch_imgs,  batch_img_masks, batch_text_inputs, batch_text_masks, batch_bbox, batch_nextwords, batch_targets in tepoch:
            with torch.no_grad():
                pred_boxes, nextwords, targets, vg_hs, vl_mask = model(img_data =  [batch_imgs.cuda(), batch_img_masks.cuda()], text_data = (batch_text_inputs.cuda(), batch_text_masks.cuda()))
            bbox_loss = loss_fn_bbox(pred_boxes, batch_bbox.cuda()).mean()
            giou_loss = loss_fn_giou(pred_boxes, batch_bbox.cuda()).mean()
            nextword_loss = loss_fn_nextword(nextwords, batch_nextwords.cuda())
            target_loss = loss_fn_target(targets, batch_targets.cuda())
            
            bbox_losses += bbox_loss.item()
            giou_losses += giou_loss.item()
            nextword_losses += nextword_loss.item()
            target_losses += target_loss.item()

            minibatch += 1.
            
            tepoch.set_postfix(bbox_loss=bbox_losses/minibatch, giou_loss = giou_losses/minibatch, nextword_loss=nextword_losses/minibatch, target_loss=target_losses/minibatch)

    return bbox_losses / len(valid_dataloader), giou_losses / len(valid_dataloader),  nextword_losses / len(valid_dataloader), target_losses/len(valid_dataloader)

def main(args):
    seed_everything(42)
    retraining = args.retraining
    last_checkpoint = args.last_checkpoint
    if retraining:
        model_dir = '/'.join(args.last_checkpoint.split('/')[:-1])
        args = argparse.Namespace(**json.load(open(join(model_dir, 'config.json'))))
        logfile = 'logs/pretraining/output_' + last_checkpoint.split('/')[-2].split('_')[-1]+'.txt'
    else:
        timenow = datetime.now().strftime("%d-%m-%Y-%H-%M-%S") 
        logfile = 'logs/pretraining/output_' + timenow + '.txt'
        model_dir = join(args.model_root, 'train_' + timenow)
        if not args.no_save:
            os.mkdir(model_dir)
        
        open(logfile, 'w').close()
        with open(logfile, "a") as myfile:
            myfile.write(str(vars(args)) + '\n\n')
            myfile.close()

    print(str(vars(args)) + '\n\n')
    if not args.no_save:
        with open(join(model_dir, 'config.json'), "w") as outfile:
            json.dump(vars(args), outfile)
            outfile.close()
    model_name = 'vgcore_'+str(args.num_encoder)+'E_'+str(args.batch_size)+'_'+str(args.vl_hidden_dim)+'d'
    tokenizer = AutoTokenizer.from_pretrained(args.lm)

    refs = pickle.load(open(join(args.dataset_dir, args.ref_file), mode='rb'))
    train_refs = deepcopy([i for i in refs if i['split'] == 'train'])
    val_refs = deepcopy([i for i in refs if i['split'] == 'val'])

    collate_fn = RefCollator()

    train_dataset = referral_dataset(train_refs, img_dir = args.img_dir, tokenizer=tokenizer, max_len=args.max_len, max_token_len = args.max_lm_len, is_train_set=True, args=args)
    valid_dataset = referral_dataset(val_refs, img_dir = args.img_dir, tokenizer=tokenizer, max_len=args.max_len, max_token_len = args.max_lm_len, is_train_set=False, args=args)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn = collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn = collate_fn)

    loss_fn_bbox = nn.L1Loss(reduction='none')
    loss_fn_giou =  lambda a,b: 1 - torch.diag(generalized_box_iou(xywh2xyxy(a),xywh2xyxy(b)))
    loss_fn_nextword = nn.CrossEntropyLoss()
    loss_fn_target = nn.CrossEntropyLoss()

    model = VGCore(args=args).cuda()
    start_epoch = 1

    if retraining:
        checkpoint = torch.load(last_checkpoint)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        print("Retraining from", start_epoch)
        del checkpoint

    model = torch.nn.DataParallel(model)
    
    
    vis_params = [p for n, p in model.named_parameters() if (("vismodel" in n) and p.requires_grad)]

    text_params = [p for n, p in model.named_parameters() if (("textmodel" in n) and p.requires_grad)]
    rest_params = [p for n, p in model.named_parameters() if (("vismodel" not in n) and ("textmodel" not in n) and p.requires_grad)]

    param_list = [{"params": rest_params},
                   {"params": vis_params, "lr": args.lr_visu_tra},
                   {"params": text_params, "lr": args.lm_lr},
                   ]
    optimizer = torch.optim.AdamW(param_list, lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    if retraining:
        checkpoint = torch.load(last_checkpoint)
        optimizer.load_state_dict(checkpoint['optim'])
        del checkpoint
    for epoch in range(start_epoch, args.epochs+1):
        start_time = timer()
        train_bbox_loss, train_giou_loss, train_nextword_loss, train_target_loss = train(epoch = epoch, args = args, model = model, optimizer = optimizer, 
            loss_fn_bbox=loss_fn_bbox, loss_fn_giou = loss_fn_giou, loss_fn_nextword=loss_fn_nextword, loss_fn_target=loss_fn_target, train_dataloader = train_dataloader, 
            model_dir = model_dir, model_name = model_name)
        end_time = timer()
        train_dataset.createEpochData(epoch)
        del train_dataloader
        train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn = collate_fn)

        valid_bbox_loss, valid_giou_loss, valid_nextword_loss, valid_target_loss = evaluate( model = model, 
            loss_fn_bbox=loss_fn_bbox, loss_fn_giou = loss_fn_giou, loss_fn_nextword=loss_fn_nextword, loss_fn_target=loss_fn_target, valid_dataloader = valid_dataloader)

        output_str = f"Epoch: {epoch}, Train bbox loss: {train_bbox_loss:.3f}, Train giou loss: {train_giou_loss:.3f}, Train next word loss: {train_nextword_loss:.3f}, Train target loss: {train_target_loss:.3f},  Val bbox loss: {valid_bbox_loss:.3f},  Valid giou loss: {valid_giou_loss:.3f}, Val next word loss: {valid_nextword_loss:.3f}, Valid target loss: {valid_target_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s, Saved to {model_dir+'/'+model_name}\n"
        print(output_str)
        with open(logfile, "a") as logf:
            logf.write(output_str)
            logf.close()
        if epoch > 1 and not args.no_save:
            checkpoint = torch.load(join(model_dir, model_name+'_'+str(epoch - 1)+'.pkg'), map_location='cpu')
            save_model_pretrain_lite(epoch - 1, args, checkpoint['model'], model_dir, model_name)
            del checkpoint
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Referral Core PreTrain', parents=[get_args_parser_pretrain()])
    args = parser.parse_args()
    main(args)