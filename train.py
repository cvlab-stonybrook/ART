from timeit import default_timer as timer
import argparse
from datetime import datetime
from copy import deepcopy
import os
from os.path import join
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils.box_utils import generalized_box_iou, xywh2xyxy
from utils.core_utils import seed_everything, get_args_parser_train, save_model_train, save_model_train_lite
from dataset.dataset import gaze_dataset, RefGazeCollator
from refgaze import RefGaze


def train(epoch, args, model, optimizer, loss_fn_bbox, loss_fn_giou, loss_fn_nextword, loss_fn_target, loss_fn_xy, loss_fn_token, train_dataloader, model_dir, model_name):
    model.train()
    bbox_losses = []
    nextword_losses = []
    target_losses = []
    giou_losses = []
    reg_losses = []
    token_losses = []
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            batch_imgs,  batch_img_masks, batch_text_inputs, batch_text_masks, batch_bbox, \
            batch_nextwords, batch_targets, batch_tgt, batch_context, batch_context_mask, batch_pad, batch_eos, batch_terminations = batch["images"].cuda(), batch["image_masks"].cuda(),\
                                            batch["text_inputs"].cuda(),batch["text_masks"].cuda(), batch["bounding_boxes"].cuda(), batch["next_words"].cuda(), batch["targets"].cuda(),\
                                            batch["scanpaths"].cuda(), batch["contexts"].cuda(), batch["context_masks"].cuda(), batch["pad_masks"].cuda(), batch["eos_masks"].cuda(),\
                                            batch['scanpath_terminations'].cuda()
            pred_boxes, nextwords, targets, token_predictions, scanpaths_x, scanpaths_y = model(img_data = [batch_imgs, batch_img_masks], text_data = (batch_text_inputs, batch_text_masks),\
                                                                context = batch_context, context_padding_mask = batch_context_mask)
            optimizer.zero_grad()

            scanpaths_x = torch.clamp(scanpaths_x, min=0, max=args.im_w - 1)
            scanpaths_y = torch.clamp(scanpaths_y, min=0, max=args.im_h - 1)
            batch_terminations = batch_terminations.squeeze(-1)
            batch_terminations_sum = batch_terminations.sum() + 1e-5

            fixation_mask = torch.logical_not(batch_pad + batch_eos).float().permute(1,0)
            token_gt =  batch_pad + batch_eos * 2
            

            #calculate grounding and target loss only post termination by human subject

            bbox_loss = (loss_fn_bbox(pred_boxes, batch_bbox).mean(dim=-1) * batch_terminations).sum() / batch_terminations_sum
            giou_loss = (loss_fn_giou(pred_boxes, batch_bbox) * batch_terminations).sum()/ batch_terminations_sum
            nextword_loss = 0#loss_fn_nextword(nextwords, batch_nextwords)  
            target_loss = loss_fn_target(targets[batch_terminations.bool(), :], batch_targets[batch_terminations.bool()]) if batch_terminations.sum() > 0 else 0

            reg_loss =(((loss_fn_xy(scanpaths_x.float().squeeze(-1), batch_tgt[:, :, 0].permute(1,0)) + \
                loss_fn_xy(scanpaths_y.float().squeeze(-1), batch_tgt[:, :, 1].permute(1,0)))*fixation_mask).sum(-1)/(fixation_mask.sum(-1)+1e-5)).mean()
            #predict padding, end of fixation or valid fixation
            token_loss = loss_fn_token(token_predictions.permute(0,2,1), token_gt.permute(1,0).long())


            if batch_terminations.sum() > 0:
                loss = bbox_loss + giou_loss + target_loss + nextword_loss + reg_loss + token_loss
            else:
                loss = bbox_loss + giou_loss + nextword_loss + reg_loss + token_loss
            
            loss.backward()
            if batch_terminations.sum() > 0:
                bbox_losses += [bbox_loss.item()]
                giou_losses += [giou_loss.item()]
                nextword_losses += [0]
                target_losses += [target_loss.item()]
            reg_losses += [reg_loss.item()]
            token_losses += [token_loss.item()]

            optimizer.step()
            
            tepoch.set_postfix(token_loss=np.mean(token_losses), reg_loss=np.mean(reg_losses), bbox_loss=np.mean(bbox_losses), giou_loss = np.mean(giou_losses),\
                 nextword_loss=np.mean(nextword_losses), target_loss=np.mean(target_losses))
    if not args.no_save:
        save_model_train(epoch, args, model, optimizer, model_dir, model_name)
    return np.mean(bbox_losses),  np.mean(giou_losses), np.mean(nextword_losses), np.mean(target_losses),\
        np.mean(reg_losses), np.mean(token_losses)
    

def evaluate(model, args, loss_fn_bbox, loss_fn_giou, loss_fn_nextword, loss_fn_target, loss_fn_xy, loss_fn_token, valid_dataloader):
    model.eval()
    bbox_losses = []
    nextword_losses = []
    target_losses = []
    giou_losses = []
    reg_losses = []
    token_losses = []
    with tqdm(valid_dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            batch_imgs,  batch_img_masks, batch_text_inputs, batch_text_masks, batch_bbox, \
            batch_nextwords, batch_targets, batch_tgt, batch_context, batch_context_mask, batch_pad, batch_eos, batch_terminations = batch["images"].cuda(), batch["image_masks"].cuda(),\
                                            batch["text_inputs"].cuda(),batch["text_masks"].cuda(), batch["bounding_boxes"].cuda(), batch["next_words"].cuda(), batch["targets"].cuda(),\
                                            batch["scanpaths"].cuda(), batch["contexts"].cuda(), batch["context_masks"].cuda(), batch["pad_masks"].cuda(), batch["eos_masks"].cuda(),\
                                            batch['scanpath_terminations'].cuda()
            with torch.no_grad():
                pred_boxes, nextwords, targets, token_predictions, scanpaths_x, scanpaths_y = model(img_data = [batch_imgs, batch_img_masks], text_data = (batch_text_inputs, batch_text_masks),\
                                                                context = batch_context, context_padding_mask = batch_context_mask)
            scanpaths_x = torch.clamp(scanpaths_x, min=0, max=args.im_w - 1)
            scanpaths_y = torch.clamp(scanpaths_y, min=0, max=args.im_h - 1)
            batch_terminations = batch_terminations.squeeze(-1)
            batch_terminations_sum = batch_terminations.sum() + 1e-5

            fixation_mask = torch.logical_not(batch_pad + batch_eos).float().permute(1,0)
            token_gt =  batch_pad + batch_eos * 2
            

            #calculate grounding and target loss only post termination by human subject

            bbox_loss = (loss_fn_bbox(pred_boxes, batch_bbox).mean(dim=-1) * batch_terminations).sum() / batch_terminations_sum
            giou_loss = (loss_fn_giou(pred_boxes, batch_bbox) * batch_terminations).sum()/ batch_terminations_sum
            nextword_loss = loss_fn_nextword(nextwords, batch_nextwords)  
            target_loss = loss_fn_target(targets[batch_terminations.bool(), :], batch_targets[batch_terminations.bool()]) if batch_terminations.sum() > 0 else 0

            reg_loss =(((loss_fn_xy(scanpaths_x.float().squeeze(-1), batch_tgt[:, :, 0].permute(1,0)) + \
                loss_fn_xy(scanpaths_y.float().squeeze(-1), batch_tgt[:, :, 1].permute(1,0)))*fixation_mask).sum(-1)/(fixation_mask.sum(-1)+1e-5)).mean()
            #predict padding, end of fixation or valid fixation
            token_loss = loss_fn_token(token_predictions.permute(0,2,1), token_gt.permute(1,0).long())

            if batch_terminations.sum() > 0:
                bbox_losses += [bbox_loss.item()]
                giou_losses += [giou_loss.item()]
                nextword_losses += [nextword_loss.item()]
                target_losses += [target_loss.item()]
            reg_losses += [reg_loss.item()]
            token_losses += [token_loss.item()]

            
            
            tepoch.set_postfix(token_loss=np.mean(token_losses), reg_loss=np.mean(reg_losses), bbox_loss=np.mean(bbox_losses), giou_loss = np.mean(giou_losses),\
                 nextword_loss=np.mean(nextword_losses), target_loss=np.mean(target_losses))

    return np.mean(bbox_losses),  np.mean(giou_losses), np.mean(nextword_losses), np.mean(target_losses),\
        np.mean(reg_losses), np.mean(token_losses)

def main(args):
    seed_everything(42)
    retraining = args.retraining
    last_checkpoint = args.last_checkpoint
    if retraining:
        model_dir = '/'.join(args.last_checkpoint.split('/')[:-1])
        args = argparse.Namespace(**json.load(open(join(model_dir, 'config.json'))))
        logfile = 'logs/training/output_' + last_checkpoint.split('/')[-2].split('_')[-1]+'.txt'
        args.pretrained_vgcore = False
    else:
        timenow = datetime.now().strftime("%d-%m-%Y-%H-%M-%S") 
        logfile = 'logs/training/output_' + timenow + '.txt'
        model_dir = join(args.model_root, 'train_' + timenow)
        if not args.no_save:
            os.mkdir(model_dir)
        
        open(logfile, 'w').close()
        with open(logfile, "a") as myfile:
            myfile.write(str(vars(args)) + '\nNo next word loss!!\n')
            myfile.close()

    print(str(vars(args)) + '\n\n')
    if not args.no_save:
        with open(join(model_dir, 'config.json'), "w") as outfile:
            json.dump(vars(args), outfile)
            outfile.close()
    model_name = 'refgaze_'+str(args.num_encoder)+'E_'+str(args.num_decoder_layers)+'D_'+str(args.batch_size)+'_'+str(args.vl_hidden_dim)+'d'
    tokenizer = AutoTokenizer.from_pretrained(args.lm)

    collate_fn = RefGazeCollator(args.max_pack_len, args.max_context_len)

    train_refgazes = json.load(open(join(args.dataset_dir, args.train_file), mode='r'))
    val_refgazes = json.load(open(join(args.dataset_dir, args.val_file), mode='r'))

    train_dataset = gaze_dataset(fixs=train_refgazes, img_dir = args.img_dir, tokenizer=tokenizer, args=args, cat_dict_file = args.cat_dict_file, max_len=args.max_len, max_token_len = args.max_lm_len)
    valid_dataset = gaze_dataset(fixs=val_refgazes, img_dir = args.img_dir, tokenizer=tokenizer, args=args, cat_dict_file = args.cat_dict_file, max_len=args.max_len, max_token_len = args.max_lm_len)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn = collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn = collate_fn)

    loss_fn_bbox = nn.L1Loss(reduction='none')
    loss_fn_giou =  lambda a,b: 1 - torch.diag(generalized_box_iou(xywh2xyxy(a),xywh2xyxy(b)))
    loss_fn_nextword = nn.CrossEntropyLoss()
    loss_fn_target = nn.CrossEntropyLoss()
    loss_fn_xy = nn.L1Loss(reduction='none')
    loss_fn_token = torch.nn.NLLLoss()

    model = RefGaze(args=args, pretrained_vgcore=args.pretrained_vgcore, vgcore_checkpoint=args.vgcore_model_checkpoint).cuda()
    start_epoch = 1

    if retraining:
        checkpoint = torch.load(last_checkpoint)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        print("Retraining from", start_epoch)
        del checkpoint
    

    model = torch.nn.DataParallel(model)

    vis_params = [p for n, p in model.named_parameters() if (("vismodel" in n and "vgcore" in n) and p.requires_grad)]

    text_params = [p for n, p in model.named_parameters() if (("textmodel" in n and "vgcore" in n) and p.requires_grad)]
    rest_vgcore_params = [p for n, p in model.named_parameters() if (("vismodel" not in n) and ("textmodel" not in n)  and "vgcore" in n and p.requires_grad)]
    decoder_params = model.module.decoder.parameters()
    rest_params = list(model.module.token_predictor.parameters()) + list(model.module.generator_y_mu.parameters()) + list(model.module.generator_x_mu.parameters()) + \
                list(model.module.generator_y_logvar.parameters()) + list(model.module.generator_x_logvar.parameters())


    param_list = [{"params": rest_vgcore_params, "lr": args.vg_core_lr},
                   {"params": vis_params, "lr": args.vm_lr},
                   {"params": text_params, "lr": args.lm_lr},
                   {"params": decoder_params, "lr": args.decoder_lr},
                   {"params": rest_params, "lr": args.refgaze_rest_lr}
                   ]
    optimizer = torch.optim.AdamW(param_list, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    if retraining:
        checkpoint = torch.load(last_checkpoint)
        optimizer.load_state_dict(checkpoint['optim'])
        del checkpoint

    for epoch in range(start_epoch, args.epochs+1):
        start_time = timer()
        train_bbox_loss, train_giou_loss, train_nextword_loss, train_target_loss, train_reg_loss, train_token_loss = train(epoch = epoch, args = args, model = model, optimizer = optimizer, \
            loss_fn_bbox=loss_fn_bbox, loss_fn_giou = loss_fn_giou, loss_fn_nextword=loss_fn_nextword, loss_fn_target=loss_fn_target, loss_fn_xy=loss_fn_xy, loss_fn_token= loss_fn_token, \
            train_dataloader = train_dataloader, model_dir = model_dir, model_name = model_name)
        end_time = timer()
        train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn = collate_fn)

        valid_bbox_loss, valid_giou_loss, valid_nextword_loss, valid_target_loss, val_reg_loss, val_token_loss = evaluate(model = model, args = args,
            loss_fn_bbox=loss_fn_bbox, loss_fn_giou = loss_fn_giou, loss_fn_nextword=loss_fn_nextword, loss_fn_target=loss_fn_target, loss_fn_xy=loss_fn_xy, loss_fn_token= loss_fn_token,
            valid_dataloader = valid_dataloader)

        output_str = f"Epoch: {epoch}, Train Reg loss: {train_reg_loss:.3f}, Val Reg loss: {val_reg_loss:.3f}, Train Token loss: {train_token_loss:.3f}, \
            Val Token loss: {val_token_loss:.3f},Train bbox loss: {train_bbox_loss:.3f}, Train giou loss: {train_giou_loss:.3f}, \
                Train next word loss: {train_nextword_loss:.3f}, Train target loss: {train_target_loss:.3f}, Val bbox loss: {valid_bbox_loss:.3f},  \
                    Valid giou loss: {valid_giou_loss:.3f}, Val next word loss: {valid_nextword_loss:.3f}, Valid target loss: {valid_target_loss:.3f}, \
                        Epoch time = {(end_time - start_time):.3f}s, Saved to {model_dir+'/'+model_name}\n"
        print(output_str)
        with open(logfile, "a") as logf:
            logf.write(output_str)
            logf.close()
        if epoch > 1 and not args.no_save:
            checkpoint = torch.load(join(model_dir, model_name+'_'+str(epoch - 1)+'.pkg'), map_location='cpu')
            save_model_train_lite(epoch - 1, args, checkpoint['model'], model_dir, model_name)
            del checkpoint
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gaze Refer Train', parents=[get_args_parser_train()])
    args = parser.parse_args()
    main(args)