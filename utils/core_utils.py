import numpy as np
import os
import random
import torch
import argparse
from os.path import join


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def get_args_parser_pretrain():
    parser = argparse.ArgumentParser('Gaze Transformer Pretrainer', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--vm_lr', default=1e-4, type=float)
    parser.add_argument('--lm_lr', default=1e-5, type=float)
    parser.add_argument('--lr_visu_cnn', default=1e-5, type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--detr_enc_num', default=6, type=int)
    parser.add_argument('--coatt_lr', default=1e-4, type=float)
    parser.add_argument('--dataset_dir', default= './data', type=str)
    parser.add_argument('--ref_file', default= 'refTrainAll_512X320.pkl', type=str)
    parser.add_argument('--img_dir', default= './data/images_gaze', type=str)
    parser.add_argument('--lm', default= 'roberta-base', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--max_len', default=20, type=int)
    parser.add_argument('--max_lm_len', default=32, type=int)
    parser.add_argument('--num_encoder', default=6, type=int)

    # DETR parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=0, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--im_h', default=320, type=int, help='image vertical size')
    parser.add_argument('--im_w', default=512, type=int, help='image horizontal size')
    parser.add_argument('--patch_size', default=32, type=int, help='image horizontal size')
    '''parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')'''

    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')
    parser.add_argument('--vl_predict_dropout', default=0.3, type=float,
                        help='Number of encoders in the vision-language transformer')

    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--img_hidden_dim', default=2048, type=int)
    parser.add_argument('--lm_hidden_dim', default=768, type=int)
    parser.add_argument('--encoder_dropout', default=0.1, type=float)
    # parser.add_argument('--substrings_num', default=0, type=int)
    # parser.add_argument('--sample_T', default=0.1, type=float)
    parser.add_argument('--num_cats', default=100, type=int)
    parser.add_argument('--frozen_lm', default=False, action='store_true')
    parser.add_argument('--retraining', default=False, action='store_true')
    parser.add_argument('--last_checkpoint', default='./checkpoints/pretraining/vgcore_6E_32_512d_100.pkg', type=str)
    parser.add_argument('--model_root', default='./checkpoints/pretraining/', type=str)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--no_save', default=False, action='store_true')
    parser.add_argument('--comment', default='all', type=str)
    return parser

def get_args_parser_train():
    parser = argparse.ArgumentParser('Gaze Transformer Trainer', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--vm_lr', default=1e-7, type=float)
    parser.add_argument('--lm_lr', default=1e-7, type=float)
    parser.add_argument('--lr_visu_cnn', default=1e-7, type=float)
    parser.add_argument('--lr_visu_tra', default=1e-7, type=float)
    parser.add_argument('--vg_core_lr', default=1e-5, type=float)
    parser.add_argument('--decoder_lr', default=1e-4, type=float)
    parser.add_argument('--refgaze_rest_lr', default=1e-4, type=float)
    parser.add_argument('--detr_enc_num', default=6, type=int)
    parser.add_argument('--dataset_dir', default= './data', type=str)
    parser.add_argument('--train_file', default= 'refcocogaze_train_correct_tf_512X320_6.json', type=str)
    parser.add_argument('--val_file', default= 'refcocogaze_val_correct_tf_512X320_6.json', type=str)
    parser.add_argument('--cat_dict_file', default= './data/catDict.pkl', type=str)
    parser.add_argument('--img_dir', default= './data/images_512X320', type=str)
    parser.add_argument('--lm', default= 'roberta-base', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--max_len', default=20, type=int)
    parser.add_argument('--max_lm_len', default=32, type=int)
    parser.add_argument('--num_encoder', default=6, type=int)
    parser.add_argument('--im_h', default=320, type=int, help='image vertical size')
    parser.add_argument('--im_w', default=512, type=int, help='image horizontal size')
    parser.add_argument('--patch_size', default=32, type=int, help='image horizontal size')

    # DETR parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=0, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')


    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')
    parser.add_argument('--vl_predict_dropout', default=0.2, type=float,
                        help='Prediction dropout in the vision-language transformer')


    parser.add_argument('--num_decoder_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--decoder_dropout', default=0.2, type=float,
                        help='Decoder dropout in the RefGaze model')
    parser.add_argument('--predictor_dropout', default=0.4, type=float,
                        help='Prediction dropout  in the RefGaze model')
    

    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--img_hidden_dim', default=2048, type=int)
    parser.add_argument('--lm_hidden_dim', default=768, type=int)
    parser.add_argument('--encoder_dropout', default=0.1, type=float)
    parser.add_argument('--max_pack_len', default=6, type=int)
    parser.add_argument('--max_context_len', default=36, type=int)
    parser.add_argument('--num_cats', default=100, type=int)
    parser.add_argument('--frozen_lm', default=False, action='store_true')
    parser.add_argument('--retraining', default=False, action='store_true')
    parser.add_argument('--pretrained_vgcore', default=False, action='store_true')
    parser.add_argument('--vgcore_model_checkpoint', default='./checkpints/pretraining/vgcore_6E_128_256d_200.pkg', type=str)
    parser.add_argument('--last_checkpoint', default='./checkpoints/training/refgaze_6E_32_512d_100.pkg', type=str)
    parser.add_argument('--model_root', default='./checkpoints/training/', type=str)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--no_save', default=False, action='store_true')
    parser.add_argument('--comment', default='all', type=str)

    return parser


def get_args_parser_test():
    parser = argparse.ArgumentParser('Gaze Transformer Tester', add_help=False)
    parser.add_argument('--lr_visu_cnn', default=0, type=float)
    parser.add_argument('--lr_visu_tra', default=0, type=float)
    parser.add_argument('--detr_enc_num', default=6, type=int)
    parser.add_argument('--dataset_dir', default= './data', type=str)
    parser.add_argument('--test_file', default= 'refcocogaze_test_correct_512X320.json', type=str)
    parser.add_argument('--cat_dict_file', default= './data/catDict.pkl', type=str)
    parser.add_argument('--img_dir', default= './data/images_512X320', type=str)
    parser.add_argument('--lm', default= 'roberta-base', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--max_len', default=20, type=int)
    parser.add_argument('--max_lm_len', default=32, type=int)
    parser.add_argument('--num_encoder', default=6, type=int)
    parser.add_argument('--im_h', default=320, type=int, help='image vertical size')
    parser.add_argument('--im_w', default=512, type=int, help='image horizontal size')
    parser.add_argument('--patch_size', default=32, type=int, help='image horizontal size')
    parser.add_argument('--num_samples', default=10, type=int, help='Number of scanpaths to sample for each cae')

    # DETR parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=0, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')


    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')
    parser.add_argument('--vl_predict_dropout', default=0.2, type=float,
                        help='Prediction dropout in the vision-language transformer')


    parser.add_argument('--num_decoder_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--decoder_dropout', default=0.2, type=float,
                        help='Decoder dropout in the RefGaze model')
    parser.add_argument('--predictor_dropout', default=0.4, type=float,
                        help='Prediction dropout  in the RefGaze model')
    

    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--img_hidden_dim', default=2048, type=int)
    parser.add_argument('--lm_hidden_dim', default=768, type=int)
    parser.add_argument('--encoder_dropout', default=0.1, type=float)
    parser.add_argument('--max_pack_len', default=6, type=int)
    parser.add_argument('--max_context_len', default=36, type=int)
    parser.add_argument('--num_cats', default=100, type=int)
    parser.add_argument('--frozen_lm', default=False, action='store_true')
    parser.add_argument('--checkpoint', default='./checkpoints/refgaze_6E_6D_64_256d_100.pkg', type=str)
    parser.add_argument('--num_workers', default=6, type=int)

    return parser

def save_model_pretrain(epoch, args, model, optim, model_dir, model_name):
    state = {
        "epoch": epoch,
        "args": vars(args),
        "model":
        model.module.state_dict()
        if hasattr(model, "module") else model.state_dict(),
        "optim":
        optim.state_dict(),   
    }
    torch.save(state, join(model_dir, model_name+'_'+str(epoch)+'.pkg'))

def save_model_pretrain_lite(epoch, args, model_state_dict, model_dir, model_name):
    state = {
        "epoch": epoch,
        "args": vars(args),
        "model":
        model_state_dict   
    }
    torch.save(state, join(model_dir, model_name+'_'+str(epoch)+'.pkg'))

def save_model_train(epoch, args, model, optim, model_dir, model_name):
    state = {
        "epoch": epoch,
        "args": vars(args),
        "model":
        model.module.state_dict()
        if hasattr(model, "module") else model.state_dict(),
        "optim":
        optim.state_dict(),   
    }
    torch.save(state, join(model_dir, model_name+'_'+str(epoch)+'.pkg'))

def save_model_train_lite(epoch, args, model_state_dict, model_dir, model_name):
    state = {
        "epoch": epoch,
        "args": vars(args),
        "model":
        model_state_dict   
    }
    torch.save(state, join(model_dir, model_name+'_'+str(epoch)+'.pkg'))