from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from os.path import join
import numpy as np
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from timeit import default_timer as timer
import random
import pickle

from utils.box_utils import xyxy2xywh, xywh2xyxy


class referral_dataset(Dataset):
    def __init__(self, refs, img_dir, tokenizer, is_train_set, args, max_len=20, max_token_len = 32, temp = 0.5, K=2):
        self.refs = refs
        self.img_dir = img_dir
        self.data = []
        self.max_len = max_len
        self.max_token_len = max_token_len
        self.is_train_set = is_train_set
        
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token

        self.createEpochData(ep=0)
        


    def createPDF(self, L, temp):
        def softmax(x, t):
            expxt = np.exp(x/t)
            return expxt/np.sum(expxt)

        PDFDict = {}
        for i in range(2, L):
            PDFDict[i] = softmax(x=np.arange(1, i)/i, t=temp)
        return PDFDict


    def createEpochData(self, ep):
        del self.data
        self.data = []

        for idx, ref in enumerate(self.refs):
            img_name, category_id , sentences_orig, bbox, split= join(self.img_dir, ref['file_name']), ref['category_id'], ref['sentences'], ref['bbox'], ref['split']
            #choose only multiple word sentences
            sentences = list(filter(lambda x: len(x.split()) > 1 and len(x.split()) < self.max_len, sentences_orig))
            if len(sentences) == 0:
                continue
            collected_data = {'img_name':img_name, 'category_id':category_id, 'bbox': bbox, 'text':sentences[int(ep%len(sentences))], 'next_word':self.eos_token}
          
            self.data.append(collected_data)
        random.shuffle(self.data)
            

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        ref = self.data[idx]

        PIL_image = Image.open(join(self.img_dir, '_'.join(ref['img_name'].split('_')[:-1])+'.jpg')).convert('RGB')
        tensor_image = T.functional.to_tensor(PIL_image)
        tensor_image = F.normalize(tensor_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        h, w = tensor_image.size()[1:]
        out_mask = torch.zeros((h, w)).int()
        inputs = self.tokenizer(ref['text'], return_tensors="pt", padding='max_length', max_length=self.max_token_len)

        next_word = self.tokenizer(ref['next_word'], return_tensors="pt", add_special_tokens=False)['input_ids'][0][0]
        bbox = torch.tensor(ref['bbox'])/ torch.tensor([w, h, w, h], dtype=torch.float32)

        return {'text_inputs': inputs['input_ids'], 'text_mask': inputs['attention_mask'], 'bbox': bbox, 'category_id':ref['category_id'], \
             'src_img': tensor_image,'img_mask': out_mask, 'next_word': next_word}



class RefCollator(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        batch_bbox = []
        batch_imgs = []
        batch_img_masks = []
        batch_text_inputs = []
        batch_text_masks = []
        batch_targets = []
        batch_nextwords = []
        
        for t in batch:
            batch_imgs.append(t['src_img'].unsqueeze(0))
            batch_img_masks.append(t['img_mask'].unsqueeze(0))
            batch_text_inputs.append(t['text_inputs'])
            batch_text_masks.append(t['text_mask'])
            batch_bbox.append(t['bbox'].unsqueeze(0))
            batch_targets.append(torch.tensor(t['category_id']).unsqueeze(0))
            batch_nextwords.append(t['next_word'].unsqueeze(0))

        
        return torch.cat(batch_imgs, dim = 0), torch.cat(batch_img_masks, dim = 0), torch.cat(batch_text_inputs, dim = 0), \
            torch.cat(batch_text_masks, dim = 0), torch.cat(batch_bbox, dim = 0), torch.cat(batch_nextwords, dim = 0), torch.tensor(batch_targets)


class gaze_dataset(Dataset):
    def __init__(self, fixs, img_dir, tokenizer, args, cat_dict_file, max_len=20, max_token_len = 32):
        self.fixs = fixs
        self.img_dir = img_dir
        self.max_len = max_len
        self.max_token_len = max_token_len
        self.cat_dict = pickle.load(open(cat_dict_file, mode='rb'))
        
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token
        self.bos_token = tokenizer.bos_token
        self.pad_token = tokenizer.pad_token
        self.max_pack_len = args.max_pack_len
        self.max_context_len = args.max_context_len

            
    def __len__(self):
        return len(self.fixs)
        
    def __getitem__(self, idx):
        fix = self.fixs[idx]

        PIL_image = Image.open(join(self.img_dir, fix['IMAGEFILE'])).convert('RGB')
        tensor_image = T.functional.to_tensor(PIL_image)
        tensor_image = F.normalize(tensor_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        h, w = tensor_image.size()[1:]
        out_mask = torch.zeros((h, w)).int()

        inputs = self.tokenizer(fix['PREFIX'].replace('</s>', self.eos_token), return_tensors="pt", padding='max_length', max_length=self.max_token_len)

        next_word = self.tokenizer(fix['NEXT_WORD'].replace('</s>', self.eos_token).replace('<pad>', self.pad_token), return_tensors="pt", add_special_tokens=False)['input_ids'][0][0]
        refEnd = fix['NEXT_WORD'] == '<pad>'
        bbox = torch.tensor(fix['BBOX'])/ torch.tensor([w, h, w, h], dtype=torch.float32)

        return {'text_inputs': inputs['input_ids'], 'text_mask': inputs['attention_mask'], 'bbox': bbox, \
            'category_id':self.cat_dict[fix['REF_ID']], 'src_img': tensor_image,'img_mask': out_mask, 
            'next_word': next_word, 'tgt_y': np.array(fix['PACK_Y'][:self.max_pack_len]), 'tgt_x': np.array(fix['PACK_X'][:self.max_pack_len]), 'context_x':np.array(fix['CONTEXT_X'][:self.max_context_len]),\
            'context_y':np.array(fix['CONTEXT_Y'][:self.max_context_len]),'context_pack':np.array(fix['CONTEXT_PACK'][:self.max_context_len]),'context_order':np.array(fix['CONTEXT_ORDER'][:self.max_context_len]), 'refEnd':refEnd,\
            'scanpath_terminated': fix['SCANPATH_TERMINATED'], 'ref_gaze_id':fix['REF_GAZE_ID']}


class RefGazeCollator(object):
    def __init__(self, max_pack_len, max_context_len):
        self.max_pack_len = max_pack_len
        self.max_context_len = max_context_len
        self.PAD = [0,0,0]
        self.EOS = torch.tensor([-4]*max_pack_len)


    def __call__(self, batch):
        batch_tgt_y = []
        batch_tgt_x = []
        batch_ctxt_y = []
        batch_ctxt_x = []
        batch_ctxt_pack = []
        batch_ctxt_order = []
        batch_ctxt_masks = []
        batch_scanpath_terminations = []
        batch_bbox = []
        batch_imgs = []
        batch_img_masks = []
        batch_text_inputs = []
        batch_text_masks = []
        batch_targets = []
        batch_nextwords = []
        batch_pad_masks = []
        batch_eos_masks = []
        
        for t in batch:
            batch_imgs.append(t['src_img'].unsqueeze(0))
            batch_img_masks.append(t['img_mask'].unsqueeze(0))
            batch_text_inputs.append(t['text_inputs'])
            batch_text_masks.append(t['text_mask'])
            batch_bbox.append(t['bbox'].unsqueeze(0))
            batch_targets.append(torch.tensor(t['category_id']).unsqueeze(0))
            batch_nextwords.append(t['next_word'].unsqueeze(0))
            batch_tgt_y.append(self.EOS) if t['scanpath_terminated'] else batch_tgt_y.append(torch.tensor(t['tgt_y']))
            batch_tgt_x.append(self.EOS) if t['scanpath_terminated'] else batch_tgt_x.append(torch.tensor(t['tgt_x']))
            batch_pad_masks.append(torch.zeros_like(self.EOS, dtype=torch.uint8) if t['scanpath_terminated'] else torch.zeros(len(t['tgt_y']), dtype=torch.uint8))
            batch_eos_masks.append(torch.ones((1, self.max_pack_len), dtype=torch.uint8) if t['scanpath_terminated'] else torch.zeros((1, self.max_pack_len), dtype=torch.uint8))
            batch_ctxt_y.append(torch.tensor(t['context_y']))
            batch_ctxt_x.append(torch.tensor(t['context_x']))
            batch_ctxt_pack.append(torch.tensor(t['context_pack']))
            batch_ctxt_order.append(torch.tensor(t['context_order']))
            batch_ctxt_masks.append(torch.zeros(len(t['context_y'])))
            batch_scanpath_terminations.append(torch.ones(1, 1) if t['scanpath_terminated'] or t['refEnd'] else torch.zeros(1, 1))

        batch_tgt_y.append(torch.zeros(self.max_pack_len))
        batch_tgt_x.append(torch.zeros(self.max_pack_len))
        batch_pad_masks.append(torch.zeros(self.max_pack_len))
        batch_ctxt_y.append(torch.zeros(self.max_context_len))
        batch_ctxt_x.append(torch.zeros(self.max_context_len))
        batch_ctxt_pack.append(torch.zeros(self.max_context_len))
        batch_ctxt_order.append(torch.zeros(self.max_context_len))
        batch_ctxt_masks.append(torch.zeros(self.max_context_len))

        batch_tgt_y = pad_sequence(batch_tgt_y, padding_value=self.PAD[0])[:, :-1].unsqueeze(-1)
        batch_tgt_x = pad_sequence(batch_tgt_x, padding_value=self.PAD[1])[:, :-1].unsqueeze(-1)
        batch_pad_masks = pad_sequence(batch_pad_masks, padding_value=1)[:, :-1]
        batch_ctxt_y = pad_sequence(batch_ctxt_y, padding_value=self.PAD[0])[:, :-1].unsqueeze(-1)
        batch_ctxt_x = pad_sequence(batch_ctxt_x, padding_value=self.PAD[1])[:, :-1].unsqueeze(-1)
        batch_ctxt_pack = pad_sequence(batch_ctxt_pack, padding_value=self.PAD[2])[:, :-1].unsqueeze(-1)
        batch_ctxt_order = pad_sequence(batch_ctxt_order, padding_value=self.PAD[2])[:, :-1].unsqueeze(-1)
        batch_ctxt_masks = torch.cat([pad_sequence(batch_ctxt_masks, padding_value=1)[:, :-1], torch.zeros_like(batch_tgt_y).squeeze(-1)], dim=0)

        return {"images":torch.cat(batch_imgs, dim = 0), 
            "image_masks":torch.cat(batch_img_masks, dim = 0), 
            "text_inputs":torch.cat(batch_text_inputs, dim = 0),
            "text_masks":torch.cat(batch_text_masks, dim = 0), 
            "bounding_boxes":torch.cat(batch_bbox, dim = 0), 
            "next_words":torch.cat(batch_nextwords, dim = 0), 
            "targets":torch.tensor(batch_targets), 
            "scanpaths":torch.cat([batch_tgt_x, batch_tgt_y], dim = -1), 
            "contexts":torch.cat([batch_ctxt_x, batch_ctxt_y, batch_ctxt_pack, batch_ctxt_order], dim = -1).long().permute(1, 0, 2),
            "context_masks": batch_ctxt_masks.permute(1,0).bool(),
            "pad_masks":batch_pad_masks,
            "eos_masks":torch.cat(batch_eos_masks, dim=0).permute(1,0),
            "scanpath_terminations":torch.cat(batch_scanpath_terminations, dim=0)}

class RefGazeCollator_noeos(object):
    def __init__(self, max_pack_len, max_context_len):
        self.max_pack_len = max_pack_len
        self.max_context_len = max_context_len
        self.PAD = [0,0,0]
        self.EOS = torch.tensor([-4]*max_pack_len)


    def __call__(self, batch):
        batch_tgt_y = []
        batch_tgt_x = []
        batch_ctxt_y = []
        batch_ctxt_x = []
        batch_ctxt_pack = []
        batch_ctxt_order = []
        batch_ctxt_masks = []
        batch_scanpath_terminations = []
        batch_bbox = []
        batch_imgs = []
        batch_img_masks = []
        batch_text_inputs = []
        batch_text_masks = []
        batch_targets = []
        batch_nextwords = []
        batch_pad_masks = []
        batch_eos_masks = []
        
        for t in batch:
            batch_imgs.append(t['src_img'].unsqueeze(0))
            batch_img_masks.append(t['img_mask'].unsqueeze(0))
            batch_text_inputs.append(t['text_inputs'])
            batch_text_masks.append(t['text_mask'])
            batch_bbox.append(t['bbox'].unsqueeze(0))
            batch_targets.append(torch.tensor(t['category_id']).unsqueeze(0))
            batch_nextwords.append(t['next_word'].unsqueeze(0))
            batch_tgt_y.append(self.EOS) if t['scanpath_terminated'] else batch_tgt_y.append(torch.tensor(t['tgt_y']))
            batch_tgt_x.append(self.EOS) if t['scanpath_terminated'] else batch_tgt_x.append(torch.tensor(t['tgt_x']))
            batch_pad_masks.append(torch.zeros_like(self.EOS, dtype=torch.uint8) if t['scanpath_terminated'] else torch.zeros(len(t['tgt_y']), dtype=torch.uint8))
            batch_eos_masks.append(torch.ones((1, self.max_pack_len), dtype=torch.uint8) if t['scanpath_terminated'] else torch.zeros((1, self.max_pack_len), dtype=torch.uint8))
            batch_ctxt_y.append(torch.tensor(t['context_y']))
            batch_ctxt_x.append(torch.tensor(t['context_x']))
            batch_ctxt_pack.append(torch.tensor(t['context_pack']))
            batch_ctxt_order.append(torch.tensor(t['context_order']))
            batch_ctxt_masks.append(torch.zeros(len(t['context_y'])))
            batch_scanpath_terminations.append(torch.ones(1, 1) if t['refEnd'] else torch.zeros(1, 1))

        batch_tgt_y.append(torch.zeros(self.max_pack_len))
        batch_tgt_x.append(torch.zeros(self.max_pack_len))
        batch_pad_masks.append(torch.zeros(self.max_pack_len))
        batch_ctxt_y.append(torch.zeros(self.max_context_len))
        batch_ctxt_x.append(torch.zeros(self.max_context_len))
        batch_ctxt_pack.append(torch.zeros(self.max_context_len))
        batch_ctxt_order.append(torch.zeros(self.max_context_len))
        batch_ctxt_masks.append(torch.zeros(self.max_context_len))

        batch_tgt_y = pad_sequence(batch_tgt_y, padding_value=self.PAD[0])[:, :-1].unsqueeze(-1)
        batch_tgt_x = pad_sequence(batch_tgt_x, padding_value=self.PAD[1])[:, :-1].unsqueeze(-1)
        batch_pad_masks = pad_sequence(batch_pad_masks, padding_value=1)[:, :-1]
        batch_ctxt_y = pad_sequence(batch_ctxt_y, padding_value=self.PAD[0])[:, :-1].unsqueeze(-1)
        batch_ctxt_x = pad_sequence(batch_ctxt_x, padding_value=self.PAD[1])[:, :-1].unsqueeze(-1)
        batch_ctxt_pack = pad_sequence(batch_ctxt_pack, padding_value=self.PAD[2])[:, :-1].unsqueeze(-1)
        batch_ctxt_order = pad_sequence(batch_ctxt_order, padding_value=self.PAD[2])[:, :-1].unsqueeze(-1)
        batch_ctxt_masks = torch.cat([pad_sequence(batch_ctxt_masks, padding_value=1)[:, :-1], torch.zeros_like(batch_tgt_y).squeeze(-1)], dim=0)

        return {"images":torch.cat(batch_imgs, dim = 0), 
            "image_masks":torch.cat(batch_img_masks, dim = 0), 
            "text_inputs":torch.cat(batch_text_inputs, dim = 0),
            "text_masks":torch.cat(batch_text_masks, dim = 0), 
            "bounding_boxes":torch.cat(batch_bbox, dim = 0), 
            "next_words":torch.cat(batch_nextwords, dim = 0), 
            "targets":torch.tensor(batch_targets), 
            "scanpaths":torch.cat([batch_tgt_x, batch_tgt_y], dim = -1), 
            "contexts":torch.cat([batch_ctxt_x, batch_ctxt_y, batch_ctxt_pack, batch_ctxt_order], dim = -1).long().permute(1, 0, 2),
            "context_masks": batch_ctxt_masks.permute(1,0).bool(),
            "pad_masks":batch_pad_masks,
            "eos_masks":torch.cat(batch_eos_masks, dim=0).permute(1,0),
            "scanpath_terminations":torch.cat(batch_scanpath_terminations, dim=0)}