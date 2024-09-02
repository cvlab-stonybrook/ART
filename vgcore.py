import torch
import torch.nn as nn
import torch.nn.functional as F

from visual_model.detr import build_detr
from language_model.lm import build_lm
from vl_transformer import build_vl_transformer
from utils.box_utils import xywh2xyxy

from transformers import AutoConfig


class VGCore(nn.Module):
    def __init__(self, args):
        super(VGCore, self).__init__()
        hidden_dim = args.vl_hidden_dim
        self.num_vis_token = (args.im_h // args.patch_size) * (args.im_w // args.patch_size)
        self.num_text_token = args.max_lm_len
        self.config = AutoConfig.from_pretrained(args.lm)

        self.vismodel = build_detr(args)
        self.textmodel = build_lm(args)

        num_total = self.num_vis_token + self.num_text_token + 1 + 1 + 1
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)
        self.nextword_token = nn.Embedding(1, hidden_dim)
        self.target_token = nn.Embedding(1, hidden_dim)


        self.vis_proj = nn.Linear(self.vismodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        self.vl_transformer = build_vl_transformer(args)
        self.bbox_predictor = MLP(hidden_dim, hidden_dim, 4, 2)
        self.nextword_predictor = MLP(hidden_dim, hidden_dim, self.config.vocab_size, 1)
        self.target_predictor = MLP(hidden_dim, hidden_dim, args.num_cats, 2)

        self.predict_dropout = nn.Dropout(p=args.vl_predict_dropout)

        #self.softmax = nn.Softmax(dim=1)


    def forward(self, img_data, text_data):
        bs = img_data[0].shape[0]
        # visual backbone
        vis_mask, vis_src = self.vismodel(img_data)
        
        vis_src = self.vis_proj(vis_src) # (N*B)xC

        # language model
        text_src, text_mask = self.textmodel(text_data)
        assert text_mask is not None
        text_src = self.text_proj(text_src)[:, :self.num_text_token, :]

        # permute BxLenxC to LenxBxC
        text_src = text_src.permute(1, 0, 2)
        text_mask = text_mask.flatten(1)[:, :self.num_text_token]

        # bbox regression token
        reg_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        reg_mask = torch.zeros((bs, 1)).to(reg_src.device).to(torch.bool)

        #next word prediction token
        nextword_src = self.nextword_token.weight.unsqueeze(1).repeat(1, bs, 1)
        nextword_mask = torch.zeros((bs, 1)).to(nextword_src.device).to(torch.bool)

        #target category prediction token
        target_src = self.target_token.weight.unsqueeze(1).repeat(1, bs, 1)
        target_mask = torch.zeros((bs, 1)).to(target_src.device).to(torch.bool)

        
        vl_src = torch.cat([reg_src, nextword_src, target_src, vis_src, text_src], dim=0)
        vl_mask = torch.cat([reg_mask.bool(), nextword_mask.bool(), target_mask.bool(), vis_mask.bool(), ~text_mask.bool()], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos) # (1+1+1+L+N)xBxC
        bbox_pred, nextword_pred, target_pred = vg_hs[0], vg_hs[1], vg_hs[2]

        pred_box = self.bbox_predictor(self.predict_dropout(bbox_pred)).sigmoid()
        nextword = self.nextword_predictor(self.predict_dropout(nextword_pred))
        target = self.target_predictor(self.predict_dropout(target_pred))

        return pred_box, nextword, target, vg_hs, vl_mask


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x