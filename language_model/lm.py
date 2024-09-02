from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn


class LM(nn.Module):
    def __init__(self, model_checkpoint: str, frozen: bool, hidden_dim: int):
        super().__init__()
        self.model_checkpoint = model_checkpoint
        self.model = AutoModel.from_pretrained(model_checkpoint)
        self.num_channels = hidden_dim
       
        if frozen:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)


    def forward(self, text):
        text_ids, text_mask = text
        lm_feats = self.model(input_ids=text_ids, attention_mask=text_mask)['last_hidden_state']
        return lm_feats, text_mask

def build_lm(args):
    model = LM(model_checkpoint=args.lm,
               frozen=args.frozen_lm,
               hidden_dim=args.lm_hidden_dim)

    return model