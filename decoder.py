from typing import Optional
import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from positional_encodings import PositionEmbeddingSine2d, PositionEmbeddingSine1d


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerDecoderWrapper(nn.Module):
    def __init__(self, activation, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.activation = _get_activation_fn(activation)
        decoder_layer = TransformerDecoderLayer(self.hidden_dim, args.nhead, args.dim_feedforward,
                                                args.decoder_dropout, activation, normalize_before=False)
        decoder_norm = nn.LayerNorm(self.hidden_dim)
        self.decoder = TransformerDecoder(decoder_layer, args.num_decoder_layers, decoder_norm,
                                          return_intermediate=False)

        self.context_len = args.max_context_len
        self.num_packs = args.max_len + 2
        self.pack_len = args.max_pack_len

        self.querypos_embed = nn.Embedding(args.max_pack_len, self.hidden_dim)
        self.contextpos_embed = nn.Embedding(self.context_len, self.hidden_dim)
        self.num_vis_token = (args.im_h // args.patch_size) * (args.im_w // args.patch_size)
        self.num_text_token = args.max_lm_len
        num_total = self.num_vis_token + self.num_text_token + 1 + 1 + 1
        self.memorypos_embed = nn.Embedding(num_total, self.hidden_dim)
        #context embeddings
        self.xy_embed= PositionEmbeddingSine2d((args.im_h, args.im_w), hidden_dim=self.hidden_dim * 2, normalize=True, flatten = False).pos
        self.pack_embed = PositionEmbeddingSine1d(self.num_packs,hidden_dim=self.hidden_dim, normalize=True).pos # nn.Embedding(self.num_packs, self.hidden_dim)
        self.order_embed = PositionEmbeddingSine1d(self.pack_len,hidden_dim=self.hidden_dim, normalize=True).pos # nn.Embedding(self.pack_len, self.hidden_dim)
        self.seg_context = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.seg_current = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.context_agg = nn.Linear(self.hidden_dim * 4, self.hidden_dim)

        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_context(self, tensor):
        xy_embedding = self.xy_embed.to(tensor.device)[tensor[:, :, 1], tensor[:,:, 0], :]
        pack_embedding = self.pack_embed.to(tensor.device)[tensor[:, :, 2]]
        order_embedding = self.order_embed.to(tensor.device)[tensor[:, :, 3]]
        return self.context_agg(torch.cat([xy_embedding, pack_embedding, order_embedding], dim=-1))

    def with_seg_embed(self, tensor, seg: Optional[Tensor]):
        return tensor if seg is None else tensor + seg
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else pos + tensor

    def forward(self,  memory, context, 
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        tgt_curr = self.with_pos_embed(torch.zeros(self.pack_len, memory.size(1), self.hidden_dim, device=self.querypos_embed.weight.device), pos=self.querypos_embed.weight.unsqueeze(1).float())
        tgt_context = self.with_pos_embed(self.encode_context(context).permute(1,0,2), pos = self.contextpos_embed.weight.unsqueeze(1).float())
        seg_context = self.with_seg_embed(tgt_context, self.seg_context)
        seg_curr = self.with_seg_embed(tgt_curr, self.seg_current)
        tgt = torch.cat([seg_context, seg_curr], dim=0)
        #decoder
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask,
                              querypos_embed = self.querypos_embed.weight.unsqueeze(1).float(),\
                                memorypos_embed = self.memorypos_embed.weight.unsqueeze(1).float(), contextpos_embed = self.contextpos_embed.weight.unsqueeze(1).float())
        return output[-self.pack_len:, :, :]


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                memorypos_embed: Optional[Tensor] = None,
                contextpos_embed: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for idx, layer in enumerate(self.layers):
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           querypos_embed = querypos_embed, 
                           memorypos_embed = memorypos_embed,
                           contextpos_embed = contextpos_embed)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output



class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else pos + tensor

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                memorypos_embed: Optional[Tensor] = None,
                contextpos_embed: Optional[Tensor] = None):
        tgtpos_embed = torch.cat([contextpos_embed, querypos_embed], dim = 0)
        
        q = k = v = self.with_pos_embed(tgt, tgtpos_embed)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=None,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, tgtpos_embed),
                                   key=self.with_pos_embed(memory, memorypos_embed),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                memorypos_embed: Optional[Tensor] = None,
                contextpos_embed: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        tgtpos_embed = torch.cat([contextpos_embed.weight.unsqueeze(1), querypos_embed.weight.unsqueeze(1)], dim = 0)
        q = k = v = self.with_pos_embed(tgt2, tgtpos_embed)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, tgtpos_embed),
                                   key=self.with_pos_embed(memory, memorypos_embed),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                memorypos_embed: Optional[Tensor] = None,
                contextpos_embed: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, 
                           querypos_embed = querypos_embed, 
                           memorypos_embed = memorypos_embed,
                           contextpos_embed=contextpos_embed)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, 
                           querypos_embed = querypos_embed, 
                           memorypos_embed = memorypos_embed,
                           contextpos_embed=contextpos_embed)


