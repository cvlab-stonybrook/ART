import torch
import torch.nn.functional as F
from torch import nn, Tensor

from vgcore import VGCore
from decoder import TransformerDecoderWrapper

class RefGaze(nn.Module):
    def __init__(self, args, pretrained_vgcore=False, vgcore_checkpoint=None):
        super().__init__()
        self.vgcore = VGCore(args=args)
        self.hidden_dim = args.hidden_dim
        if pretrained_vgcore:
            checkpoint = torch.load(vgcore_checkpoint)
            self.vgcore.load_state_dict(checkpoint['model'])
            del checkpoint
        self.decoder = TransformerDecoderWrapper(activation="relu", args=args)

        self.token_predictor = nn.Linear(self.hidden_dim, 3)
        self.generator_y_mu = nn.Linear(args.hidden_dim, 1)
        self.generator_x_mu = nn.Linear(args.hidden_dim, 1)
        # self.generator_t_mu = nn.Linear(args.hidden_dim, 1)
        self.generator_y_logvar = nn.Linear(args.hidden_dim, 1)
        self.generator_x_logvar = nn.Linear(args.hidden_dim, 1)
        # self.generator_t_logvar = nn.Linear(args.hidden_dim, 1)

        self.dropout = nn.Dropout(args.predictor_dropout)
        self.activation = F.relu
        self.softmax = nn.LogSoftmax(dim=-1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return (mu + eps*std)

    def forward(self, img_data: Tensor, text_data: Tensor, context: Tensor, context_padding_mask: Tensor):
        pred_box, nextword, target, src, memory_key_padding_mask = self.vgcore(img_data=img_data, text_data=text_data)
        outs = self.decoder(memory=src, context=context, tgt_key_padding_mask=context_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        outs = self.dropout(outs)
        
        #get Gaussian parameters for (x,y)
        y_mu, y_logvar, x_mu, x_logvar = self.generator_y_mu(outs),self.generator_y_logvar(outs), self.generator_x_mu(outs), self.generator_x_logvar(outs)

        return pred_box, nextword, target, self.softmax(self.token_predictor(outs)).permute(1,0,2), self.activation(self.reparameterize(x_mu, x_logvar)).permute(1,0,2),\
            self.activation(self.reparameterize(y_mu, y_logvar)).permute(1,0,2)





    '''batch_indices = torch.arange(1)
    indices_to_set = torch.randint(0, output.size(1), (output.size(0),))
    mask = torch.zeros(output.size(0), 1)
    mask[batch_indices, indices_to_set] = 1.
    return mask'''

