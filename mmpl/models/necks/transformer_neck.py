import torch
import torch.nn as nn

from mmpl.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class TransformerEncoderNeck(BaseModule):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, model_dim, with_cls_token=True, num_encoder_layers=3):
        super(TransformerEncoderNeck, self).__init__()
        self.embed_dims = model_dim
        self.with_cls_token = with_cls_token
        if self.with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        encoder_layer = nn.TransformerEncoderLayer(
            self.embed_dims, nhead=8, dim_feedforward=self.embed_dims*2,
            dropout=0.1, batch_first=True
        )
        encoder_norm = nn.LayerNorm(self.embed_dims)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B = x.shape[0]
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer_encoder(x)
        if self.with_cls_token:
            return x[:, 0], x
        return None, x
