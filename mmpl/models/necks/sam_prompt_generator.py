import math

import torch
import torch.nn as nn
from einops import rearrange

from mmpl.registry import MODELS


@MODELS.register_module()
class SAMPromptGenNeck(nn.Module):
    def __init__(
            self,
            prompt_shape=(100, 6),
            img_feat_channels=1280,
            out_put_channels=256,
            num_img_feat_level=4,
            img_feat_size=32,
            n_cls=2,
    ):
        super(SAMPromptGenNeck, self).__init__()
        self.prompt_shape = prompt_shape
        self.num_queries = prompt_shape[0]
        self.per_query_point = prompt_shape[1]
        self.img_feat_channels = img_feat_channels
        self.out_put_channels = out_put_channels
        self.num_img_feat_level = num_img_feat_level
        self.img_feat_size = img_feat_size
        self.n_cls = n_cls

        decoder_embed_dims = out_put_channels // num_img_feat_level
        self.decoder_input_projs = nn.ModuleList()
        # from low resolution to high resolution
        for _ in range(num_img_feat_level):
            self.decoder_input_projs.append(
                nn.Sequential(
                    nn.Conv2d(img_feat_channels, 2 * decoder_embed_dims, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(2 * decoder_embed_dims, 2 * decoder_embed_dims, kernel_size=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(2 * decoder_embed_dims, 2 * decoder_embed_dims, kernel_size=3, stride=2, padding=1)
                ))
        self.level_embed = nn.Embedding(self.num_img_feat_level, 2 * decoder_embed_dims)
        self.gather_img_feats = nn.Sequential(
            nn.Conv2d(out_put_channels * 2, out_put_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_put_channels, out_put_channels, 3, padding=1),
        )
        self.img_feats_pe = nn.Parameter(torch.zeros(1, out_put_channels, 32, 32))
        self.transformer = self.build_transformer()
        self.query_feat = nn.Embedding(self.num_queries, out_put_channels)
        self.cls_head = nn.Sequential(
            nn.Linear(out_put_channels, out_put_channels),
            nn.ReLU(),
            nn.Linear(out_put_channels, n_cls)
        )

        self.point_emb = nn.Sequential(
            nn.Linear(out_put_channels, out_put_channels),
            nn.ReLU(),
            nn.Linear(out_put_channels, out_put_channels),
            nn.ReLU(),
            nn.Linear(out_put_channels, self.per_query_point * out_put_channels)
        )

    def build_transformer(
            self, num_encoder_layers=2, num_decoder_layers=3, embed_dims=256, num_heads=8,
            mlp_ratio=2, dropout_rate=0.0, act_cfg=dict(type="gelu")):
        """Build transformer decoder."""
        transformer = nn.Transformer(
            d_model=embed_dims, nhead=num_heads, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=mlp_ratio * embed_dims,
            dropout=dropout_rate, activation=act_cfg['type'], batch_first=True, norm_first=True,
        )
        '''
        src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the Tensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the Tensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the Tensor mask for memory keys per batch (optional)
        '''
        return transformer

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs):
        inner_states = [x.permute(0, 3, 1, 2) for x in inputs]  # from low2high, all 4 layers
        bs = inner_states[0].shape[0]
        # inputs: list([B, C, H, W])
        num_layers = len(inputs)
        # import ipdb; ipdb.set_trace()
        # select the feature maps from the selected layers
        layer_start_id = num_layers - self.num_img_feat_level
        decoder_inputs = []
        for i in range(self.num_img_feat_level):
            decoder_input = self.decoder_input_projs[i](inner_states[i + layer_start_id])  # Bx256x64x64
            level_embed = self.level_embed.weight[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(bs, -1, -1, -1)
            decoder_input = decoder_input + level_embed
            decoder_inputs.append(decoder_input)
        decoder_inputs = torch.cat(decoder_inputs, dim=1)  # Bx256x64x64
        decoder_inputs = self.gather_img_feats(decoder_inputs)
        img_pe = self.img_feats_pe.expand(bs, -1, -1, -1)  # Bx256x64x64
        decoder_inputs = decoder_inputs + img_pe
        decoder_inputs = rearrange(decoder_inputs, 'b c h w -> b (h w) c')  # Bx4096x256
        query_feat = self.query_feat.weight.unsqueeze(0).expand(bs, -1, -1)  # Bx256x256
        decoder_outputs = self.transformer(
            src=decoder_inputs, tgt=query_feat
        )
        cls_logits = self.cls_head(decoder_outputs)
        point_embs = self.point_emb(decoder_outputs)
        point_embs = rearrange(point_embs, 'b n (t c) -> b n t c', t=self.per_query_point)  # Bx100x6x256

        return point_embs, cls_logits


@MODELS.register_module()
class SAMPromptConvNeck(nn.Module):
    def __init__(
            self,
            prompt_shape=(100, 5),
            img_feat_channels=1280,
            out_put_channels=256,
            num_img_feat_level=16,
            n_cls=2,
    ):
        super(SAMPromptConvNeck, self).__init__()
        self.prompt_shape = prompt_shape
        self.num_queries = prompt_shape[0]
        self.per_query_point = prompt_shape[1]
        self.point_size = int(math.sqrt(prompt_shape[0]))

        self.img_feat_channels = img_feat_channels
        self.out_put_channels = out_put_channels
        self.num_img_feat_level = num_img_feat_level
        self.n_cls = n_cls

        # decoder_embed_dims = img_feat_channels // 32
        decoder_embed_dims = 32
        self.decoder_input_projs = nn.ModuleList()
        # from low resolution to high resolution
        for _ in range(num_img_feat_level):
            self.decoder_input_projs.append(
                nn.Sequential(
                    nn.Conv2d(img_feat_channels, decoder_embed_dims, kernel_size=1),
                    # nn.BatchNorm2d(decoder_embed_dims),
                    nn.ReLU(),
                    nn.Conv2d(decoder_embed_dims, decoder_embed_dims, kernel_size=3, padding=1),
                    # nn.BatchNorm2d(decoder_embed_dims),
                    nn.ReLU(),
                ))
        self.level_embed = nn.Embedding(self.num_img_feat_level, decoder_embed_dims)
        self.gather_img_feats = nn.Sequential(
            nn.Conv2d(num_img_feat_level * decoder_embed_dims, out_put_channels, kernel_size=1),
            # nn.BatchNorm2d(out_put_channels),
            nn.ReLU(),
            nn.Conv2d(out_put_channels, out_put_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_put_channels, out_put_channels*2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_put_channels * 2, out_put_channels * 2, 3, padding=1),
        )

        self.img_feats_pe = nn.Parameter(torch.zeros(1, out_put_channels*2, self.point_size, self.point_size))

        self.cls_head = nn.Sequential(
            nn.Conv2d(out_put_channels * 2, out_put_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_put_channels, n_cls, 1)
        )

        self.point_emb = nn.Sequential(
            nn.Conv2d(out_put_channels * 2, out_put_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_put_channels, out_put_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_put_channels, self.per_query_point * out_put_channels, 1)
        )

    def forward(self, inputs):
        inner_states = [x.permute(0, 3, 1, 2) for x in inputs]  # from low2high, all 4 layers
        bs = inner_states[0].shape[0]
        # inputs: list([B, C, H, W])
        num_layers = len(inputs)
        # import ipdb; ipdb.set_trace()
        # select the feature maps from the selected layers
        layer_start_id = num_layers - self.num_img_feat_level
        decoder_inputs = []
        for i in range(self.num_img_feat_level):
            decoder_input = self.decoder_input_projs[i](inner_states[i + layer_start_id])  # Bx256x64x64
            level_embed = self.level_embed.weight[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(bs, -1, -1, -1)
            decoder_input = decoder_input + level_embed
            decoder_inputs.append(decoder_input)
        decoder_inputs = torch.cat(decoder_inputs, dim=1)  # Bx256x64x64
        decoder_inputs = self.gather_img_feats(decoder_inputs)
        # import pdb;
        # pdb.set_trace()
        decoder_inputs = torch.nn.functional.interpolate(decoder_inputs, size=(self.point_size, self.point_size), mode='bilinear', align_corners=True)
        img_pe = self.img_feats_pe.expand(bs, -1, -1, -1)  # Bx256x64x64
        decoder_inputs = decoder_inputs + img_pe

        cls_logits = self.cls_head(decoder_inputs)  # b c h w
        cls_logits = rearrange(cls_logits, 'b c h w -> b (h w) c')
        point_embs = self.point_emb(decoder_inputs)  # b c h w
        point_embs = rearrange(point_embs, 'b (t c) h w -> b (h w) t c', t=self.per_query_point)  # Bx100x6x256

        return point_embs, cls_logits
