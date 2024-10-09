from collections import OrderedDict
from typing import Tuple, Union
from itertools import repeat
import collections.abc
import sys
sys.path.append('/data10T/wangbingbing/Chinese-CLIP/cn_clip/training')
from params import is_DDIM, is_att, is_matrix#, num_class
import math
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

import importlib.util
if importlib.util.find_spec('flash_attn'):
    FlashMHA = importlib.import_module('flash_attn.flash_attention').FlashMHA

from cn_clip.clip import _tokenizer
from cn_clip.clip.configuration_bert import BertConfig
from cn_clip.clip.modeling_bert import BertModel

import torch.nn.functional as F
import matplotlib.pyplot as plt

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

def Cross_Attention(input_size, vector1,vector2):
    # 初始化权重参数
    q = nn.Linear(input_size, input_size).cuda()
    k = nn.Linear(input_size, input_size).cuda()
    v = nn.Linear(input_size, input_size).cuda()

    query = q(vector1)
    key = k(vector2)
    value = v(vector2)

    # 计算注意力分数
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    attention_scores = F.softmax(attention_scores, dim=-1)

    # 利用注意力分数对value进行加权求和
    attended_vector = torch.matmul(attention_scores, value)
    # plt.imshow(attention_scores.detach().numpy(), cmap='hot', interpolation='nearest')
    # plt.xlabel('Text Position')
    # plt.ylabel('Image Position')
    # plt.title('Cross-Attention Weights')
    # plt.colorbar()
    # plt.savefig('1.jpg')
    # exit()
    return attended_vector

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, use_flash_attention: bool = False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head) if not use_flash_attention else FlashMHA(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.use_flash_attention = use_flash_attention

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if self.use_flash_attention:
            # Batch first is needed for FlashAttention. See https://github.com/HazyResearch/flash-attention/issues/84 for more information.
            return self.attn(x.transpose(1, 0))[0].transpose(1, 0)
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, use_flash_attention: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, use_flash_attention) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for r in self.resblocks:
                x = checkpoint(r, x)
            return x        
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, use_flash_attention: bool = False):
        super().__init__()
        self.input_resolution = input_resolution
        self.grid_size = (self.input_resolution // patch_size, self.input_resolution // patch_size)
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, use_flash_attention=use_flash_attention)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int((L - 1) * (1 - mask_ratio))

        noise = torch.rand(N, L - 1, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1) + torch.ones(N, L - 1, device=x.device,
                                                               dtype=int)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        x0 = x[:, 0, :]
        x0 = x0.reshape(N, 1, D)
        x_masked_add = torch.cat([x0, x_masked], axis=1)
        return x_masked_add

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        if mask_ratio != 0:
            x = self.random_masking(x, mask_ratio)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

def LearnableRelationMatrix(input_dim, representation1, representation2):
    relation_matrix = nn.Parameter(torch.randn(input_dim, input_dim)).cuda()

    # 将两个表示通过关系矩阵进行映射
    mapped_representation1 = torch.matmul(representation1, relation_matrix)
    mapped_representation2 = torch.matmul(representation2, relation_matrix)

    # 可以根据具体需求进一步操作，例如计算相似性得分等

    return mapped_representation1, mapped_representation2

class SemanticAttn(nn.Module):
    def __init__(self):
        super(SemanticAttn, self).__init__()
        self.emb = 512

    def forward(self, x, k_ges,k_pos,k_face,k_ver):
        x = Cross_Attention(self.emb, x, k_ges)
        x = Cross_Attention(self.emb, x, k_pos)
        x = Cross_Attention(self.emb, x, k_face)
        x = Cross_Attention(self.emb, x, k_ver)
        return x

# 定义简单的分类器模型
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes).cuda()

    def forward(self, x):
        # print(x.size())
        # exit()
        intent_logits  = self.fc(x)
        # intent_logits = self.classifier(text_representation)
        intent_probs = nn.functional.softmax(intent_logits, dim=1)
        predicted_labels = torch.argmax(intent_probs, dim=1)
        return intent_probs, predicted_labels


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 vocab_size: int,
                 text_attention_probs_dropout_prob: float, 
                 text_hidden_act: str, 
                 text_hidden_dropout_prob: float, 
                 text_hidden_size: int,
                 text_initializer_range: float, 
                 text_intermediate_size: int, 
                 text_max_position_embeddings: int, 
                 text_num_attention_heads: int, 
                 text_num_hidden_layers: int, 
                 text_type_vocab_size: int,
                 tokenizer = _tokenizer,
                 # vision head width, added this param for ViT-H
                 vision_head_width: int = 64,
                 use_flash_attention: bool = False,
                 ):
        super().__init__()

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // vision_head_width
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // vision_head_width
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                use_flash_attention=use_flash_attention
            )

        self.bert_config = BertConfig(
            vocab_size_or_config_json_file=vocab_size,
            hidden_size=text_hidden_size,
            num_hidden_layers=text_num_hidden_layers,
            num_attention_heads=text_num_attention_heads,
            intermediate_size=text_intermediate_size,
            hidden_act=text_hidden_act,
            hidden_dropout_prob=text_hidden_dropout_prob,
            attention_probs_dropout_prob=text_attention_probs_dropout_prob,
            max_position_embeddings=text_max_position_embeddings,
            type_vocab_size=text_type_vocab_size,
            initializer_range=text_initializer_range,
            layer_norm_eps=1e-12,
            use_flash_attention=use_flash_attention
        )
        self.bert = BertModel(self.bert_config)
        self.embed_dim = embed_dim
        self.text_projection = nn.Parameter(torch.empty(text_hidden_size, embed_dim))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.tokenizer = tokenizer

        self.initialize_parameters()

        # 给每个speaker设置一个style存储器
        self.memory_dict = {}
        self.speaker_list = []
        # self.relation_matrix_model = LearnableRelationMatrix(embed_dim)
        # self.cross_attention = Cross_Attention(embed_dim)

        # self.transformer = nn.Transformer(
        #     d_model=hidden_dim,
        #     nhead=nhead,
        #     num_encoder_layers=num_layers,
        #     num_decoder_layers=num_layers
        # )
        # self.classifier = nn.Linear(embed_dim, num_class)

    def initialize_parameters(self):
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.bert_config.hidden_size ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.bert.set_grad_checkpointing(enable)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, mask_ratio=0):
        if isinstance(self.visual, ModifiedResNet):
            # mask_ratio > 0 (FLIP strategy) is currently only implemented for VisualTransformer.
            return self.visual(image.type(self.dtype))
        return self.visual(image.type(self.dtype), mask_ratio)

    def encode_text(self, text, sent_text=None):
        pad_index = self.tokenizer.vocab['[PAD]']
        attn_mask = text.ne(pad_index).type(self.dtype)
        x = self.bert(text, attention_mask=attn_mask,sent_ids = sent_text)[0].type(self.dtype) # [batch_size, seq_length, hidden_size]
        return x[:, 0, :] @ self.text_projection

    def style_cosine(self, image, style_image):
        # 将特征张量展平为一维向量
        feature1_flat = image.view(-1)
        feature2_flat = style_image.view(-1)

        # 计算 cosine similarity
        cosine_sim = F.cosine_similarity(feature1_flat, feature2_flat, dim=0)

        return (0.5 * (cosine_sim + 1))

    def forward(self, Flag, image,text,sent_text,keywords,speaker,number, mask_ratio=0,comet=None,
                k_gesture=None,k_posture=None,k_facial=None,k_verbal=None,
                imgid2image =None,imgid2kges=None,imgid2kpos=None,imgid2kface=None,imgid2kver=None,
                intent2token=None,id2intent=None):
        # print('speaker:',speaker)
        # exit()
        # classifier  = nn.Linear(embed_dim, num_class)
        style_loss = 0.0
        for i in range(len(speaker)):
            if speaker[i] not in self.speaker_list:
                self.memory_dict[speaker[i]] = {'real': 0.25, 'animal': 0.25, 'cartoon': 0.25, 'people': 0.25}
                self.speaker_list.append(speaker[i])
                # print('speaker_list:', self.speaker_list)

            # real_cos = self.style_cosine(image[i], real_image[i])
            # cartoon_cos = self.style_cosine(image[i], cartoon_image[i])
            # people_cos = self.style_cosine(image[i], people_image[i])
            # animal_cos = self.style_cosine(image[i], animal_image[i])

            # w_real = self.memory_dict[speaker[i]]['real']
            # w_cartoon = self.memory_dict[speaker[i]]['cartoon']
            # w_people = self.memory_dict[speaker[i]]['people']
            # w_animal = self.memory_dict[speaker[i]]['animal']

            # print('w_real:', w_real,real_cos)

            # update
            # cos = real_cos * w_real + cartoon_cos * w_cartoon + \
            #       people_cos * w_people + animal_cos * w_animal
            #
            # update_real = real_cos * w_real
            # update_cartoon = cartoon_cos * w_cartoon
            # update_people = people_cos * w_people
            # update_animal = animal_cos * w_animal

            # 计算更新的和
            # total_update = update_real + update_cartoon + update_people + update_animal

            # 对每个更新进行归一化
            # update_real = update_real / total_update
            # update_cartoon = update_cartoon / total_update
            # update_people = update_people / total_update
            # update_animal = update_animal / total_update

            # self.memory_dict[speaker[i]]['real'] = update_real
            # self.memory_dict[speaker[i]]['cartoon'] = update_cartoon
            # self.memory_dict[speaker[i]]['people'] = update_people
            # self.memory_dict[speaker[i]]['animal'] = update_animal
            #
            # style_loss+=(1-cos)
            #
            # if cos > 1:
            #     print('cos:',cos)
            #     exit()

        assert image is not None or text is not None, "text and image cannot both be None!"

        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)

        # print(keywords)
        # keywords = keywords.tolist()

        # print(text)
        # k_gesture,k_posture,k_facial,k_verbal = [],[],[],[]
        # print(k_gesture,k_posture,k_facial,k_verbal)
        # exit()
        # keyword_features = self.encode_text(keywords, sent_text)

        # all_image_features = {}
        # all_k_ges,all_k_pos,all_k_face,all_k_ver = {},{},{},{}
        # Flag = True
        if Flag:
            all_attn_features = {}
            all_intent_features = {}
            for k, v in id2intent.items():
                # print('k v:',k, v)
                intent_token = intent2token[v].unsqueeze(0)
                # print(torch.equal(intent_token[0],intent_token[1]))
                # print(intent_token.size())
                # exit()
                intent_token = self.encode_text(intent_token)
                intent_token = intent_token / intent_token.norm(dim=-1, keepdim=True)
                # print(intent_token.size())
                all_intent_features[k] = intent_token

            # print(len(imgid2image))
            # exit()
            num= 0
            for k, v in imgid2image.items():
                num+=1
                # print(num,k,v[0].unsqueeze(0).size())

                img = self.encode_image(v.unsqueeze(0))  # 3,224,224
                img = img / img.norm(dim=-1, keepdim=True)

                if 'animal'  in k or 'real' in k or 'cartoon' in k or 'people' in k:
                    continue
                k_ges = imgid2kges[k].unsqueeze(0)
                k_ges = self.encode_text(k_ges)
                k_ges = k_ges / k_ges.norm(dim=-1, keepdim=True)

                k_pos = imgid2kpos[k].unsqueeze(0)
                k_pos = self.encode_text(k_pos)
                k_pos = k_pos / k_pos.norm(dim=-1, keepdim=True)

                k_face = imgid2kface[k].unsqueeze(0)
                k_face = self.encode_text(k_face)
                k_face = k_face / k_face.norm(dim=-1, keepdim=True)

                k_ver = imgid2kver[k].unsqueeze(0)
                k_ver = self.encode_text(k_ver)
                k_ver = k_ver / k_ver.norm(dim=-1, keepdim=True)

                img = Cross_Attention(512, img, k_ges)
                img = Cross_Attention(512, img, k_pos)
                img = Cross_Attention(512, img, k_face)
                img = Cross_Attention(512, img, k_ver)

                # print(k,v)
                # print(torch.equal(v[0],v[1]))
                # print(temp.size())
                # exit()
                # all_image_features[k] = temp
                all_attn_features[k] = img

        '''
        for k,v in imgid2kges.items():
            temp = self.encode_text(v[0].unsqueeze(0))
            # print(temp.size())
            temp = temp / temp.norm(dim=-1, keepdim=True)
            all_k_ges[k] = temp
        for k,v in imgid2kpos.items():
            temp = self.encode_text(v[0].unsqueeze(0))
            temp = temp / temp.norm(dim=-1, keepdim=True)
            all_k_pos[k] = temp
        for k,v in imgid2kface.items():
            temp = self.encode_text(v[0].unsqueeze(0))
            temp = temp / temp.norm(dim=-1, keepdim=True)
            all_k_face[k] = temp
        for k,v in imgid2kver.items():
            temp = self.encode_text(v[0].unsqueeze(0))
            temp = temp / temp.norm(dim=-1, keepdim=True)
            all_k_ver[k] = temp
        '''
        image_features = self.encode_image(image, mask_ratio)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.encode_text(text, sent_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        '''
        k_gesture_features = self.encode_text(k_gesture,None)
        k_gesture_features = k_gesture_features / k_gesture_features.norm(dim=-1, keepdim=True)
        k_posture_features = self.encode_text(k_posture, None)
        k_posture_features = k_posture_features / k_posture_features.norm(dim=-1, keepdim=True)
        k_facial_features = self.encode_text(k_facial, None)
        k_facial_features = k_facial_features / k_facial_features.norm(dim=-1, keepdim=True)
        k_verbal_features = self.encode_text(k_verbal, None)
        k_verbal_features = k_verbal_features / k_verbal_features.norm(dim=-1, keepdim=True)
        '''
        # text_features_list = []
        # n = 1
        # print(type(number),number.tolist())
        # print('text:',text_features.size())

        # intent_aware_representation = Cross_Attention(self.embed_dim,keyword_features,text_features)
        # print(intent_aware_representation.size())
        # text_representation, image_representation = text_features, image_features
        if is_att:
            text_features = Cross_Attention(self.embed_dim,keyword_features,text_features)
        text_representation, image_representation = text_features, image_features
        # text_representation = text_features
        if is_matrix:
            text_representation, image_representation = LearnableRelationMatrix(self.embed_dim,text_features, image_features)

        # print(text_representation.size())
        # self.classifier(text_representation)
        # intent_logits = self.classifier(text_representation)
        # intent_probs = nn.functional.softmax(intent_logits, dim=1)
        # predicted_labels = torch.argmax(intent_probs, dim=1)
        # exit()
        # 输出的形状是 (batch_size, input_dim)
        # print(mapped_representation1.shape)
        # print(mapped_representation2.shape)        # B = number.size()
        #
        # for j in range(B):
        #     batch_feature_list = []
        #     #   每个batch进行处理
        #     dialogue = text[j]
        #     n = 1
        #     for each_context in dialogue:
        #         if n>number[j]:
        #             break
        #         batch_feature = self.encode_text(each_context)
        #         batch_feature = batch_feature / batch_feature.norm(dim=-1, keepdim=True)
        #
        #         batch_feature_list.append(batch_feature)
        #
        #     concatenated_features = torch.cat(batch_features_list, dim=0)
        #     concatenated_features = concatenated_features.unsqueeze(1).transpose(0, 1)
        #     # 这里应该还有个映射
        #     print('con:',concatenated_features.size())
        #     exit()
        #     # Run the concatenated features through the transformer model
        #     final_representation = self.transformer_model(concatenated_features)

        # for each in text:
        #     if n > int(number):
        #         break
        #     n+=1
        #     print(each.size())
        #     exit()
        #     text_feature = self.encode_text(each)
        #     text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        #     text_features_list.append(text_feature)


        # print(image_features.size(),keyword_features.size(),image_features.size())
        # exit()
        style_loss = 0
        if Flag:
            return image_representation, text_representation, self.logit_scale.exp(), style_loss, \
                   all_attn_features, all_intent_features  # k_gesture_features,k_posture_features,k_facial_features,k_verbal_features,\
        else:
            return image_representation, text_representation, self.logit_scale.exp(), style_loss
            #all_image_features, all_k_ges,all_k_pos,all_k_face,all_k_ver#, intent_probs, predicted_labels#, memory_dict, speaker_list

    def get_similarity(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        if isinstance(l, BertModel):
            l.to(torch.half)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def restore_model(model, clip_state_dict: dict, bert_state_dict: dict, use_flash_attention: bool):
    merged_state_dict = {}

    # use clip_state_dict to initialize the image encoder & logit scale
    if clip_state_dict is not None:
        for k, v in clip_state_dict.items():
            if k.startswith("visual") or k == "logit_scale":
                merged_state_dict[k] = v

    # use bert_state_dict to initialize the text encoder
    if bert_state_dict is not None:
        for k, v in bert_state_dict.items():
            if k.startswith("bert") and "bert.pooler" not in k:
                merged_state_dict[k] = v

    # adapt flash attention
    if use_flash_attention:
        merged_state_dict = convert_state_dict(merged_state_dict)

    convert_weights(model)
    resize_pos_embed(merged_state_dict, model)
    model.load_state_dict(merged_state_dict, strict=False)
    return model.eval()


def convert_state_dict(state_dict):
    """Adapt to Flash Attention"""
    if not state_dict:
        return state_dict

    prefix = 'module.' if list(state_dict.keys())[0].startswith('module') else ''

    if f'{prefix}visual.transformer.resblocks.0.attn.in_proj_weight' in state_dict:
        for k in list(state_dict.keys()):
            if 'attn.in_proj_weight' in k:
                state_dict[k.replace('attn.in_proj_weight', 'attn.Wqkv.weight')] = state_dict.pop(k)
            elif 'attn.in_proj_bias' in k:
                state_dict[k.replace('attn.in_proj_bias', 'attn.Wqkv.bias')] = state_dict.pop(k)
    elif f'{prefix}visual.transformer.resblocks.0.attn.Wqkv.weight' in state_dict:
        for k in list(state_dict.keys()):
            if 'attn.Wqkv.weight' in k:
                state_dict[k.replace('attn.Wqkv.weight', 'attn.in_proj_weight')] = state_dict.pop(k)
            elif 'attn.Wqkv.bias' in k:
                state_dict[k.replace('attn.Wqkv.bias', 'attn.in_proj_bias')] = state_dict.pop(k)

    if f'{prefix}bert.encoder.layer.0.attention.self.query.weight' in state_dict:
        i = 0
        while f'{prefix}bert.encoder.layer.{i}.attention.self.query.weight' in state_dict:
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.weight'] = torch.cat(
                (state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.query.weight'),
                 state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.key.weight'),
                 state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.value.weight'))
            )
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.bias'] = torch.cat(
                (state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.query.bias'),
                 state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.key.bias'),
                 state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.value.bias'))
            )
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.out_proj.weight'] = \
                state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.output.dense.weight')
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.out_proj.bias'] = \
                state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.output.dense.bias')
            i += 1
    elif f'{prefix}bert.encoder.layer.0.attention.self.Wqkv.weight' in state_dict:
        i = 0
        while f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.weight' in state_dict:
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.query.weight'], \
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.key.weight'], \
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.value.weight'] = \
                torch.chunk(state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.weight'), chunks=3)
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.query.bias'], \
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.key.bias'], \
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.value.bias'] = \
                torch.chunk(state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.bias'), chunks=3)
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.output.dense.weight'] = \
                state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.out_proj.weight')
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.output.dense.bias'] = \
                state_dict.pop(f'module.bert.encoder.layer.{i}.attention.self.out_proj.bias')
            i += 1

    return state_dict


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1, prefix=""):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get(prefix + 'visual.positional_embedding', None)
    model = model.module if hasattr(model, 'module') else model
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict[prefix + 'visual.positional_embedding'] = new_pos_embed


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)
