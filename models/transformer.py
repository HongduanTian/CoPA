'''
This file provide codes of Transformer model.
'''
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils import spectral_norm
from config import args


class AttentionModule(nn.Module):
    '''
    A single attention module with queries, keys and values head.
    '''
    def __init__(self, num_heads:int, in_channels: int=512, out_channels: int=512, 
                 is_bias: bool=True, init_mode:str='random') -> None:
        super(AttentionModule, self).__init__()
        """
        A single attention head with one or several heads.
        Args:
            num_head: int. Number of heads.
            in_channels: int. Number of input channels, default 512.
            out_channels: int. Number of output channels, default 512.
            is_bias: bool. Whether add bias in linear layers.
            init_mode: str. Initialization mode. Choose from ['random', 'eye']
        """
        # constants
        self.num_heads = num_heads
        self.init_mode = init_mode
        self.out_dim = out_channels

        # learnable modules: query head, key head
        self.q_head = nn.Linear(in_features=in_channels, out_features=out_channels, bias=is_bias)
        self.k_head = nn.Linear(in_features=in_channels, out_features=out_channels, bias=is_bias)
        self.v_head = nn.Linear(in_features=in_channels, out_features=out_channels, bias=is_bias)

        # initialize the parameters
        self._init_params()
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x: torch.tensor. input data. Required size format: [B (number of classes), S (sequence length: H x W), D (dimension)].
        Return:
            atten_feat: torch.tensor. Attention features with the size of [B, S, D].
        """
        def split_last(X: torch.tensor, shape: tuple) -> torch.tensor:
            """
            Split given features into multi-head.
            Args:
                X: torch.tensor. Given features obtained from q/k/v heads.
                shape: tuple. Expected shape.
            Return:
                reshaped X: torch.tensor. Shape: [B, S, num_heads, D / num_heads]
            """
            shape = list(shape)
            assert shape.count(-1) <= 1     # ONLY one -1 is included in the shape tuple.
            if -1 in shape:
                shape[shape.index(-1)] = int(X.size(-1) / -np.prod(shape))  # Dim / num_heads
            return X.view(*X.size()[:-1], *shape)   # Shape: [B, S, num_head, Dim/num_head]


        def merge_last(X: torch.tensor, num_dims: int) -> torch.tensor:
            """
            Merge given dimensions.
            Args:
                X: torch.tensor. Attention features obtained from attetion operations.
                num_dims: int. Number of dimensions to be merged.
            Return:
                Reshaped X: torch.tensor. Shape: [B, S, D].
            """
            size = X.size()
            assert num_dims > 1 and num_dims < len(size)
            return X.view(*size[:-num_dims], -1)


        def softmax_func(score_mat: torch.tensor) -> torch.tensor:
            """
            Calculate softmax score.
            Args:
                score_mat: torch.tensor. Features of queries, keys and values with the shape of [B, num_heads, S, S].
            Return:
                softmax_mat: torch.tensor with shape of [B, num_heads, S, S].
            """
            max_vals, _ = torch.max(score_mat, dim=-1, keepdim=True)
            score_mat -= max_vals
            exp_logits = torch.exp(score_mat)
            softmax_mat = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
            return softmax_mat


        # Project the feature embeddings.
        queries, keys, values = self.q_head(x), self.k_head(x), self.v_head(x) # All queries, keys and values own the shape [B, S, D]
        queries, keys, values = (split_last(x, (self.num_heads, -1)).transpose(1, 2) for x in [queries, keys, values])  # reshape q, k, v to [B, num_heads, S, new_dim], new_dim = D / num_heads.

        # attention calculation
        sim_score = softmax_func(torch.matmul(queries, keys.transpose(-2, -1)) * torch.rsqrt(torch.tensor(keys.size(-1)).type_as(keys)).to(keys.device))
        atten_feat = torch.matmul(sim_score, values).transpose(1, 2).contiguous()   # size: [B, num_heads, S, new_dim] -> [B, S, num_heads, new_dim]
        atten_feat = merge_last(atten_feat, 2)

        return atten_feat

    @torch.no_grad()
    def _init_params(self) -> None:
        "Initialize the modules in this attention module."
        if self.init_mode == 'random':
            nn.init.xavier_normal_(self.q_head.weight)
            if hasattr(self.q_head, 'bias') and self.q_head.bias is not None:
                nn.init.normal_(self.q_head.bias)
        
            nn.init.xavier_normal_(self.k_head.weight)
            if hasattr(self.k_head, 'bias') and self.k_head.bias is not None:
                nn.init.normal_(self.k_head.bias)
        
            nn.init.xavier_normal_(self.v_head.weight)
            if hasattr(self.v_head, 'bias') and self.v_head.bias is not None:
                nn.init.normal_(self.v_head.bias)
        
        elif self.init_mode == 'eye':
            self.q_head.weight = nn.Parameter(torch.eye(self.out_dim, self.out_dim))
            self.q_head.bias = nn.Parameter(torch.zeros_like(self.q_head.bias))

            self.k_head.weight = nn.Parameter(torch.eye(self.out_dim, self.out_dim))
            self.k_head.bias = nn.Parameter(torch.zeros_like(self.k_head.bias))

            self.v_head.weight = nn.Parameter(torch.eye(self.out_dim, self.out_dim))
            self.v_head.bias = nn.Parameter(torch.zeros_like(self.v_head.bias))
        
        else:
            raise ValueError("Unrecognized initialization mode. Please choose from ['random', 'eye].")
        
    @torch.no_grad()
    def reset_params(self) -> None:
        self._init_params()


class SelfAttentionBlock(nn.Module):
    "An typical attention block."
    def __init__(self, num_heads:int=8, in_channels:int=512, out_channels:int=512, 
                 is_bias:bool=True, init_mode:str='random'):
        super(SelfAttentionBlock, self).__init__()
        self.init_mode = init_mode
        self.out_dim = out_channels
        self.ff_dim = out_channels * 2

        self.atten_module = AttentionModule(num_heads=num_heads, in_channels=in_channels, out_channels=out_channels, 
                                            is_bias=is_bias, init_mode=init_mode)
        self.layer_norm1 = nn.LayerNorm(out_channels, eps=1e-6)

        self.ff_linear1 = nn.Linear(in_features=out_channels, out_features=self.ff_dim)
        self.ff_linear2 = nn.Linear(in_features=self.ff_dim, out_features=out_channels)
        self.layer_norm2 = nn.LayerNorm(out_channels, eps=1e-6)

        self._init_params()
    
    def forward(self, x:torch.tensor) -> torch.tensor:
        "Return x with shape [B, S, D]"
        #h = self.projection(self.atten_module(self.layer_norm1(x)))
        h = self.atten_module(self.layer_norm1(x))
        x = x + h
        h = self.ff_linear2(F.gelu(self.ff_linear1(self.layer_norm2(x))))
        x = x + h
        return x
    
    @torch.no_grad()
    def _init_params(self) -> None:
        "Initialize parameters"
        if self.init_mode == 'random':   
            nn.init.xavier_normal_(self.ff_linear1.weight)
            if hasattr(self.ff_linear1, 'bias') and self.ff_linear1.bias is not None:
                nn.init.normal_(self.ff_linear1.bias, std=1e-6)
            
            nn.init.xavier_normal_(self.ff_linear2.weight)
            if hasattr(self.ff_linear2, 'bias') and self.ff_linear2.bias is not None:
                nn.init.normal_(self.ff_linear2.bias, std=1e-6)
        
        elif self.init_mode == 'eye':
            self.ff_linear1.weight = nn.Parameter(torch.ones_like(self.ff_linear1.weight))
            self.ff_linear1.bias = nn.Parameter(torch.zeros_like(self.ff_linear1.bias))
            self.ff_linear2.weight = nn.Parameter(torch.ones_like(self.ff_linear2.weight))
            self.ff_linear2.bias = nn.Parameter(torch.zeros_like(self.ff_linear2.bias))
        
        else:
            raise ValueError("Unrecognized initialization mode. Please choose from ['random', 'eye].")
        
        self.layer_norm1.weight = nn.Parameter(torch.ones_like(self.layer_norm1.weight))
        self.layer_norm1.bias = nn.Parameter(torch.zeros_like(self.layer_norm1.bias))

        self.layer_norm2.weight = nn.Parameter(torch.ones_like(self.layer_norm2.weight))
        self.layer_norm2.bias = nn.Parameter(torch.zeros_like(self.layer_norm2.bias))
    
    @torch.no_grad()
    def reset_params(self) -> None:
        self.atten_module.reset_params()
        self._init_params()



class SimpleSelfAttentionBlock(nn.Module):
    """A simple self-attention module with only attention and a linear layer."""
    def __init__(self, num_heads:int=8, in_channels:int=512, out_channels:int=512, 
                 is_bias:bool=True, init_mode:str='eye'):
        super(SimpleSelfAttentionBlock, self).__init__()
        self.out_dim = out_channels
        self.init_mode = init_mode

        self.atten_module = AttentionModule(num_heads=num_heads, in_channels=in_channels, out_channels=out_channels, 
                                            is_bias=is_bias, init_mode=init_mode)
        self.layernorm = nn.LayerNorm(out_channels, eps=1e-6)
        self.linear_head = nn.Linear(in_features=out_channels, out_features=out_channels)
    
    def forward(self, x:torch.tensor):
        return self.linear_head(self.layernorm(x+self.atten_module(x)))
    
    @torch.no_grad()
    def _init_params(self) -> None:
        if self.init_mode == 'eye':
            self.linear_head.weight = nn.Parameter(torch.eye(self.out_dim, self.out_dim))
            self.linear_head.bias = nn.Parameter(torch.zeros_like(self.linear_head.bias.data))

        elif self.init_mode == 'random':
            nn.init.xavier_normal_(self.linear_head.weight)
            if hasattr(self.linear_head, 'bias') and self.linear_head.bias is not None:
                nn.init.normal_(self.linear_head.bias, std=1e-6)
        
        else:
            raise ValueError("Unrecognized initialization mode. Please choose from ['random', 'eye].")
        
        self.atten_module.reset_params()
        self.layernorm.weight = nn.Parameter(torch.ones_like(self.layernorm.weight))
        self.layernorm.bias = nn.Parameter(torch.zeros_like(self.layernorm.bias))
    
    @torch.no_grad()
    def reset_params(self) -> None:
        self._init_params()
        

class Transformer(nn.Module):
    '''
    A Transformer model with several heads.
    '''
    def __init__(self, num_blocks:int, num_heads:int=8, num_dim:int=512, 
                 is_bias:bool=True, is_simpleatten:bool=False, init_mode:str='random') -> None:
        super(Transformer, self).__init__()
        """
        Transformer model.
        Args:
            num_blocks: int. Number of attention blocks used in a Transformer model.
            num_heads: int. Number of heads in an attention block.
            num_dim: int. The dimension of output feature maps. The 'num_dim' is same among all heads in a Transformer model.
            ff_dim: int. The intermediate dimension of the projection block after the attention head in an attention block.
            is_bias: bool. Whether use bias in a single layer.
            dropout: float. The probability used in a Dropout layer.
            feat_mode: str. The image size and the position of the corresponding blocks in a ResNet model. 
                        The features are used for generating prototypes.
        """
        self.init_mode = init_mode
        self.out_dim = num_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_dim))   # class token parameters
        self.pos_embeddings = nn.Parameter(torch.zeros(1, 37, num_dim)) # position embeddings
        self._init_params()
        if is_simpleatten:
            self.blocks = nn.ModuleList([
                SimpleSelfAttentionBlock(num_heads=num_heads, in_channels=num_dim, out_channels=num_dim, is_bias=is_bias, init_mode=init_mode) for _ in range(num_blocks)
            ])
        else:
            self.blocks = nn.ModuleList([
                SelfAttentionBlock(num_heads=num_heads, in_channels=num_dim, out_channels=num_dim, is_bias=is_bias, init_mode=init_mode) for _ in range(num_blocks)
            ])


    def forward(self, protos: torch.tensor) -> torch.tensor:
        """
        Args:
            protos: torch.tensor with shape [num_cls, c, h, w]. 
                    Initial prototypes of input data in a given task. Calculated by averaging all samples in each class.
        Return:
            torch.tensor with shape [num_cls, S, D]. The prototypes of each class.
        """
        protos = torch.transpose(torch.flatten(protos, start_dim=2), 1, 2) # shape [num_cls, 512, w, h] -> [num_cls, 512, wh] -> [num_cls, wh, 512]

        protos = torch.cat((self.cls_token.expand(protos.size(0), -1, -1), protos), dim=1)
        protos += self.pos_embeddings

        for block in self.blocks:
            protos = block(protos)
        protos = protos[:, 0, :]
        return protos
    
    def get_state_dict(self):
        return self.state_dict()
    
    @torch.no_grad()
    def _init_params(self) -> None:
        nn.init.constant_(self.cls_token, 0)
        nn.init.normal_(self.pos_embeddings, std=0.02)
        
    @torch.no_grad()
    def reset_params(self) -> None:
        self._init_params()

        for block in self.blocks:
            block.reset_params()


if __name__ == '__main__':
    seed_id = 42
    torch.manual_seed(seed_id)
    torch.cuda.manual_seed_all(seed_id)
    np.random.seed(seed_id)
    random.seed(seed_id)
    torch.backends.cudnn.deterministic=True

    model = Transformer(num_blocks=1)
    for params in model.parameters():
        print(params)
    print("++++++++++++++++++++++++++++++++++++++++++++++")
    model.reset_params()
    for reset_params in model.parameters():
        print(reset_params)
        


