'''
pa.py
Created by Wei-Hong Li [https://weihonglee.github.io]
This code allows you to attach pre-classifier alignment (PA) mapping to a pretrained backbone
and learn it on the support set to adapt features to a discriminative space.
'Universal Representation Learning from Multiple Domains for Few-shot Classification'
(https://arxiv.org/pdf/2103.13841.pdf)
'''
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.init as init

from models.model_utils import sigmoid, cosine_sim
from models.losses import prototype_loss, symmetric_CE, compute_prototypes
from utils import device
import torch.nn.functional as F

from config import args

def apply_selection(features, vartheta):
    """
    Performs pre-classifier alignment of features (feature adaptation) via a linear transformation.
    """

    #features = features.unsqueeze(-1).unsqueeze(-1)
    features = F.conv2d(features, vartheta[0]).flatten(1)

    return features


class MLP(nn.Module):
    def __init__(self, in_dim:int, out_dim:int) -> None:
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear1 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.out_dim, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=self.out_dim)
        self.linear2 = nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=1, stride=1, bias=False)
    
    def forward(self, x:torch.tensor) -> torch.tensor:
        '''
        Args:
            x: torch.tensor. With the shape [N, C, W, H].
        '''
        out = self.linear2(self.bn(self.linear1(x))).flatten(1)
        return out

    def _init_params(self) -> None:
        self.linear1.weight = nn.Parameter(torch.eye(self.out_dim, self.in_dim).unsqueeze(-1).unsqueeze(-1))
        self.linear2.weight = nn.Parameter(torch.eye(self.out_dim, self.in_dim).unsqueeze(-1).unsqueeze(-1))
        nn.init.constant_(self.bn.weight, 1.)
        nn.init.constant_(self.bn.bias, 0.)
    
    def reset_params(self) -> None:
        self._init_params()



def pa(context_features, context_labels, target_features, target_labels, 
       max_iter=40, ad_opt='linear', lr=0.1, distance='cos', prot_type='avg'):
    """
    PA method: learning a linear transformation per task to adapt the features to a discriminative space 
    on the support set during meta-testing
    """
    input_dim = context_features.size(1)
    output_dim = input_dim
    stdv = 1. / math.sqrt(input_dim)
    vartheta = []
    if ad_opt == 'linear':
        vartheta.append(torch.eye(output_dim, input_dim).unsqueeze(-1).unsqueeze(-1).to(device).requires_grad_(True))

    optimizer = torch.optim.Adadelta(vartheta, lr=lr)

    for i in range(max_iter):

        optimizer.zero_grad()
        selected_features = apply_selection(context_features, vartheta)
        loss, stat, _ = prototype_loss(selected_features, context_labels,
                                       selected_features, context_labels, distance=distance, prot_type=prot_type)

        loss.backward()
        optimizer.step()
    
        if i == max_iter - 1:
            with torch.no_grad():
                selected_context = apply_selection(context_features, vartheta)
                selected_target = apply_selection(target_features, vartheta)
                _, val_stat, _ = prototype_loss(
                    selected_context, context_labels,
                    selected_target, target_labels, distance=args['test.distance']
                )

    return vartheta, stat


def pa_mlp(context_features, context_labels, target_features, target_labels, max_iter=40, ad_opt='linear', lr=0.1, distance='cos'):
    
    input_dim = context_features.size(1)
    output_dim = input_dim

    # generate an MLP head
    mlp_head = MLP(input_dim, output_dim)

    optimizer = torch.optim.Adadelta(mlp_head.parameters(), lr=lr)

    mlp_head.reset_params()

    for i in range(max_iter):
        mlp_head.to(device)
        optimizer.zero_grad()
        selected_features = mlp_head(context_features)
        loss, stat, _ = prototype_loss(selected_features, context_labels,
                                       selected_features, context_labels, distance=distance, prot_type='avg')

        loss.backward()
        optimizer.step()
    
        if i == max_iter - 1:
            with torch.no_grad():
                selected_context = mlp_head(context_features)
                selected_target = mlp_head(target_features)
                _, val_stat, _ = prototype_loss(
                    selected_context, context_labels,
                    selected_target, target_labels, distance=args['test.distance']
                )
    return stat, val_stat
