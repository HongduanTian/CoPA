
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import args
from utils import device
from models.tsa import conv_tsa
from models.losses import symmetric_CE
from models.transformer import Transformer
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, MetaDatasetBatchReader, TRAIN_METADATASET_NAMES,
                                      ALL_METADATASET_NAMES) 


def compute_prototypes(embeddings:torch.tensor, labels:torch.tensor) -> torch.tensor:
    """
    Compute prototypes for each class in a given task.
    Args:
        embeddings: torch.tensor. The embedded features of a set of given images, with the shape [num_bacth, c, h, w]
        labels: torch.tensor. The corresponding labels of the embeddings, with the shape [num_batch, ].
    Return:
        prototypes: torch.tensor. A set of prototypes of the given embeddings, with the shape [num_classes, c, h, w].
    """
    unique_labels = torch.range(start=0, end=torch.max(labels)).unsqueeze(dim=1).type_as(labels)    # shape: [num_cls, 1]
    indicate_mat = unique_labels.eq(labels.reshape(1, list(labels.shape)[0])).type_as(embeddings)
    flatten_protos = torch.matmul(indicate_mat, embeddings.flatten(1)) / indicate_mat.sum(dim=1, keepdim=True)
    _, c, h, w = embeddings.size()
    prototypes = flatten_protos.view(flatten_protos.size(0), c, h, w)
    return prototypes


def pa(features, vars):

    return F.conv2d(features, vars[0])


def pa_adaptation(context_feat:torch.tensor, proto_feat:torch.tensor, context_labels:torch.tensor,
                  target_feat:torch.tensor, target_labels:torch.tensor,
                  prototype_encoder, feature_encoder, dataset_name:str, max_iter:int=50) -> dict:
    """
    Args:
        context_feat: torch.tensor. Context features (extracted from support data);
        proto_feat: torch.tensor. Prototypes (Average of context features within each class);
        context_labels: torch.tensor. Labels of support data;
        target_feat: torch.tensor. Query features (extracted from query data);
        target_labels: torch.tensor. Labels of query data;
        prototype_encoder: nn.Module. Encoder model for prototypes;
        feature_encoder: nn.Module. Encoder model for context features;
        max_iter: int. Maximum of learning iteration of each episode;
    Return:
        data_recoder: dict. A dictionary of running results.
    """
    
    # Data recorder
    data_recorder = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': []
    }

    # Reset Transformer model parameter
    if args['encoder.type'] == 'transformer':
        assert prototype_encoder is not None
        assert feature_encoder is not None
        
        prototype_encoder.reset_params()
        prototype_encoder.to(device)
        
        feature_encoder.reset_params()
        feature_encoder.to(device)
        
        # learning rate settings
        if dataset_name in ['traffic_sign', 'mnist']:
            lr = 1e-3
        else:
            lr = 1e-4
        
        weight_decay = 0.0 if dataset_name in ['traffic_sign', 'mnist'] else 0.1

        # Optimizer
        optimizer = torch.optim.Adam([{'params': feature_encoder.parameters()},
                                      {'params': prototype_encoder.parameters()}], 
                                      lr=lr,
                                      weight_decay=weight_decay)
    else:
        feat_dim = context_feat.size(1)
        feat_linear_vars = [torch.eye(feat_dim, feat_dim).unsqueeze(-1).unsqueeze(-1).to(device).requires_grad_(True)]
        proto_linear_vars = [torch.eye(feat_dim, feat_dim).unsqueeze(-1).unsqueeze(-1).to(device).requires_grad_(True)]

        # learning rate settings
        if dataset_name in ['traffic_sign', 'mnist']:
            lr = 5e-3
        else:
            lr = 1e-3

        #weight_decay = 0.1
        weight_decay = 0.0 if dataset_name in ['traffic_sign', 'mnist'] else 0.1
        
        # Optimizer
        optimizer = torch.optim.Adam([{'params': feat_linear_vars},
                                      {'params': proto_linear_vars,}], 
                                      lr=lr,
                                      weight_decay=weight_decay)

    # Prepare prototypes
    prototypes = compute_prototypes(proto_feat, context_labels) # shape: [num_cls, c, h, w]

    # Adaptation phase
    for i in range(max_iter):

        with torch.no_grad():
            if feature_encoder is not None:
                prototype_encoder.eval()
                feature_encoder.eval()
                transformed_target_feats = feature_encoder(target_feat)
                transformed_prototypes = prototype_encoder(prototypes)
            else:
                target_feat = F.adaptive_avg_pool2d(target_feat, (1, 1))
                prototypes = F.adaptive_avg_pool2d(prototypes, (1, 1))
                transformed_target_feats = pa(target_feat, feat_linear_vars)
                transformed_prototypes = pa(prototypes, proto_linear_vars)

            _, dynamic_dict = symmetric_CE(transformed_target_feats.flatten(1), transformed_prototypes.flatten(1), target_labels)

            data_recorder['val_losses'].append(dynamic_dict['loss'])
            data_recorder['val_accs'].append(dynamic_dict['acc'])

            if feature_encoder is not None:
                feature_encoder.train()
                prototype_encoder.train()
        
        optimizer.zero_grad()

        if feature_encoder is not None:
            transformed_context_feats = feature_encoder(context_feat)
            transformed_prototypes = prototype_encoder(prototypes)   # shape: [num_cls, c, h, w]
        else:
            context_feat = F.adaptive_avg_pool2d(context_feat, (1, 1))
            prototypes = F.adaptive_avg_pool2d(prototypes, (1, 1))
            transformed_context_feats = pa(context_feat, feat_linear_vars)
            transformed_prototypes = pa(prototypes, proto_linear_vars)

        loss, train_dynamic = symmetric_CE(transformed_context_feats.flatten(1), 
                                           transformed_prototypes.flatten(1), context_labels)
        
        data_recorder['train_losses'].append(train_dynamic['loss'])
        data_recorder['train_accs'].append(train_dynamic['acc'])

        loss.backward()
        optimizer.step()

        if i == max_iter - 1:
            with torch.no_grad():
                
                if feature_encoder is not None:
                    prototype_encoder.eval()
                    feature_encoder.eval()
                    transformed_target_feats = feature_encoder(target_feat)
                    transformed_prototypes = prototype_encoder(prototypes)
                else:
                    target_feat = F.adaptive_avg_pool2d(target_feat, (1, 1))
                    prototypes = F.adaptive_avg_pool2d(prototypes, (1, 1))
                    transformed_target_feats = pa(target_feat, feat_linear_vars)
                    transformed_prototypes = pa(prototypes, proto_linear_vars)

                _, dynamic_dict = symmetric_CE(transformed_target_feats.flatten(1), transformed_prototypes.flatten(1), target_labels)

                data_recorder['val_losses'].append(dynamic_dict['loss'])
                data_recorder['val_accs'].append(dynamic_dict['acc'])
    
    return data_recorder


class wide_conv_tsa(nn.Module):
    def __init__(self, orig_conv):
        super(wide_conv_tsa, self).__init__()
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        
        self.dropout = nn.Dropout(p=0)
        self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        self.alpha.requires_grad = True
        self.garm = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        self.garm.requires_grad = True
    
    def forward(self, x):
        y = self.conv(x)
        y1 = F.conv2d(x, self.alpha, stride=self.conv.stride)
        #y2 = F.conv2d(x, self.garm, stride=self.conv.stride)
        y += y1
        return y


class clipcfc_tsa_backbone(nn.Module):
    def __init__(self, original_backbone, modified_list:list):
        super(clipcfc_tsa_backbone, self).__init__()
        
        # backbone settings
        for k, v in original_backbone.named_parameters():
            v.requires_grad = False
        
        for layer_name in modified_list:
            if layer_name == 'layer1':
                layer = original_backbone.layer1
            elif layer_name == 'layer2':
                layer = original_backbone.layer2
            elif layer_name == 'layer3':
                layer = original_backbone.layer3
            elif layer_name == 'layer4':
                layer = original_backbone.layer4
            else:
                raise ValueError("Invalid layer name.")

            for block in layer:
                for name, m in block.named_children():
                    if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                        new_conv = wide_conv_tsa(m)
                        setattr(block, name, new_conv)

        self.backbone = original_backbone
    
    def embed(self, x:torch.tensor) -> torch.tensor:
        return self.backbone.embed(x, is_pooling=False)

    def get_state_dict(self):
        return self.backbone.state_dict()

    def reset(self):
        for k, v in self.backbone.named_parameters():
            if 'alpha' in k or 'garm' in k:
                if v.size(0) > 1:
                    v.data = 0.0001 * torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)
                else:
                    v.data = 0.0001 * torch.ones(v.size()).to(device)



def tsa_adaptation(context_images:torch.tensor, context_labels:torch.tensor,
                   target_images:torch.tensor, target_labels:torch.tensor,
                   backbone, prototype_encoder, feature_encoder, dataset_name:str, max_iter:int=40):
    # Data recorder
    data_recorder = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': []
    }

    # Model settings
    backbone.eval()
    tsa_opt = 'alpha+beta'
    alpha_params = [v for k, v in backbone.named_parameters() if 'alpha' in k] 

    feat_dim = 512
    feat_linear_vars = [torch.eye(feat_dim, feat_dim).unsqueeze(-1).unsqueeze(-1).to(device).requires_grad_(True)]
    proto_linear_vars = [torch.eye(feat_dim, feat_dim).unsqueeze(-1).unsqueeze(-1).to(device).requires_grad_(True)]
        
    # learning rate settings
    if dataset_name in ['traffic_sign', 'mnist', 'cifar10', 'cifar100']:
        lr = 5e-3
        scale = 0.5
    else:
        lr = 1e-3
        scale = 0.1

    if args['test.type'] == '1shot':
        lr *= 0.1
    #weight_decay = 0.1
    weight_decay = 0.0 if dataset_name in ['traffic_sign', 'mnist'] else 0.1
        
    # Optimizer
    optimizer = torch.optim.Adam([{'params': alpha_params, 'lr':lr*scale},
                                  {'params': feat_linear_vars},
                                  {'params': proto_linear_vars,}],
                                  lr=lr,
                                  weight_decay=weight_decay)

    for i in range(max_iter):
        # validation
        with torch.no_grad():
            # feature embeddings
            context_features = backbone.embed(context_images)
            target_features = backbone.embed(target_images)
            prototypes = compute_prototypes(context_features, context_labels)
            if feature_encoder is not None:
                prototype_encoder.eval()
                feature_encoder.eval()
                transformed_target_feats = feature_encoder(target_features)
                transformed_prototypes = prototype_encoder(prototypes)
            else:
                target_features = F.adaptive_avg_pool2d(target_features, (1, 1))
                prototypes = F.adaptive_avg_pool2d(prototypes, (1, 1))
                transformed_target_feats = pa(target_features, feat_linear_vars)
                transformed_prototypes = pa(prototypes, proto_linear_vars)
            
            _, dynamic_dict = symmetric_CE(transformed_target_feats.flatten(1), 
                                           transformed_prototypes.flatten(1), target_labels)

            data_recorder['val_losses'].append(dynamic_dict['loss'])
            data_recorder['val_accs'].append(dynamic_dict['acc'])

            if feature_encoder is not None:
                feature_encoder.train()
                prototype_encoder.train()

        optimizer.zero_grad()

        context_features = backbone.embed(context_images)
        target_features = backbone.embed(target_images)

        prototypes = compute_prototypes(context_features, context_labels)

        if feature_encoder is not None:
            transformed_context_features = feature_encoder(context_features)
            transformed_prototypes = prototype_encoder(prototypes)
        else:
            transformed_context_features = pa(F.adaptive_avg_pool2d(context_features, (1, 1)),
                                              feat_linear_vars)
            transformed_prototypes = pa(F.adaptive_avg_pool2d(prototypes, (1, 1)),
                                        proto_linear_vars)
        
        loss, train_dynamic = symmetric_CE(transformed_context_features.flatten(1),
                                           transformed_prototypes.flatten(1),
                                           context_labels)
        
        data_recorder['train_losses'].append(train_dynamic['loss'])
        data_recorder['train_accs'].append(train_dynamic['acc'])

        loss.backward()
        optimizer.step()

        if i == max_iter - 1:
            with torch.no_grad():
                # feature embeddings
                context_features = backbone.embed(context_images)
                target_features = backbone.embed(target_images)
                prototypes = compute_prototypes(context_features, context_labels)
                if feature_encoder is not None:
                    prototype_encoder.eval()
                    feature_encoder.eval()
                    transformed_target_feats = feature_encoder(target_features)
                    transformed_prototypes = prototype_encoder(prototypes)
                else:
                    target_features = F.adaptive_avg_pool2d(target_features, (1, 1))
                    prototypes = F.adaptive_avg_pool2d(prototypes, (1, 1))
                    transformed_target_feats = pa(target_features, feat_linear_vars)
                    transformed_prototypes = pa(prototypes, proto_linear_vars)
            
                _, dynamic_dict = symmetric_CE(transformed_target_feats.flatten(1), 
                                               transformed_prototypes.flatten(1), target_labels)

                data_recorder['val_losses'].append(dynamic_dict['loss'])
                data_recorder['val_accs'].append(dynamic_dict['acc'])
    
    return data_recorder
            

            


   

    
