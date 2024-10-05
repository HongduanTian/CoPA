import torch
import gin
import numpy as np

from torch import nn
import torch.nn.functional as F
from models.cka import linear_CKA, kernel_CKA
from models.scm import scm
from utils import device
from config import args

def cal_dist(inputs, inputs_center):
    n = inputs.size(0)
    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs, inputs.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    # Compute pairwise distance, replace by the official when merged
    dist_center = torch.pow(inputs_center, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_center = dist_center + dist_center.t()
    dist_center.addmm_(1, -2, inputs_center, inputs_center.t())
    dist_center = dist_center.clamp(min=1e-12).sqrt()  # for numerical stability
    loss = torch.mean(torch.norm(dist-dist_center,p=2))
    return loss

def cal_dist_cosine(inputs, inputs_center):
    # We would like to perform cosine similarity for pairwise distance
    n = inputs.size(0)
    # Compute pairwise distance, replace by the official when merged
    dist = 1 - F.cosine_similarity(inputs.unsqueeze(1), inputs, dim=-1, eps=1e-30)
    dist = dist.clamp(min=1e-12)

    # Compute pairwise distance, replace by the official when merged
    dist_center = 1 - F.cosine_similarity(inputs_center.unsqueeze(1), inputs_center, dim=-1, eps=1e-30)
    dist_center = dist_center.clamp(min=1e-12)
    loss = torch.mean(torch.norm(dist-dist_center,p=2))
    return loss

def distillation_loss(fs, ft, opt='l2', delta=0.5):
    if opt == 'l2':
        return (fs-ft).pow(2).sum(1).mean()
    if opt == 'l1':
        return (fs-ft).abs().sum(1).mean()
    if opt == 'huber':
        l1 = (fs-ft).abs()
        binary_mask_l1 = (l1.sum(1) > delta).type(torch.FloatTensor).unsqueeze(1).cuda()
        binary_mask_l2 = (l1.sum(1) <= delta).type(torch.FloatTensor).unsqueeze(1).cuda()
        loss = (l1.pow(2) * binary_mask_l2 * 0.5).sum(1) + (l1 * binary_mask_l1).sum(1) * delta - delta ** 2 * 0.5
        loss = loss.mean()
        return loss
    if opt == 'rkd':
        return cal_dist(fs, ft)
    if opt == 'cosine':
        return 1 - F.cosine_similarity(fs, ft, dim=-1, eps=1e-30).mean()
    if opt == 'rkdcos':
        return cal_dist_cosine(fs, ft)
    if opt == 'linearcka':
        return 1 - linear_CKA(fs, ft)
    if opt == 'kernelcka':
        return 1 - kernel_CKA(fs, ft)


def cross_entropy_loss(logits, targets):
    log_p_y = F.log_softmax(logits, dim=1)
    preds = log_p_y.argmax(1)
    labels = targets.type(torch.long)
    loss = F.nll_loss(log_p_y, labels, reduction='mean')
    acc = torch.eq(preds, labels).float().mean()
    stats_dict = {'loss': loss.item(), 'acc': acc.item()}
    pred_dict = {'preds': preds.cpu().numpy(), 'labels': labels.cpu().numpy()}
    return loss, stats_dict, pred_dict

# logistic regression
def lr_loss(support_embeddings, support_labels,
            query_embeddings, query_labels, normalize=False):
    n_way = len(query_labels.unique())
    if normalize:
        support_embeddings = F.normalize(support_embeddings, dim=-1, p=2)
        query_embeddings = F.normalize(query_embeddings, dim=-1, p=2)
    support_embeddings = support_embeddings.detach().cpu().numpy()
    query_embeddings = query_embeddings.detach().cpu().numpy()
    support_labels = support_labels.view(-1).cpu().numpy()
    clf = LogisticRegression(penalty='none',
                             random_state=0,
                             C=1.0,
                             solver='lbfgs',
                             max_iter=1000,
                             multi_class='multinomial')
    clf.fit(support_embeddings, support_labels)
    logits_ = clf.predict(query_embeddings)
    logits_ = torch.from_numpy(logits_).to(query_labels.device)
    logits = torch.zeros(query_labels.size(0), n_way).to(query_labels.device).scatter_(1, logits_.view(-1,1), 1) * 10

    return cross_entropy_loss(logits, query_labels)

# support vector machines
def svm_loss(support_embeddings, support_labels,
            query_embeddings, query_labels, normalize=False):
    n_way = len(query_labels.unique())
    if normalize:
        support_embeddings = F.normalize(support_embeddings, dim=-1, p=2)
        query_embeddings = F.normalize(query_embeddings, dim=-1, p=2)
    support_embeddings = support_embeddings.detach().cpu().numpy()
    query_embeddings = query_embeddings.detach().cpu().numpy()
    support_labels = support_labels.view(-1).cpu().numpy()
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto',
                                              C=1,
                                              kernel='linear',
                                              decision_function_shape='ovr'))
    clf.fit(support_embeddings, support_labels)
    logits_ = clf.predict(query_embeddings)
    logits_ = torch.from_numpy(logits_).to(query_labels.device)
    logits = torch.zeros(query_labels.size(0), n_way).to(query_labels.device).scatter_(1, logits_.view(-1,1), 1) * 10

    return cross_entropy_loss(logits, query_labels)

# Mahalanobis Distance from Simple CNAPS (scm)
def scm_loss(support_embeddings, support_labels,
            query_embeddings, query_labels, normalize=False):
    n_way = len(query_labels.unique())
    if normalize:
        support_embeddings = F.normalize(support_embeddings, dim=-1, p=2)
        query_embeddings = F.normalize(query_embeddings, dim=-1, p=2)
    logits = torch.logsumexp(scm(support_embeddings, support_labels, query_embeddings), dim=0)
    return cross_entropy_loss(logits, query_labels)
    
# NCC
def prototype_loss(support_embeddings, support_labels,
                   query_embeddings, query_labels, distance='cos', prot_type='avg'):
    n_way = len(query_labels.unique())

    if prot_type == 'avg':
        prots = compute_avg_prototypes(support_embeddings, support_labels).unsqueeze(0)
    elif prot_type == 'max':
        prots = compute_max_prototype(support_embeddings, support_labels).unsqueeze(0)
    elif prot_type == 'sum':
        prots = compute_sum_prototype(support_embeddings, support_labels).unsqueeze(0)
    elif prot_type == 'max_sample':
        prots = compute_most_sim_prototype(support_embeddings, support_labels)
    else:
        raise ValueError("Unrecognized type of prototypes.")

    embeds = query_embeddings.unsqueeze(1)

    if distance == 'l2':
        logits = -torch.pow(embeds - prots, 2).sum(-1)    # shape [n_query, n_way]
    elif distance == 'cos':
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 10
    elif distance == 'lin':
        logits = torch.einsum('izd,zjd->ij', embeds, prots)
    elif distance == 'corr':
        logits = F.normalize((embeds * prots).sum(-1), dim=-1, p=2) * 10
    elif distance == 'cka':
        logits = kernel_CKA(embeds, prots)

    return cross_entropy_loss(logits, query_labels)


def compute_prototypes(embeddings, labels, n_way):
    prots = torch.zeros(n_way, embeddings.shape[-1]).type(
        embeddings.dtype).to(embeddings.device)
    for i in range(n_way):
        if torch.__version__.startswith('1.1'):
            prots[i] = embeddings[(labels == i).nonzero(), :].mean(0)
        else:
            prots[i] = embeddings[(labels == i).nonzero(as_tuple=False), :].mean(0)
    return prots


def compute_max_prototype(embeddings, labels):
    n_way = len(labels.unique())
    protos = torch.zeros(n_way, embeddings.shape[-1]).type(embeddings.dtype).to(embeddings.device)
    for i in range(n_way):
        if torch.__version__.startswith('1.1'):
            protos[i], _ = embeddings[(labels == i).nonzero(), :].max(dim=0)
        else:
            protos[i], _ = embeddings[(labels == i).nonzero(as_tuple=False), :].max(dim=0)
    return protos


def compute_most_sim_prototype(embeddings, labels):

    n_way = len(labels.unique())
    protos = torch.zeros(n_way, embeddings.shape[-1]).type(embeddings.dtype).to(embeddings.device)
    
    for i in range(n_way):
        if torch.__version__.startswith('1.1'):
            class_samples = embeddings[(labels == i).nonzero(), :]
            num_samples = class_samples.shape[0]
            normed_class_samples = class_samples.flatten(1) / class_samples.flatten(1).norm(dim=1, keepdim=True)
            sim_score_mat = torch.mm(normed_class_samples, normed_class_samples.t()) - torch.eye(num_samples).to(embeddings.device)  # [num_samples, ]
            protos[i] = class_samples[torch.argmax(sim_score_mat.sum(dim=1)), :]
        else:
            class_samples = embeddings[(labels == i).nonzero(as_tuple=False), :]
            num_samples = class_samples.shape[0]
            normed_class_samples = class_samples.flatten(1) / class_samples.flatten(1).norm(dim=1, keepdim=True)
            sim_score_mat = torch.mm(normed_class_samples, normed_class_samples.t()) - torch.eye(num_samples).to(embeddings.device)   # [num_samples, ]
            protos[i] = class_samples[torch.argmax(sim_score_mat.sum(dim=1)), :]
    return protos


def compute_sum_prototype(embeddings, labels, is_flatten=True):
    n_way = len(labels.unique())
    prots = torch.zeros(n_way, embeddings.shape[-1]).type(
        embeddings.dtype).to(embeddings.device)
    for i in range(n_way):
        if torch.__version__.startswith('1.1'):
            prots[i] = embeddings[(labels == i).nonzero(), :].sum(0)
        else:
            prots[i] = embeddings[(labels == i).nonzero(as_tuple=False), :].sum(0)
    return prots


def compute_avg_prototypes(embeddings:torch.tensor, labels:torch.tensor, is_flatten:bool=True) -> torch.tensor:
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
    if not is_flatten:
        _, c, h, w = embeddings.size()
        prototypes = flatten_protos.view(flatten_protos.size(0), c, h, w)
    else:
        prototypes = flatten_protos
    return prototypes

# prototypical loss rewrite
def prototypical_loss(embeddings:torch.tensor, protos:torch.tensor, labels:torch.tensor):
    """
    Calculate the prototypical loss between embeddings and prototypes generated from transformer.
    Args:
        embeddings: torch.tensor. Feature embeddings of data. Shape: [B, SD]
        protos: torch.tensor. Embeddings of prototypes generated from Transformer. Shape: [num_cls, SD]
        support_labels: torch.tensor. The labels of support data.
    Return:
        loss: torch.tensor. The contrastive loss for optimization.
        dynamic_dict: dict. A dict of dynamic values.
    """
    # Normalize both embeddings and prototypes
    embeddings = F.normalize(embeddings, dim=-1)
    protos = F.normalize(protos, dim=-1)

    logits = F.cosine_similarity(embeddings.unsqueeze(1),
                                 protos.unsqueeze(0),
                                 dim=-1,
                                 eps=1e-30) * 10
    # acc
    log_p_y = F.log_softmax(logits, dim=1)
    preds = log_p_y.argmax(1)
    acc = torch.eq(preds, labels.type(torch.long)).float().mean()

    # loss
    loss = F.nll_loss(log_p_y, labels.type(torch.long), reduction='mean')

    dynamic_dict = {
        'loss': loss.item(),
        'acc': acc.item()
    }
    return loss, dynamic_dict


def symmetric_CE(embeddings:torch.tensor, protos:torch.tensor, labels:torch.tensor) -> torch.tensor:
    """
    Calculate symmetric cross-entropy loss between embeddings and prototypes generated from transformer.
    Args:
        embeddings: torch.tensor. Feature embeddings of data. Shape: [B, SD]
        protos: torch.tensor. Embeddings of prototypes generated from Transformer. Shape: [num_cls, SD]
        labels: torch.tensor. The labels of support data.
    Return:
        loss: torch.tensor. The contrastive loss for optimization.
        dynamic_dict: dict. A dict of dynamic values.
    """
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    protos = protos / protos.norm(dim=1, keepdim=True)
    
    # spread protos, protos range from 0 to num_cls - 1 from row 1 to row num_cls - 1
    one_hot_labels = F.one_hot(labels)  # shape: [num_samples, num_cls]
    proto_labels = one_hot_labels.type_as(protos) @ protos  # shape: [num_samples, dim]

    logits = torch.exp(torch.tensor(args['SCE.tau'])) * embeddings @ proto_labels.t().type_as(embeddings)

    pseudo_labels = torch.arange(logits.size(0), device=device).long()

    loss = 0.5 * (F.cross_entropy(logits, pseudo_labels) + F.cross_entropy(logits.t(), pseudo_labels))

    # acc
    acc_logits = F.cosine_similarity(embeddings.unsqueeze(1),
                                 protos.unsqueeze(0),
                                 dim=-1,
                                 eps=1e-30) * 10
    
    log_p_y = F.log_softmax(acc_logits, dim=1)
    preds = log_p_y.argmax(1)
    acc = torch.eq(preds, labels.type(torch.long)).float().mean()

    dynamic_dict = {
        'loss': loss.item(),
        'acc': acc.item()
    }
    return loss, dynamic_dict


# knn
def knn_loss(support_embeddings, support_labels,
             query_embeddings, query_labels, distance='cos'):
    n_way = len(query_labels.unique())

    prots = support_embeddings
    embeds = query_embeddings.unsqueeze(1)

    if distance == 'l2':
        dist = -torch.pow(embeds - prots, 2).sum(-1)    # shape [n_query, n_way]
    elif distance == 'cos':
        dist = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 10
    elif distance == 'lin':
        dist = torch.einsum('izd,zjd->ij', embeds, prots)
    _, inds = torch.topk(dist, k=1)

    logits = torch.zeros(embeds.size(0), n_way).to(embeds.device).scatter_(1, support_labels[inds.flatten()].view(-1,1), 1) * 10

    return cross_entropy_loss(logits, query_labels)


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss


class AdaptiveCosineNCC(nn.Module):
    def __init__(self):
        super(AdaptiveCosineNCC, self).__init__()
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)

    def forward(self, support_embeddings, support_labels,
                query_embeddings, query_labels, return_logits=False):
        n_way = len(query_labels.unique())

        prots = compute_prototypes(support_embeddings, support_labels, n_way).unsqueeze(0)
        embeds = query_embeddings.unsqueeze(1)
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * self.scale

        if return_logits:
            return logits

        return cross_entropy_loss(logits, query_labels)