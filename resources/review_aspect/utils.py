import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer, util

with open('resources/review_aspect/synonyms.json') as file:
    synonyms = {k.lower(): v for k, v in json.loads(file.read()).items()}

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        BCE_loss = F.binary_cross_entropy(probs, targets, reduction="none")
        pt = probs * targets + (1 - probs) * (1 - targets)  # pt = p if t=1 else 1-p
        focal_loss = self.alpha * (1 - pt).pow(self.gamma) * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def merge_synonyms(labels):
    output = []
    for label in labels:
        label = label.rstrip(',')
        if label.lower() in synonyms:
            output.append(synonyms[label.lower()])
        else:
            output.append(label)
    return output

def jaccard_similarity(list_a, list_b):
    set_a, set_b = set(list_a), set(list_b)
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(intersection) / len(union) if union else 0

def dice_coefficient(set_a, set_b):
    intersection = len(set_a & set_b)
    return 2 * intersection / (len(set_a) + len(set_b)) if (len(set_a) + len(set_b)) > 0 else 0

def overlap_coefficient(set_a, set_b):
    intersection = len(set_a & set_b)
    return intersection / min(len(set_a), len(set_b)) if min(len(set_a), len(set_b)) > 0 else 0

def kulczynski_similarity(set_a, set_b):
    intersection = len(set_a & set_b)
    len_a = len(set_a)
    len_b = len(set_b)
    return 0.5 * (intersection / len_a + intersection / len_b) if (len(set_a) * len(set_b)) != 0 else 0

def calculate_jaccard_similarity_for_lists(list_a, list_b):
    if len(list_a) != len(list_b):
        raise ValueError('Lists must have the same length.')
    
    similarities = []
    for sublist_a, sublist_b in zip(list_a, list_b):
        similarity = jaccard_similarity(sublist_a, sublist_b)
        similarities.append(similarity)
    return similarities

def sbert_similarity(text_a, text_b, model):
    embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()