import torch
import torch.nn as nn
import torch.nn.functional as F

class arcface_loss(nn.Module):
    def __init__(self, embedding_size, num_classes, scale=30.0, margin=0.5):
        super(arcface_loss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes
        self.weights = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weights))
        theta = torch.acos(torch.clamp(cosine, -1.0, 1.0))
        target_logits = torch.cos(theta + self.margin)
        logits = cosine * (1 - labels) + target_logits * labels
        logits *= self.scale
        return F.cross_entropy(logits, labels)
