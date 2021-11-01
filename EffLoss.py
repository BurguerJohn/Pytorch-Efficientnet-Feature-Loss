import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms

class EfficLoss(torch.nn.Module):
    def __init__(self, rescale):
        super(EfficLoss, self).__init__()
        self.features  = models.efficientnet_b7(pretrained=True).eval().features
        if rescale:
          self.normalize = transforms.Compose([transforms.Resize(224), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        else:
          self.normalize = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        
        self.l1  = nn.L1Loss()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y):
        X = self.normalize(X)
        Y = self.normalize(Y)

        indices = [1, 2, 3, 4, 5]
        weight = [1 / 16, 1 / 8, 1/ 4, 1, 1]
        
        k = 0
        loss = 0
        
        for i in range(indices[-1] + 1):
          X = self.features[i](X)
          Y = self.features[i](Y)
          if i in indices:
            curr = self.l1(X, Y) * weight[k]
            #curr = (X - Y).abs().mean() * weight[k]
            loss += curr
            k += 1
        return loss