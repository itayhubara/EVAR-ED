import torch
import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['evar_model']

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightedCrossEntropyLoss,self).__init__()
        self.epsilon=1e-12
    def cross_entropy(self,predictions, targets,idx, weights=None):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
                targets (N, k) ndarray
                Returns: scalar
                """
        #weights=None
        predictions = torch.clamp(predictions, self.epsilon, 1. - self.epsilon)
        N = predictions.shape[0]
        weight_idx=weights.gather(0,idx) if weights is not None else 1.0
        #print(weight_idx[0])
        #import pdb; pdb.set_trace()
        ce = -torch.sum(weight_idx*torch.sum(targets*torch.log(predictions+1e-9),1))/N
        if weights is not None: ce=ce/weight_idx.mean()
        return ce

    def forward(self,input,target,idx,weights):
        return self.cross_entropy(input,target,idx,weights)

class EvarModel(nn.Module):

    def __init__(self, num_classes=1000):
        super(EvarModel, self).__init__()
        self.evar = nn.Sequential(
            #nn.BatchNorm1d(40),
            nn.Linear(40, 200, bias=True),
            #nn.Softmax(),
            nn.Hardtanh(),
            nn.Linear(200, 100, bias=True),
            #nn.Softmax(),
            nn.Hardtanh(),
            #nn.Linear(1024, 512, bias=True),
            #nn.BatchNorm1d(512),
            #nn.Dropout(0.2),
            #nn.BatchNorm1d(512),
            #nn.ReLU(inplace=True),
            nn.Linear(100,2, bias=True),
            nn.Softmax()
        )
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 1e-3,
                'weight_decay': 5e-4, 'momentum': 0.9},
            100: {'lr': 1e-3},
            300: {'lr': 1e-5, 'weight_decay': 0},
            350: {'lr': 5e-5},
            400: {'lr': 1e-5}
        }
    def init_model(model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                n = m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        y = self.evar(input)
        return y


def evar_model(**kwargs):
    num_classes = getattr(kwargs, 'num_classes', 2)
    return EvarModel(num_classes)
