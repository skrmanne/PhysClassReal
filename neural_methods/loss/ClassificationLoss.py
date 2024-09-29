import torch
from torch import nn

def get_classes(signal):
    # run through the signal and append 1.0 if signal[i] > signal[i-1] else 0.0
    B = signal.shape[0]
    classes = torch.zeros_like(signal)

    for b in range(B):
        for i in range(1, signal.shape[1]):
            classes[b,i] = 1 if signal[b,i] > signal[b,i-1] else 0
    
    classes.requires_grad = True
    return classes

class ClassLoss(nn.Module):
    """
    Classification loss using BCELoss by modifying the respiration signal to class labels.
    """
    def __init__(self):
        super(ClassLoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, preds, labels):
        # convert respiration signal to class labels. - data already converted to labels in data loader.
        # labels = get_classes(labels)

        # print(preds[0][::40], labels[0][::40])
        # print count of ones and zeros in labes
        pos_cnt = torch.sum(labels)
        neg_cnt = labels.shape[0]*labels.shape[1] - torch.sum(labels)

        #self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_cnt/pos_cnt]).cuda())
        return self.bce(preds, labels)
        
