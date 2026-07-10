import torch
def dice(prediction,mask):
    prediction=torch.sigmoid(prediction)
    prediction= (prediction > 0.5).float()
    inter=(prediction*mask).sum()
    dic=(2*inter)/(prediction.sum() + mask.sum() + 1e-8)
    return dic