import torch
def save(model,optimizer,epoch,filename="checkpoint.pt"):
    checkpoint={
        "epoch":epoch,
        "modelstate":model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }
    torch.save(checkpoint,filename)
