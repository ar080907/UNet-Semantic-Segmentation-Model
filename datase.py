from torch.utils.data import DataLoader
from m2 import Data
train_data = Data(img_dir="data/train/images",mask_dir="data/train/masks",transform=None,mask_transform=None,img_size=256)
train_loader = DataLoader(train_data, batch_size=4,num_workers=0, shuffle=True)
validation_data = Data(img_dir="data/validation/images",mask_dir="data/validation/masks",transform=None,mask_transform=None,img_size=256)
validation_loader = DataLoader(validation_data, batch_size=4,num_workers=0, shuffle=False)