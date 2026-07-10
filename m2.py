import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Data(Dataset):
    def __init__(self,img_dir,mask_dir,transform=None,mask_transform=None,img_size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mask_transform=mask_transform
        self.img_size=img_size
        self.images =os.listdir(img_dir)
        if transform is None:
            self.transform=transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor()
            ])
        else:
            self.transform=transform
        if mask_transform is None:
            self.mask_transform=transforms.Compose([
                transforms.Resize((img_size,img_size),interpolation=Image.NEAREST),
                transforms.ToTensor()
            ])
        else :
            self.mask_transform=mask_transform


    def __len__(self):
        return len(self.images)
      
    def __getitem__(self,index):
        image_name=self.images[index]
        img_path = os.path.join(self.img_dir,image_name)
        mask_path = os.path.join(self.mask_dir,image_name)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        if mask.size(1) == 3:
            mask = mask[:, 0:1, :, :]
        
        return image,mask
    