"""
Developed by Daniel Crovo

"""

from torch.utils.data import Dataset 
import os
import numpy as np
from PIL import Image

class CIMC_data(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images =os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path =os.path.join(self.mask_dir, self.images[index].replace('.jpg','_mask.gif'))
        image = np.array(Image.open(img_path).convert('RGB')) # RGB space
        mask = np.array(Image.open(mask_path).convert('L')) # Grey scale
        mask[mask==255.0] = 1.0 # To work with posterior probabilities [0=black, 1=white]

        if self.transform is not None:
            transformations = self.transform(image=image, mask=mask)
            image = transformations['image']
            mask = transformations['mask']
        return image, mask
    