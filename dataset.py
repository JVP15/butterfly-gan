import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
# from torchvision.utils import save_image

class ButterflyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
#         print(self.list_files)
        
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :256, :] #based on train image dimensions
        target_image = image[:, 256:, :]
        
        augmentations = config.both_transform(image = input_image, image0 = target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]
        
        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_input(image=target_image)["image"]
        
        return input_image, target_image

        
        