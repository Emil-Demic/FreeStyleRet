import os
import json
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader


class StyleDataset(Dataset):
    def __init__(self, root_path, json_path, image_transform, tokenizer):
        self.root_path = root_path
        self.dataset = json.load(open(json_path,'r'))
        self.image_transform = image_transform
    

    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, index):
        caption_path = os.path.join(self.root_path, self.dataset[index]['caption'])
        image_path = os.path.join(self.root_path, self.dataset[index]['image'])
        negative_path = os.path.join(self.root_path, self.dataset[np.random.randint(1, len(self.dataset))]['image'])
        
        f = open(caption_path, 'r')
        caption = f.readline().replace('\n', '')
        pair_image = self.image_transform(Image.open(image_path))
        negative_image = self.image_transform(Image.open(negative_path))

        return [caption, pair_image, negative_image]