import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class StyleT2IDataset(Dataset):
    def __init__(self, root_path, json_path, image_transform):
        self.root_path = root_path
        self.dataset = json.load(open(json_path, 'r'))
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        caption_path = os.path.join(self.root_path, 'text/' + self.dataset[index]['caption'])
        image_path = os.path.join(self.root_path, 'images/' + self.dataset[index]['image'])
        neg = np.random.randint(1, len(self.dataset))
        while neg == index:
            neg = np.random.randint(1, len(self.dataset))
        negative_path = os.path.join(self.root_path, 'images/' + self.dataset[neg]['image'])

        f = open(caption_path, 'r')
        caption = f.readline().replace('\n', '')
        pair_image = self.image_transform(Image.open(image_path))
        negative_image = self.image_transform(Image.open(negative_path))

        return [caption, pair_image, negative_image]


class StyleI2IDataset(Dataset):
    def __init__(self, root_path, json_path, image_transform):
        self.root_path = root_path
        self.dataset = json.load(open(json_path, 'r'))
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ori_path = os.path.join(self.root_path, 'images/' + self.dataset[index]['image'])
        rand = np.random.randint(1, 4)
        if rand == 1:
            pair_path = os.path.join(self.root_path, 'sketch/' + self.dataset[index]['image'])
        elif rand == 2:
            pair_path = os.path.join(self.root_path, 'art/' + self.dataset[index]['image'])
        elif rand == 3:
            pair_path = os.path.join(self.root_path, 'mosaic/' + self.dataset[index]['image'])

        neg = np.random.randint(1, len(self.dataset))
        while neg == index:
            neg = np.random.randint(1, len(self.dataset))
        negative_path = os.path.join(self.root_path, 'images/' + self.dataset[neg]['image'])

        ori_image = self.image_transform(Image.open(ori_path))
        pair_image = self.image_transform(Image.open(pair_path))
        negative_image = self.image_transform(Image.open(negative_path))

        return [ori_image, pair_image, negative_image]


class T2ITestDataset(Dataset):
    def __init__(self, root_path, json_path, image_transform):
        self.root_path = root_path
        self.dataset = json.load(open(json_path, 'r'))
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        caption_path = os.path.join(self.root_path, 'text/' + self.dataset[index]['caption'])
        image_path = os.path.join(self.root_path, 'images/' + self.dataset[index]['image'])

        f = open(caption_path, 'r')
        caption = f.readline().replace('\n', '')
        pair_image = self.image_transform(Image.open(image_path))

        return [caption, pair_image, index]


class I2ITestDataset(Dataset):
    def __init__(self, root, mode="test", transforms_sketch=None, transforms_image=None):

        self.root = root

        val_path = "val_normal.txt"
        with open(os.path.join(self.root, val_path), 'r') as f:
            lines = f.readlines()
            val_ids = set(map(int, lines))

        self.files = []
        for i in range(1, 101):
            file_names = os.listdir(os.path.join(root, "images", str(i)))
            file_names = [file.split('.')[0] for file in file_names]
            if mode == "train":
                file_names = [file for file in file_names if int(file) not in val_ids]
            else:
                file_names = [file for file in file_names if int(file) in val_ids]
            for file_name in file_names:
                self.files.append(os.path.join(str(i), file_name + ".jpg"))

        self.transforms_sketch = transforms_sketch
        self.transforms_image = transforms_image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.root, "raster_sketches", self.files[idx])
        image_path = os.path.join(self.root, "images", self.files[idx])

        sketch = Image.open(sketch_path)
        image = Image.open(image_path)

        if self.transforms_sketch:
            sketch = self.transforms_sketch(sketch)

        if self.transforms_image:
            image = self.transforms_image(image)

        return sketch, image

    # def __init__(self, style, root_path, json_path, image_transform):
    #     self.style = style
    #     self.root_path = root_path
    #     self.dataset = json.load(open(json_path,'r'))
    #     self.image_transform = image_transform
    #
    #
    # def __len__(self):
    #     return len(self.dataset)
    #
    #
    # def __getitem__(self, index):
    #     ori_path = os.path.join(self.root_path, 'images/'+self.dataset[index]['image'])
    #     pair_path = os.path.join(self.root_path, '{}/'.format(self.style)+self.dataset[index]['image'])
    #
    #     ori_image = self.image_transform(Image.open(ori_path))
    #     pair_image = self.image_transform(Image.open(pair_path))
    #
    #     return [ori_image, pair_image, index]


class X2ITestDataset(Dataset):
    def __init__(self, style, root_path, json_path, image_transform):
        self.style = style
        self.root_path = root_path
        self.dataset = json.load(open(json_path, 'r'))
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        caption_path = os.path.join(self.root_path, 'text/' + self.dataset[index]['caption'])
        ori_path = os.path.join(self.root_path, 'images/' + self.dataset[index]['image'])
        pair_path = os.path.join(self.root_path, '{}/'.format(self.style) + self.dataset[index]['image'])

        f = open(caption_path, 'r')
        caption = f.readline().replace('\n', '')
        ori_image = self.image_transform(Image.open(ori_path))
        pair_image = self.image_transform(Image.open(pair_path))

        return [caption, ori_image, pair_image, index]


class VisualizationDataset(Dataset):
    def __init__(self, root_path, json_path, image_transform):
        self.root_path = root_path
        self.dataset = json.load(open(json_path, 'r'))
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ori_path = os.path.join(self.root_path, 'images/' + self.dataset[index]['image'])
        sketch_path = os.path.join(self.root_path, 'sketch/' + self.dataset[index]['image'])
        art_path = os.path.join(self.root_path, 'art/' + self.dataset[index]['image'])
        mosaic_path = os.path.join(self.root_path, 'mosaic/' + self.dataset[index]['image'])

        ori_image = self.image_transform(Image.open(ori_path))
        sketch_image = self.image_transform(Image.open(sketch_path))
        art_image = self.image_transform(Image.open(art_path))
        mosaic_image = self.image_transform(Image.open(mosaic_path))

        return [ori_image, sketch_image, art_image, mosaic_image, index]