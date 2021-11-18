from torch.utils import data
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import pandas as pd
# from torchvision.io import read_image
from PIL import Image
import torch
import numpy as np

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def get_transform(random_crop=True):
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = []
    if random_crop:
        transform.append(transforms.CenterCrop(800))
        # transform.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
        # transform.append(transforms.Grayscale(num_output_channels=1))
        # transform.append(transforms.RandomRotation(30))

    else:
        transform.append(transforms.CenterCrop(800))
        # transform.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
        # transform.append(transforms.Grayscale(num_output_channels=1))
        # transform.append(transforms.RandomRotation(20))
        
    transform.append(transforms.Resize(400))
    transform.append(transforms.ToTensor())
    # transform.append(Cutout(n_holes=1, length=112))
    transform.append(normalize)
    return transforms.Compose(transform)



class CustomDataset(data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = dict([l.replace('\n', '').split(' ') for l in open(annotations_file).readlines()])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_origin_path = list(self.img_labels.keys())[idx]
        img_path_S1 = os.path.join(self.img_dir, img_origin_path) + '-S01.jpg'
        img_path_M1 = os.path.join(self.img_dir, img_origin_path) + '-M01.jpg'
        img_path_E1 = os.path.join(self.img_dir, img_origin_path) + '-E01.jpg'
        image_id = img_path_S1.split('-')[0].split('/')[-1]
        image_S1 = Image.open(img_path_S1)
        image_M1 = Image.open(img_path_M1)
        image_E1 = Image.open(img_path_E1)
        label = self.img_labels[img_origin_path]

        if self.transform:
            image_S1 = self.transform(image_S1)
            image_M1 = self.transform(image_M1)
            image_E1 = self.transform(image_E1)
            image = torch.cat((image_S1, image_M1, image_E1))
        label = int(label)


        return image_id, image, label




class TestDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.image_dir = root.replace('label','data')
        self.image_list = sorted(list(set(map(lambda x: x.split('-')[0], os.listdir(root)))))
        self.transform = transform
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_id  = self.image_list[idx]
        img_path_S1 = os.path.join(self.image_dir, img_id) + '-S01.jpg'
        img_path_M1 = os.path.join(self.image_dir, img_id) + '-M01.jpg'
        img_path_E1 = os.path.join(self.image_dir, img_id) + '-E01.jpg'
        image_S1 = Image.open(img_path_S1)
        image_M1 = Image.open(img_path_M1)
        image_E1 = Image.open(img_path_E1)

        image_S1 = self.transform(image_S1)
        image_M1 = self.transform(image_M1)
        image_E1 = self.transform(image_E1)
        image = torch.cat((image_S1, image_M1, image_E1))

        return img_id, image

def test_data_loader(root, phase='train', batch_size=128):
    if phase == 'train':
        is_train = False
    elif phase == 'test':
        is_train = False
    else:
        raise KeyError

    dataset = TestDataset(
        root.replace('label','data'),
        transform=get_transform(random_crop=is_train)
    )
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=is_train)


def data_loader_with_split(root, train_split=0.95, batch_size=128, val_label_file='./val_label'):
    if root[-1]=='/':
        mode = root.split('/')[-2]
    else:
        mode = root.split('/')[-1]
    dataset = CustomDataset(
        os.path.join(root, mode+'_label'),
        os.path.join(root, mode+'_data'),
        transform=get_transform(
        random_crop=True)
    )
    split_size = int(len(dataset) * train_split)
    train_set, valid_set = data.random_split(dataset, [split_size, len(dataset) - split_size])
    tr_loader = data.DataLoader(dataset=train_set,
                                batch_size=batch_size,
                                shuffle=True)
    val_loader = data.DataLoader(dataset=valid_set,
                                 batch_size=batch_size,
                                 shuffle=False)



    print('generate val labels')
    gt_labels = {valid_set[idx][0]: valid_set[idx][2] for idx in tqdm(range(len(valid_set)))}
    gt_labels_string = [' '.join([str(s) for s in l]) for l in tqdm(list(gt_labels.items()))]
    with open(val_label_file, 'w') as file_writer:
        file_writer.write("\n".join(gt_labels_string))


    return tr_loader, val_loader, val_label_file
