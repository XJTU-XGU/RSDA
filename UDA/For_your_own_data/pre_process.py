from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

def image_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size,resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
    return transforms.Compose([
    transforms.Resize((resize_size,resize_size)),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    normalize
    ])

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'
                 ,second=False):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("No image found !"))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.second=second
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        if not self.second:
            path, target = self.imgs[index]
            path = os.path.join("root to your data",path)
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target
        else:
            path, label, weight, sigma = self.imgs[index]
            path = os.path.join("root to your data",path)
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)

            return img, label, weight, sigma



    def __len__(self):
        return len(self.imgs)

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0],int(val.split()[1]),float(val.split()[2]),float(val.split()[3])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')