from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os




def get_dataloaders(data_dir, img_size, batch_size, augment=False):
    base = os.path.join(data_dir, 'split')


    train_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
    transforms.ToTensor()
    ])


    test_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
    ])


    train_ds = datasets.ImageFolder(os.path.join(base, 'train'), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(base, 'val'), test_tf)
    test_ds = datasets.ImageFolder(os.path.join(base, 'test'), test_tf)


    return (
        DataLoader(train_ds, batch_size, shuffle=True),
        DataLoader(val_ds, batch_size),
        DataLoader(test_ds, batch_size)
    )