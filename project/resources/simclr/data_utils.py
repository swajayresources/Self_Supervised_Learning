from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import random
import torch

def compute_train_transform(seed=123456):
    """
    This function returns a composition of data augmentations to a single training image.
    Complete the following lines. Hint: look at available functions in torchvision.transforms
    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    
    # Transformation that applies color jitter with brightness=0.4, contrast=0.4, saturation=0.4, and hue=0.1
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
    
    train_transform = transforms.Compose([
        
        transforms.RandomResizedCrop(32),
        
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return train_transform
    
def compute_test_transform():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return test_transform


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        x_i = None
        x_j = None

        if self.transform is not None:
            

            if self.target_transform is not None:
                target = self.target_transform(target)

        return x_i, x_j, target