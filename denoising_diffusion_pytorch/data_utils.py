import torch
import os
import torchvision.transforms as transforms

from torchvision import datasets
from torch.utils.data import Subset
from PIL import Image
from multiprocessing import cpu_count


def get_data(args, indices=None):
    # define transforms
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),  # CIFAR-10
    ])

    resize = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.RandomResizedCrop(vit_image_size, scale=global_crops_scale,
        #                              interpolation=transforms.InterpolationMode.BICUBIC),
    ])

    # transformation for the local small crops
    transform = transforms.Compose([
        normalize,
        resize,
    ])

    if os.path.isdir("/alto/shared/DataSets/cifar-10-batches-py"):
        root = "/alto/shared/DataSets"
        download = False
    else:
        root = "./data"
        download = True
    dataset = CIFAR10_ID(root=root,
                         args=args,
                         train=True,
                         download=download,
                         transform=transform)
    if indices is not None:
        dataset = Subset(dataset, indices)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size_per_gpu,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=cpu_count(),
    )
    return dataset, data_loader


class CIFAR10_ID(datasets.CIFAR10):
    """Override getitem function to return the index instead of the label"""
    def __init__(self, root, args, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.args = args

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, index) where index is index of the image.
        """
        # img = self.data[index]
        img = self.data[index]  # Always return image 0

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, index

    # def __len__(self):
    #     return self.args.batch_size_per_gpu * self.args.num_gpus * 30