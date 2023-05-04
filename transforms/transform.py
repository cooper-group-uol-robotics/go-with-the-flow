from typing import Callable

import torchvision.transforms as transforms


def create_loading_transform(img_x: int, img_y: int) -> Callable:
    """ Return data loading transform """
    return transforms.Compose([transforms.Resize([img_x, img_y]),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5], std=[0.5])])
