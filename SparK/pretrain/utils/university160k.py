# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Any, Callable, List, Optional, Tuple
import PIL.Image as PImage
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import transforms
from torch.utils.data import Dataset, ConcatDataset

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC


def pil_loader(path: str) -> PImage.Image:
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


class University160kDataset(Dataset):
    """
    Dataset for University-160k geo-localization data.

    Expected folder structure (images sit directly inside each view folder):
        <root>/
          drone/
            image_0001.jpeg
            image_0002.jpeg
            ...
          satellite/
            image_0001.jpeg
            image_0002.jpeg
            ...

    This dataset is self-supervised (no labels returned), matching the
    interface of ImageNetDataset used in pre-training pipelines.
    """

    VALID_VIEWS    = ('drone', 'satellite')
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    def __init__(
        self,
        root_folder: str,
        transform: Callable,
        view: str = 'drone',
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        """
        Args:
            root_folder:   root path that contains drone/ and satellite/ dirs.
                           e.g. "/data/University160k"
            transform:     image transform applied in __getitem__.
            view:          which camera view to load ('drone' or 'satellite').
            is_valid_file: optional callable to further filter file paths.
        """
        if view not in self.VALID_VIEWS:
            raise ValueError(
                f"Unsupported view='{view}'. Choose from {self.VALID_VIEWS}."
            )

        self.root = os.path.join(os.path.abspath(root_folder), view)

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")

        self.transform = transform
        self.loader    = pil_loader
        self.samples: Tuple[str, ...] = tuple(
            self._collect_images(self.root, is_valid_file)
        )

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {self.root}")

        self.targets = None  # self-supervised pre-training, no labels

    def _collect_images(
        self,
        root: str,
        is_valid_file: Optional[Callable[[str], bool]],
    ) -> List[str]:
        """Collect image file paths directly from root/ (no sub-folders)."""
        paths: List[str] = []
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                continue
            if is_valid_file is not None:
                if is_valid_file(fpath):
                    paths.append(fpath)
            else:
                if fname.lower().endswith(self.IMG_EXTENSIONS):
                    paths.append(fpath)
        return paths

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        return self.transform(self.loader(self.samples[index]))


def build_dataset_to_pretrain(dataset_path: str, input_size: int, two_view: bool = True) -> Dataset:
    """
    Build pre-training dataset from the University-160k folder structure.

    Loads the drone view by default. Set two_view=True to concatenate
    both drone and satellite views for richer pre-training data.

    Folder expected at dataset_path:
        University160k/
          drone/      *.jpeg
          satellite/  *.jpeg

    :param dataset_path: path to the root folder (contains drone/ & satellite/)
    :param input_size:   image resolution fed to the model
    :param two_view:     if True, concatenate drone + satellite datasets
    :return:             dataset used for pre-training
    """
    trans_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    dataset_path = os.path.abspath(dataset_path)

    ds_drone = University160kDataset(
        root_folder=dataset_path,
        transform=trans_train,
        view='drone',
    )

    if two_view:
        ds_sat        = University160kDataset(dataset_path, transform=trans_train, view='satellite')
        dataset_train = ConcatDataset([ds_drone, ds_sat])
    else:
        dataset_train = ds_drone

    print_transform(trans_train, '[pre-train]')
    print(f'University160kDataset: {len(dataset_train)} images loaded from {dataset_path}')
    return dataset_train


def print_transform(transform: transforms.Compose, s: str) -> None:
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')