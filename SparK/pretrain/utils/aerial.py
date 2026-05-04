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
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


class AerialExtreMatchDataset(Dataset):
    """
    Dataset for flat geo-localization data without train/test splits.

    Expected folder structure:
        <root>/
          drone/
            {Place1}/
              image1.jpg
              image2.jpg
              ...
            {Place2}/
              ...
          satellite/
            {Place1}/
              image1.jpg
              ...
            {Place2}/
              ...

    This dataset is self-supervised (no labels returned), matching the
    interface of ImageNetDataset used in pre-training pipelines.
    """

    VALID_VIEWS   = ('drone', 'satellite')
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    def __init__(
        self,
        root_folder: str,
        transform: Callable,
        view: str = 'drone',                          # 'drone' | 'satellite'
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        """
        Args:
            root_folder:   root path that contains `drone/` and `satellite/` dirs.
                           e.g. "/data/AerialExtreMatchCustom"
            transform:     image transform applied in __getitem__.
            view:          which camera view to load ('drone' or 'satellite').
            is_valid_file: optional callable to filter file paths.
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

        # self-supervised pre-training → no labels
        self.targets = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _collect_images(
        self,
        root: str,
        is_valid_file: Optional[Callable[[str], bool]],
    ) -> List[str]:
        """Walk root/{Place}/ sub-folders and collect image file paths."""
        paths: List[str] = []
        for place_id in sorted(os.listdir(root)):
            place_dir = os.path.join(root, place_id)
            if not os.path.isdir(place_dir):
                continue
            for fname in sorted(os.listdir(place_dir)):
                fpath = os.path.join(place_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                if is_valid_file is not None:
                    if is_valid_file(fpath):
                        paths.append(fpath)
                else:
                    if fname.lower().endswith(self.IMG_EXTENSIONS):
                        paths.append(fpath)
        return paths

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        img_file_path = self.samples[index]
        return self.transform(self.loader(img_file_path))


def build_dataset_to_pretrain(dataset_path: str, input_size: int, two_view: bool = True) -> Dataset:
    """
    Build pre-training dataset from the flat AerialExtreMatch folder structure.

    Loads the **drone** view by default. Set `two_view=True` to concatenate
    both drone and satellite views for richer pre-training data.

    Folder expected at `dataset_path`:
        AerialExtreMatchCustom/
          drone/     {Place}/*.jpg
          satellite/ {Place}/*.jpg

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

    ds_drone = AerialExtreMatchDataset(
        root_folder=dataset_path,
        transform=trans_train,
        view='drone',
    )

    if two_view:
        ds_sat        = AerialExtreMatchDataset(dataset_path, transform=trans_train, view='satellite')
        dataset_train = ConcatDataset([ds_drone, ds_sat])
    else:
        dataset_train = ds_drone

    print_transform(trans_train, '[pre-train]')
    print(f'AerialExtreMatchDataset: {len(dataset_train)} images loaded from {dataset_path}')
    return dataset_train


def print_transform(transform: transforms.Compose, s: str) -> None:
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')