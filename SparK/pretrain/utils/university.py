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


class UniversityDataset(Dataset):
    """
    Dataset for University-1652 style geo-localization data.

    Supported split/view combinations:
        train:  drone      -> university/train/drone/<id>/*.jpeg
        train:  satellite  -> university/train/satellite/<id>/*.jpg
        test:   drone      -> university/test/query_drone/<id>/*.jpeg
        test:   satellite  -> university/test/gallery_satellite/<id>/*.jpg

    This dataset is self-supervised (no labels returned), matching the
    interface of ImageNetDataset used in pre-training pipelines.
    """

    # Map (train, view) -> sub-folder name inside split dir
    _VIEW_FOLDER = {
        (True,  'drone'):     'drone',
        (True,  'satellite'): 'satellite',
        (False, 'drone'):     'query_drone',
        (False, 'satellite'): 'gallery_satellite',
    }

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    def __init__(
        self,
        university_folder: str,
        train: bool,
        transform: Callable,
        view: str = 'drone',                          # 'drone' | 'satellite'
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        """
        Args:
            university_folder: root path that contains `train/` and `test/` dirs.
            train:             True → training split, False → test split.
            transform:         image transform applied in __getitem__.
            view:              which camera view to load ('drone' or 'satellite').
            is_valid_file:     optional callable to filter file paths.
        """
        key = (train, view)
        if key not in self._VIEW_FOLDER:
            raise ValueError(
                f"Unsupported (train={train}, view='{view}'). "
                f"Choose view from {{'drone', 'satellite'}}."
            )

        split_dir = 'train' if train else 'test'
        view_dir  = self._VIEW_FOLDER[key]
        self.root = os.path.join(os.path.abspath(university_folder), split_dir, view_dir)

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")

        self.transform    = transform
        self.loader       = pil_loader
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
        """Walk root/<id>/ sub-folders and collect image file paths."""
        paths: List[str] = []
        for class_id in sorted(os.listdir(root)):
            class_dir = os.path.join(root, class_id)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                fpath = os.path.join(class_dir, fname)
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


def build_dataset_to_pretrain(dataset_path: str, input_size: int, two_view: bool = False) -> Dataset:
    """
    Drop-in replacement for the ImageNet version of build_dataset_to_pretrain.

    Loads the **drone** view of the **training** split by default, which gives
    the largest number of images and multiple shots per building — ideal for
    self-supervised pre-training.

    You can change `view` to 'satellite' if you want satellite imagery instead,
    or load both views by calling this function twice and concatenating with
    torch.utils.data.ConcatDataset.

    :param dataset_path: path to the university root folder (contains train/ & test/)
    :param input_size:   image resolution fed to the model
    :return:             dataset used for pre-training
    """
    trans_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    # Strip trailing 'train' or 'val' suffix the same way imagenet.py does
    dataset_path = os.path.abspath(dataset_path)
    for postfix in ('train', 'val'):
        if dataset_path.endswith(postfix):
            dataset_path = dataset_path[:-len(postfix)]

    ds_drone = UniversityDataset(
        university_folder=dataset_path,
        transform=trans_train,
        train=True,
        view='drone',   # change to 'satellite' or use ConcatDataset for both
    )
    if two_view == True:
        ds_sat   = UniversityDataset(dataset_path, train=True, transform=trans_train, view='satellite')
        dataset_train  = ConcatDataset([ds_drone, ds_sat])
    else:
        dataset_train = ds_drone
    print_transform(trans_train, '[pre-train]')
    print(f'UniversityDataset: {len(dataset_train)} images loaded from {dataset_path}')
    return dataset_train


def print_transform(transform: transforms.Compose, s: str) -> None:
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')