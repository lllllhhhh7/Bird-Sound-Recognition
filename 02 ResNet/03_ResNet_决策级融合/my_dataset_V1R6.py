from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import random
import torchaudio.transforms as T
import numpy as np


class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


class RandomFrequencyMasking(T.FrequencyMasking):
    def __init__(self, freq_mask_param=10, min_masks=0, max_masks=3, apply_prob=0.8):
        super().__init__(freq_mask_param=freq_mask_param)
        self.min_masks = min_masks
        self.max_masks = max_masks
        self.apply_prob = apply_prob

    def forward(self, spectrogram):
        if random.random() < self.apply_prob:
            num_masks = random.randint(self.min_masks, self.max_masks)
            for _ in range(num_masks):
                spectrogram = super().forward(spectrogram)
        return spectrogram


class RandomTimeMasking(T.TimeMasking):
    def __init__(self, time_mask_param=10, min_masks=1, max_masks=3, apply_prob=0.8):
        super().__init__(time_mask_param=time_mask_param)
        self.min_masks = min_masks
        self.max_masks = max_masks
        self.apply_prob = apply_prob

    def forward(self, spectrogram):
        if random.random() < self.apply_prob:
            num_masks = random.randint(self.min_masks, self.max_masks)
            for _ in range(num_masks):
                spectrogram = super().forward(spectrogram)
        return spectrogram


def make_weights_for_balanced_classes(dataset, nclasses):
    count = [0] * nclasses
    for item in dataset:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(dataset)
    for idx, val in enumerate(dataset):
        weight[idx] = weight_per_class[val[1]]
    return weight

class CustomWeightedRandomSampler(WeightedRandomSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(
            range(0, len(self.weights)),
            size=self.num_samples,
            p=self.weights.numpy() / torch.sum(self.weights).numpy(),
            replace=self.replacement
        )
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())
