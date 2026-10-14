import os
import json
import random
import numpy as np
import torch
from pypinyin import pinyin, Style
from torchvision import datasets
from torch.utils.data import Dataset
import torch.nn as nn

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def read_split_data(data_root: str,
                    test_scale: float = 0.2,
                    val_scale: float = 0.2,
                    suffix: tuple = ('.png', '.PNG', '.jpg', '.JPG')):
    set_seed(0)

    if not os.path.isdir(data_root):
        raise ValueError(f"数据目录不存在: {data_root}")

    cls_dirs = [d for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d))]
    cls_dirs.sort(key=lambda x: pinyin(x, style=Style.TONE2))

    class_names = [name.split('_')[-1] for name in cls_dirs]
    class_indices = {v: k for k, v in enumerate(class_names)}

    with open('class_indices.json', 'w', encoding='utf-8') as f:
        json.dump({str(v): k for k, v in class_indices.items()},
                  f, ensure_ascii=False, indent=4)

    train_data, val_data, test_data = [], [], []

    for cls_dir in cls_dirs:
        cls_path = os.path.join(data_root, cls_dir)

        imgs = [os.path.join(cls_path, im) for im in os.listdir(cls_path)
                if os.path.splitext(im)[-1] in suffix]

        label = class_indices[class_names[cls_dirs.index(cls_dir)]]

        n_test = max(int(len(imgs) * test_scale), 1)
        n_val = max(int(len(imgs) * val_scale), 1)
        random.shuffle(imgs)

        test_data.extend([(img, label) for img in imgs[:n_test]])
        val_data.extend([(img, label) for img in imgs[n_test:n_test + n_val]])
        train_data.extend([(img, label) for img in imgs[n_test + n_val:]])

    train_imgs, train_labs = zip(*train_data) if train_data else ([], [])
    val_imgs, val_labs = zip(*val_data) if val_data else ([], [])
    test_imgs, test_labs = zip(*test_data) if test_data else ([], [])

    print(f"数据集统计【{os.path.basename(data_root)}】: "
          f"训练集: {len(train_imgs)}, "
          f"验证集: {len(val_imgs)}, "
          f"测试集: {len(test_imgs)}")

    return (list(train_imgs), list(train_labs),
            list(val_imgs), list(val_labs),
            list(test_imgs), list(test_labs))


class MultiFeatureDataset(Dataset):

    def __init__(self, img_paths, img_labels, transform=None):
        self.paths = img_paths
        self.labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = datasets.folder.default_loader(self.paths[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        imgs, labs = zip(*batch)
        return torch.stack(imgs, 0), torch.tensor(labs, dtype=torch.long)


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup: bool = True,
                        warmup_epochs: int = 1,
                        warmup_factor: float = 1e-3,
                        end_factor: float = 1e-6):

    if not warmup:
        warmup_epochs = 0

    def lr_lambda(step):
        if warmup and step <= warmup_epochs * num_step:
            alpha = step / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            cur_step = step - warmup_epochs * num_step
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + np.cos(np.pi * cur_step / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_params_groups(model: nn.Module, weight_decay: float = 1e-4):

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.}
    ]

class AddFusionModel(nn.Module):

    def __init__(self, num_classes=20, feature_dim=2048):
        super().__init__()
        from model import resnet50
        self.backbone = resnet50(num_classes=num_classes, include_top=False)
        self.fusion_params = nn.Parameter(torch.ones(3)/3.0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def extract_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


    def forward(self, x_c, x_m, x_s):
        f_c = self.extract_features(x_c)
        f_m = self.extract_features(x_m)
        f_s = self.extract_features(x_s)
        weights = torch.softmax(self.fusion_params, dim=0)
        fused_features = weights[0] * f_c + weights[1] * f_m + weights[2] * f_s
        return self.classifier(fused_features)

    def get_fusion_weights(self):
        weights = torch.softmax(self.fusion_params, dim=0)
        return {
            'alpha': weights[0].item(),
            'beta': weights[1].item(),
            'gamma': weights[2].item()
        }
