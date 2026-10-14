import os
import sys
import json
import pickle
import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt
from pypinyin import pinyin, Style

from thop import profile
from thop import clever_format

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
import seaborn as sns


def save_training_data(train_losses, val_losses, train_accuracies, val_accuracies, learning_rates, folder_path):
    np.savez(folder_path,
             train_losses=train_losses,
             val_losses=val_losses,
             train_accuracies=train_accuracies,
             val_accuracies=val_accuracies,
             learning_rates=learning_rates)

def count_parameters(model):
    def bytes_to_human_readable(size):
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_idx = 0
        while size >= 1000 and unit_idx < len(units) - 1:
            size /= 1000.0
            unit_idx += 1
        size = round(size, 2)
        return f"{size} {units[unit_idx]}"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params_str = bytes_to_human_readable(total_params)
    trainable_params_str = bytes_to_human_readable(trainable_params)

    return total_params_str, trainable_params_str

def flops_params(model, dummy_input):
    flops, params = profile(model, (dummy_input))
    flops, params = clever_format([flops, params], "%.2f")
    print('flops: ', flops, 'params: ', params)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def read_data(scr_data_path: str, format: tuple = ('.png', '.PNG')):
    assert os.path.exists(scr_data_path), f"dataset root: {scr_data_path} does not exist."

    flower_class_with_pinyin = [(os.path.join(scr_data_path, cla), pinyin(cla, style=Style.TONE2)) for cla in
                                os.listdir(scr_data_path) if os.path.isdir(os.path.join(scr_data_path, cla))]
    flower_class_with_pinyin.sort(key=lambda x: x[1])
    flower_class = [os.path.basename(cla) for cla, _ in flower_class_with_pinyin]
    processed_class = [name.split('_')[-1] for name in flower_class]
    class_indices = dict((k, v) for v, k in enumerate(processed_class))
    images_path = []
    images_label = []
    supported = format
    for cla in flower_class:
        cla_path = os.path.join(scr_data_path, cla)
        images = [os.path.join(cla_path, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cla.split('_')[-1]]

        for img_path in images:
            images_path.append(img_path)
            images_label.append(image_class)

    total_samples = len(images_path)
    print(f"共包含 {total_samples} 个文件。")

    return images_path, images_label

def read_split_data_science(scr_data_path: str, json_name: str, test_scale: float = 0.2, val_scale: float = 0.2, format: tuple = ('.png', '.PNG')):
    set_seed(0)

    assert os.path.exists(scr_data_path), f"dataset root: {scr_data_path} does not exist."
    flower_class_with_pinyin = [(os.path.join(scr_data_path, cla), pinyin(cla, style=Style.TONE2)) for cla in
                                os.listdir(scr_data_path) if os.path.isdir(os.path.join(scr_data_path, cla))]
    flower_class_with_pinyin.sort(key=lambda x: x[1])
    flower_class = [os.path.basename(cla) for cla, _ in flower_class_with_pinyin]
    processed_class = [name.split('_')[0] for name in flower_class]
    class_indices = dict((k, v) for v, k in enumerate(processed_class))

    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4, ensure_ascii=False)
    with open(json_name, 'w', encoding='utf-8') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    test_images_path = []
    test_images_label = []
    every_class_num = []

    supported = format

    for cla in flower_class:
        cla_path = os.path.join(scr_data_path, cla)
        images = [os.path.join(cla_path, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cla.split('_')[0]]
        every_class_num.append(len(images))

        test_count = int(len(images) * test_scale)
        val_count = int(len(images) * val_scale)

        test_paths = random.sample(images, k=test_count)
        val_paths = random.sample([img for img in images if img not in test_paths], k=val_count)

        for img_path in images:
            if img_path in test_paths:
                test_images_path.append(img_path)
                test_images_label.append(image_class)
            elif img_path in val_paths:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    total_samples = sum(every_class_num)
    train_count = total_samples - len(val_images_path) - len(test_images_path)
    print(f"共包含 {total_samples} 个文件。\n{train_count} 个文件划入训练集, "
          f"{len(val_images_path)} 个文件划入验证集, "
          f"{len(test_images_path)} 个文件划入测试集.")

    return train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = '../ConvNeXt-V2/class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

class FocalLoss(nn.Module):
    def __init__(self, num_labels, activation_type='softmax', gamma=2.0, alpha=0.25, epsilon=1.e-9, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing
        self.activation_type = activation_type

    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            smooth_positives = 1.0 - self.label_smoothing
            smooth_negatives = self.label_smoothing / self.num_labels
            one_hot_key = torch.zeros_like(inputs).to(inputs.device)
            one_hot_key.scatter_(1, targets.view(-1, 1), smooth_positives)
            one_hot_key += smooth_negatives

        if self.activation_type == 'softmax':
            logits = torch.softmax(inputs, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = targets
            logits = torch.sigmoid(inputs)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()

        return loss.mean()

def train_one_epoch(model, optimizer, data_loader, device, epoch, epochs, lr_scheduler):
    model.train()

    loss_function = torch.nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, desc="Training")
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}/{}] loss: {:.4f}, acc: {:.4f}, lr: {:.5f}".format(
            epoch,
            epochs,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, epochs):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, desc="Validating")
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}/{}] loss: {:.4f}, acc: {:.4f}".format(
            epoch,
            epochs,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def get_model_files(model_dir):
    files = os.listdir(model_dir)
    model_files = [os.path.join(model_dir, f) for f in files if f.endswith('.pth')]
    return model_files

def evaluate_model(model, device, test_loader, class_indict_path, confusion_matrix_path, classification_report_path):
    with open(class_indict_path, "r", encoding="utf-8") as f:
        class_indict = json.load(f)

    model.to(device)
    model.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            outputs = torch.squeeze(model(inputs))
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(0)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            total_correct += (predictions == labels.to(device)).sum().item()
            total_samples += len(labels)
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    accuracy = total_correct / total_samples
    print(f"测试集上的整体准确率: {accuracy * 100:.2f}%")

    if len(set(all_true_labels)) > 1:
        class_names = list(class_indict.values())

        matplotlib.rcParams['font.family'] = 'SimHei'
        matplotlib.rcParams['axes.unicode_minus'] = False
        existing_labels = list(set(all_true_labels))
        report = classification_report(
            all_true_labels,
            all_predictions,
            target_names=[class_indict[str(i)] for i in existing_labels],
            labels=existing_labels,
            digits=4,
            zero_division=0
        )

        with open(classification_report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print("分类报告：\n", report)

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())