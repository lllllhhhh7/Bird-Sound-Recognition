
import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from pypinyin import pinyin, Style

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def plot_roc_curve(true_labels, pred_probs, save_path, class_names):
    plt.figure(figsize=(8, 6))
    n_classes = len(class_names)
    fpr, tpr, roc_auc = dict(), dict(), dict()
    true_onehot = np.eye(n_classes)[true_labels]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_onehot[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return roc_auc

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train