import os
import json
import random
import numpy as np
import torch
from pypinyin import pinyin, Style


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_split_data_science(scr_data_path: str, json_name: str, test_scale: float = 0.2,
                            val_scale: float = 0.2, format: tuple = ('.png', '.PNG'), seed: int = 0):
    set_seed(seed)
    assert os.path.exists(scr_data_path), f"dataset root: {scr_data_path} does not exist."

    # 类别文件夹按拼音排序，保证类别顺序确定（跨特征目录一致）
    flower_class_with_pinyin = [(os.path.join(scr_data_path, cla), pinyin(cla, style=Style.TONE2)) for cla in
                                os.listdir(scr_data_path) if os.path.isdir(os.path.join(scr_data_path, cla))]
    flower_class_with_pinyin.sort(key=lambda x: x[1])
    flower_class = [os.path.basename(cla) for cla, _ in flower_class_with_pinyin]
    processed_class = [name.split('_')[0] for name in flower_class]
    class_indices = dict((k, v) for v, k in enumerate(processed_class))

    dup_classes = {name for name in processed_class if processed_class.count(name) > 1}
    if dup_classes:
        print(f"[警告] 检测到重复的类别前缀 {dup_classes}，这些文件夹将被视为同一类别，请确认是否符合预期")

    supported = format

    if os.path.exists(json_name):
        try:
            with open(json_name, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            if 'train_images' in cache:
                def to_abs(paths):
                    return [os.path.join(scr_data_path, p) for p in paths]

                train_images_path = to_abs(cache['train_images'])
                val_images_path = to_abs(cache['val_images'])
                test_images_path = to_abs(cache['test_images'])

                # 校验缓存中的文件当前是否都存在（数据目录变动则重新划分）
                if all(os.path.exists(p) for p in train_images_path + val_images_path + test_images_path):
                    train_images_label = cache['train_labels']
                    val_images_label = cache['val_labels']
                    test_images_label = cache['test_labels']
                    print(f"使用现有划分文件: {json_name} "
                          f"(训练{len(train_images_path)}/验证{len(val_images_path)}/测试{len(test_images_path)})")
                    return (train_images_path, train_images_label,
                            val_images_path, val_images_label,
                            test_images_path, test_images_label)
                print(f"[提示] {json_name} 中记录的文件在当前数据目录中已不存在（数据可能已变动），将重新划分。")
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"[提示] {json_name} 格式异常（可能是旧版类别映射文件），将重新划分。")

    train_images_path, train_images_label = [], []
    val_images_path, val_images_label = [], []
    test_images_path, test_images_label = [], []
    every_class_num = []

    for cla in flower_class:
        cla_path = os.path.join(scr_data_path, cla)
        images = [os.path.join(cla_path, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()  # 关键：显式排序，保证多特征目录按相同顺序取样

        image_class = class_indices[cla.split('_')[0]]
        every_class_num.append(len(images))

        test_count = int(len(images) * test_scale)
        val_count = int(len(images) * val_scale)
        if len(images) >= 5:  # 样本足够时保证验证/测试集至少各 1 张
            test_count = max(1, test_count)
            val_count = max(1, val_count)

        test_paths = random.sample(images, k=test_count)
        val_candidates = [img for img in images if img not in set(test_paths)]
        val_paths = random.sample(val_candidates, k=val_count)
        test_set, val_set = set(test_paths), set(val_paths)

        for img_path in images:
            if img_path in test_set:
                test_images_path.append(img_path)
                test_images_label.append(image_class)
            elif img_path in val_set:
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

    def to_rel(paths):
        return [os.path.relpath(p, scr_data_path).replace(os.sep, '/') for p in paths]

    cache_data = {
        'seed': seed,
        'class_indices': dict((str(val), key) for key, val in class_indices.items()),
        'train_images': to_rel(train_images_path), 'train_labels': train_images_label,
        'val_images': to_rel(val_images_path), 'val_labels': val_images_label,
        'test_images': to_rel(test_images_path), 'test_labels': test_images_label,
    }
    with open(json_name, 'w', encoding='utf-8') as json_file:
        json.dump(cache_data, json_file, indent=2, ensure_ascii=False)
    print(f"数据划分已保存至: {json_name}")

    return (train_images_path, train_images_label,
            val_images_path, val_images_label,
            test_images_path, test_images_label)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train
