
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from itertools import product
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms, models as torchvision_models
import json
import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

from my_dataset_V1R6 import MyDataSet
from utils_MultiModelDecisionFusion import read_split_data_science
from convnextv2 import convnextv2_tiny, convnextv2_base, convnextv2_large, convnextv2_huge
from swin_transformer import swin_base_patch4_window12_384

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SimpleConvNeXtV2Model(nn.Module):
    def __init__(self, backbone_type='convnextv2_tiny', num_classes=20):
        super().__init__()

        if backbone_type == 'convnextv2_tiny':
            self.backbone = convnextv2_tiny(num_classes=num_classes)
        elif backbone_type == 'convnextv2_base':
            self.backbone = convnextv2_base(num_classes=num_classes)
        elif backbone_type == 'convnextv2_large':
            self.backbone = convnextv2_large(num_classes=num_classes)
        elif backbone_type == 'convnextv2_huge':
            self.backbone = convnextv2_huge(num_classes=num_classes)
        else:
            raise ValueError(f"不支持的backbone类型: {backbone_type}")

        print(f"创建 ConvNeXtV2 {backbone_type} 模型，类别数: {num_classes}")

    def forward(self, x):
        out = self.backbone(x)
        if isinstance(out, tuple):
            return out[0]
        return out


class SimpleResNet50Model(nn.Module):
    def __init__(self, num_classes=20, pretrained=False):
        super().__init__()
        try:
            weights = torchvision_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = torchvision_models.resnet50(weights=weights)
        except AttributeError:
            self.backbone = torchvision_models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        print(f"创建 ResNet50 模型，类别数: {num_classes}, 预训练: {pretrained}")

    def forward(self, x):
        return self.backbone(x)



class SimpleSwinTransformerModel(nn.Module):
    def __init__(self, num_classes=20, pretrained=False):
        super().__init__()
        self.backbone = swin_base_patch4_window12_384(num_classes=num_classes)
        print(f"创建Swin Transformer (swin_base_patch4_window12_384) 模型，类别数: {num_classes}")

    def forward(self, x):
        return self.backbone(x)


class FusionStrategy:
    @staticmethod
    def weighted_average(probs_list, weights=None):
        if weights is None:
            weights = [1. / len(probs_list)] * len(probs_list)
        fused = torch.zeros_like(probs_list[0])
        for p, w in zip(probs_list, weights):
            fused += p * w
        return fused

    @staticmethod
    def ds_evidence_theory(probs_list, uncertainty=0.1):
        def prob2bpa(prob):
            bpa = prob * (1 - uncertainty)
            bpa = torch.cat([bpa, torch.full((prob.size(0), 1), uncertainty, device=prob.device)], dim=1)
            return bpa

        def combine(m1, m2):
            C = m1.shape[1] - 1
            eye = torch.eye(C, device=m1.device)
            k = torch.sum(m1[:, :-1].unsqueeze(2) * m2[:, :-1].unsqueeze(1) * (1 - eye).unsqueeze(0), dim=(1, 2))
            mask = k >= 1.0
            m_comb = torch.zeros_like(m1)
            if mask.any():
                m_comb[mask] = (m1[mask] + m2[mask]) / 2.0
            m_norm = (~mask)
            if m_norm.any():
                m1n, m2n, kn = m1[m_norm], m2[m_norm], k[m_norm].unsqueeze(1)
                cls_agree = m1n[:, :-1] * m2n[:, :-1]
                m_comb[m_norm] = torch.cat([cls_agree, kn], dim=1) / (1 - kn)
            return m_comb

        bpas = [prob2bpa(p) for p in probs_list]
        fused = bpas[0]
        for b in bpas[1:]:
            fused = combine(fused, b)
        return fused[:, :-1]

    @staticmethod
    def majority_voting(probs_list):
        preds_list = [torch.argmax(p, dim=1) for p in probs_list]
        batch_size = preds_list[0].shape[0]
        num_classes = probs_list[0].shape[1]
        num_models = len(probs_list)

        fused_probs = torch.zeros_like(probs_list[0])

        for i in range(batch_size):
            votes = torch.zeros(num_classes)
            for model_pred in preds_list:
                votes[model_pred[i]] += 1
            fused_probs[i] = votes / num_models

        return fused_probs


def format_classification_report(report_dict, class_names, title="分类报告"):
    max_class_name_len = max([len(name) for name in class_names] + [20])

    lines = [f"{title}：", ""]
    header = f"{'':<{max_class_name_len}} {'precision':>12} {'recall':>12} {'f1-score':>12} {'support':>12}"
    lines.append(header)
    lines.append("-" * (max_class_name_len + 12 * 4))
    for i, class_name in enumerate(class_names):
        if str(i) in report_dict:
            metrics = report_dict[str(i)]
            line = f"{class_name:<{max_class_name_len}} {metrics['precision']:>12.4f} {metrics['recall']:>12.4f} {metrics['f1-score']:>12.4f} {metrics['support']:>12.0f}"
            lines.append(line)
    lines.append("")
    if 'accuracy' in report_dict:
        lines.append(f"{'accuracy':<{max_class_name_len}} {'':>12} {'':>12} {'':>12} {report_dict['accuracy']:>12.4f}")
    if 'macro avg' in report_dict:
        metrics = report_dict['macro avg']
        line = f"{'macro avg':<{max_class_name_len}} {metrics['precision']:>12.4f} {metrics['recall']:>12.4f} {metrics['f1-score']:>12.4f} {metrics['support']:>12.0f}"
        lines.append(line)
    if 'weighted avg' in report_dict:
        metrics = report_dict['weighted avg']
        line = f"{'weighted avg':<{max_class_name_len}} {metrics['precision']:>12.4f} {metrics['recall']:>12.4f} {metrics['f1-score']:>12.4f} {metrics['support']:>12.0f}"
        lines.append(line)

    return "\n".join(lines)


def evaluate_combination(models, loaders_list, dataset_type, args, class_names, feature_names,
                         combination_name, combination_info, weights):
    print(f"\n{'=' * 80}")
    print(f"评估组合: {combination_name}")
    print(f"组合详情: {combination_info}")
    print(f"数据集: {dataset_type}")
    print(f"{'=' * 80}")

    all_probs, all_labels = [], None
    individual_results = {}
    for idx, (model, loader, feat) in enumerate(zip(models, loaders_list, feature_names)):
        model.eval()
        probs_buffer, labels_buffer = [], []
        with torch.no_grad():
            for imgs, labs in tqdm(loader, desc=f"{feat} {dataset_type}"):
                imgs = imgs.to(args.device)
                out = model(imgs)
                probs_buffer.append(F.softmax(out, dim=1).cpu())
                labels_buffer.extend(labs.numpy())

        probs = torch.cat(probs_buffer, dim=0)
        preds = probs.argmax(dim=1).numpy()
        labels = np.array(labels_buffer)
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)

        individual_results[feat] = {
            'accuracy': acc,
            'report': report,
            'predictions': preds,
            'probabilities': probs.numpy()
        }

        all_probs.append(probs)
        if all_labels is None:
            all_labels = labels

        print(f"{feat} 单一模型 {dataset_type} 准确率: {acc:.4f}")
    fused_w = FusionStrategy.weighted_average(all_probs, weights)
    fused_ds = FusionStrategy.ds_evidence_theory(all_probs)
    fused_mv = FusionStrategy.majority_voting(all_probs)

    preds_w = fused_w.argmax(dim=1).numpy()
    preds_ds = fused_ds.argmax(dim=1).numpy()
    preds_mv = fused_mv.argmax(dim=1).numpy()
    def calc_metrics(pred):
        acc = accuracy_score(all_labels, pred)
        report = classification_report(all_labels, pred, target_names=class_names, output_dict=True, zero_division=0)
        return acc, report

    acc_w, rep_w = calc_metrics(preds_w)
    acc_ds, rep_ds = calc_metrics(preds_ds)
    acc_mv, rep_mv = calc_metrics(preds_mv)

    return {
        'combination_name': combination_name,
        'combination_info': combination_info,
        'individual': individual_results,
        'weighted_fusion': {'accuracy': acc_w, 'report': rep_w},
        'ds_fusion': {'accuracy': acc_ds, 'report': rep_ds},
        'majority_voting': {'accuracy': acc_mv, 'report': rep_mv}
    }

def inference(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    os.makedirs(args.report_dir, exist_ok=True)

    with open(args.class_indict_path, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
    class_indices = {int(k): v for k, v in class_indices.items()}

    with open('class_indices_en.json', 'r', encoding='utf-8') as f:
        class_names_en = json.load(f)

    sorted_indices = sorted(class_indices.keys())
    class_names = [class_names_en[str(idx)] for idx in sorted_indices]
    print('类别映射：', dict(enumerate(class_names)))

    feature_types = ['Chirplet', 'Mel', 'STFT']
    model_types = ['resnet50', 'convnextv2', 'swin_transformer']

    all_combinations = list(product(model_types, repeat=len(feature_types)))
    print(f"总共有 {len(all_combinations)} 种组合需要评估")

    data_loaders = {}
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    feature_info = [
        ('Chirplet', args.chirplet_data_path),
        ('Mel', args.mel_data_path),
        ('STFT', args.stft_data_path)
    ]

    for feat_name, data_path in feature_info:
        indices_file = f'tgt_{feat_name}_indices.json'

        if not os.path.exists(indices_file):
            print(f"索引文件 {indices_file} 不存在，将重新生成...")
            train_imgs, train_labs, val_imgs, val_labs, test_imgs, test_labs = read_split_data_science(
                data_path, indices_file, val_scale=0.2, test_scale=0.2
            )
        else:
            print(f"使用现有索引文件: {indices_file}")
            train_imgs, train_labs, val_imgs, val_labs, test_imgs, test_labs = read_split_data_science(
                data_path, indices_file, val_scale=0.2, test_scale=0.2
            )

        val_set = MyDataSet(val_imgs, val_labs, transform)
        test_set = MyDataSet(test_imgs, test_labs, transform)
        val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                sampler=SequentialSampler(val_set),
                                num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size,
                                 sampler=SequentialSampler(test_set),
                                 num_workers=args.num_workers, pin_memory=True)

        data_loaders[feat_name] = {
            'val': val_loader,
            'test': test_loader,
            'val_size': len(val_set),
            'test_size': len(test_set)
        }

        print(f'- {feat_name}: 验证集{len(val_set)}，测试集{len(test_set)}')

    weights = None
    if args.weights:
        weights = list(map(float, args.weights.split(',')))
        print('使用自定义权重：', weights)

    all_results = {'val': [], 'test': []}

    for i, combination in enumerate(all_combinations):
        combination_info = [
            (feature_types[0], combination[0]),
            (feature_types[1], combination[1]),
            (feature_types[2], combination[2])
        ]
        combination_name = f"组合{i+1:02d}_{combination[0]}_{combination[1]}_{combination[2]}"

        print(f"\n{'=' * 60}")
        print(f"评估第 {i+1}/{len(all_combinations)} 个组合")
        print(f"组合名称: {combination_name}")
        print(f"组合详情: {combination_info}")
        print(f"{'=' * 60}")

        models = []
        feature_names = []

        for feat_idx, (feat_name, model_type) in enumerate(combination_info):
            if model_type == 'convnextv2':
                model = SimpleConvNeXtV2Model(
                    backbone_type=args.backbone_type,
                    num_classes=args.num_classes
                ).to(device)
                ckpt = os.path.join(args.model_save_path, f'best_convnextv2_{feat_name}.pth')

            elif model_type == 'resnet50':
                model = SimpleResNet50Model(
                    num_classes=args.num_classes,
                    pretrained=False
                ).to(device)
                ckpt = os.path.join(args.model_save_path, f'best_resnet50_{feat_name}.pth')

            elif model_type == 'swin_transformer':
                model = SimpleSwinTransformerModel(
                    num_classes=args.num_classes
                ).to(device)
                ckpt = os.path.join(args.model_save_path, f'best_swin_{feat_name}.pth')

            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

            if not os.path.exists(ckpt):
                alt_ckpts = [
                    os.path.join(args.model_save_path, f'best_{model_type}_{feat_name}.pth'),
                    os.path.join(args.model_save_path, f'best_{feat_name}.pth'),
                    os.path.join(args.model_save_path, f'{model_type}_{feat_name}_best.pth')
                ]

                for alt_ckpt in alt_ckpts:
                    if os.path.exists(alt_ckpt):
                        ckpt = alt_ckpt
                        break
                else:
                    raise FileNotFoundError(f"权重文件不存在：{ckpt}")

            print(f'加载 {feat_name} ({model_type}) 模型权重: {ckpt}')

            try:
                state_dict = torch.load(ckpt, map_location=device)

                if isinstance(state_dict, dict):
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    elif 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']
                    elif 'model' in state_dict:
                        state_dict = state_dict['model']

                if state_dict and list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                if model_type == 'convnextv2':
                    if state_dict and list(state_dict.keys())[0].startswith('backbone.'):
                        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
                    model.backbone.load_state_dict(state_dict, strict=False)
                elif model_type == 'resnet50':
                    model.backbone.load_state_dict(state_dict, strict=False)
                elif model_type == 'swin_transformer':
                    model.backbone.load_state_dict(state_dict, strict=False)

                print(f"成功加载 {feat_name} ({model_type}) 模型权重")

            except Exception as e:
                print(f"加载 {feat_name} ({model_type}) 权重失败: {e}")
                print("跳过此组合...")
                break

            model.eval()
            models.append(model)
            feature_names.append(feat_name)
            print(f'- 已加载 {feat_name} ({model_type}) 模型')
        if len(models) == 3:
            val_loaders_list = [
                data_loaders[feature_types[0]]['val'],
                data_loaders[feature_types[1]]['val'],
                data_loaders[feature_types[2]]['val']
            ]

            test_loaders_list = [
                data_loaders[feature_types[0]]['test'],
                data_loaders[feature_types[1]]['test'],
                data_loaders[feature_types[2]]['test']
            ]
            val_result = evaluate_combination(
                models, val_loaders_list, '验证集', args,
                class_names, feature_names, combination_name,
                combination_info, weights
            )
            all_results['val'].append(val_result)
            test_result = evaluate_combination(
                models, test_loaders_list, '测试集', args,
                class_names, feature_names, combination_name,
                combination_info, weights
            )
            all_results['test'].append(test_result)
        for model in models:
            del model
        torch.cuda.empty_cache()

        print(f"完成组合 {combination_name} 的评估")

    print("\n" + "=" * 80)
    print("生成所有组合的汇总报告")
    print("=" * 80)

    summary_file = os.path.join(args.report_dir, 'all_combinations_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f'多模型决策级融合全组合评估报告  {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('=' * 80 + '\n\n')
        f.write(f'总组合数: {len(all_combinations)}\n')
        f.write(f'模型配置:\n')
        f.write(f'  特征类型: {feature_types}\n')
        f.write(f'  模型类型: {model_types}\n')
        f.write(f'  类别数: {args.num_classes}\n')
        f.write(f'  输入尺寸: {args.input_size}x{args.input_size}\n')
        f.write(f'  融合权重: {weights if weights else "等权融合"}\n\n')

        f.write('验证集结果汇总:\n')
        f.write('-' * 120 + '\n')
        f.write(f"{'组合':<30} {'加权平均':<15} {'D-S证据':<15} {'多数投票':<15} {'最佳融合方法':<20}\n")
        f.write('-' * 120 + '\n')

        for val_result in all_results['val']:
            name = val_result['combination_name']
            weighted_acc = val_result['weighted_fusion']['accuracy']
            ds_acc = val_result['ds_fusion']['accuracy']
            mv_acc = val_result['majority_voting']['accuracy']

            fusion_methods = ['Weighted', 'D-S', 'Majority']
            fusion_accuracies = [weighted_acc, ds_acc, mv_acc]
            best_idx = np.argmax(fusion_accuracies)
            best_method = fusion_methods[best_idx]
            best_acc = fusion_accuracies[best_idx]

            f.write(f"{name:<30} {weighted_acc:<15.4f} {ds_acc:<15.4f} {mv_acc:<15.4f} {best_method} ({best_acc:.4f})\n")

        f.write('\n\n')

        f.write('测试集结果汇总:\n')
        f.write('-' * 120 + '\n')
        f.write(f"{'组合':<30} {'加权平均':<15} {'D-S证据':<15} {'多数投票':<15} {'最佳融合方法':<20}\n")
        f.write('-' * 120 + '\n')

        for test_result in all_results['test']:
            name = test_result['combination_name']
            weighted_acc = test_result['weighted_fusion']['accuracy']
            ds_acc = test_result['ds_fusion']['accuracy']
            mv_acc = test_result['majority_voting']['accuracy']

            fusion_methods = ['Weighted', 'D-S', 'Majority']
            fusion_accuracies = [weighted_acc, ds_acc, mv_acc]
            best_idx = np.argmax(fusion_accuracies)
            best_method = fusion_methods[best_idx]
            best_acc = fusion_accuracies[best_idx]

            f.write(f"{name:<30} {weighted_acc:<15.4f} {ds_acc:<15.4f} {mv_acc:<15.4f} {best_method} ({best_acc:.4f})\n")

        f.write('\n\n')

        best_val_idx = np.argmax([max(
            r['weighted_fusion']['accuracy'],
            r['ds_fusion']['accuracy'],
            r['majority_voting']['accuracy']
        ) for r in all_results['val']])

        best_test_idx = np.argmax([max(
            r['weighted_fusion']['accuracy'],
            r['ds_fusion']['accuracy'],
            r['majority_voting']['accuracy']
        ) for r in all_results['test']])

        best_val_result = all_results['val'][best_val_idx]
        best_test_result = all_results['test'][best_test_idx]

        f.write('最佳组合分析:\n')
        f.write('=' * 80 + '\n')
        f.write(f'验证集最佳组合: {best_val_result["combination_name"]}\n')
        f.write(f'验证集组合详情: {best_val_result["combination_info"]}\n')

        val_fusion_accuracies = [
            best_val_result['weighted_fusion']['accuracy'],
            best_val_result['ds_fusion']['accuracy'],
            best_val_result['majority_voting']['accuracy']
        ]
        val_best_method_idx = np.argmax(val_fusion_accuracies)
        val_best_method = ['加权平均', 'D-S证据', '多数投票'][val_best_method_idx]
        val_best_acc = val_fusion_accuracies[val_best_method_idx]

        f.write(f'验证集最佳融合方法: {val_best_method} (准确率: {val_best_acc:.4f})\n\n')

        f.write(f'测试集最佳组合: {best_test_result["combination_name"]}\n')
        f.write(f'测试集组合详情: {best_test_result["combination_info"]}\n')

        test_fusion_accuracies = [
            best_test_result['weighted_fusion']['accuracy'],
            best_test_result['ds_fusion']['accuracy'],
            best_test_result['majority_voting']['accuracy']
        ]
        test_best_method_idx = np.argmax(test_fusion_accuracies)
        test_best_method = ['加权平均', 'D-S证据', '多数投票'][test_best_method_idx]
        test_best_acc = test_fusion_accuracies[test_best_method_idx]

        f.write(f'测试集最佳融合方法: {test_best_method} (准确率: {test_best_acc:.4f})\n\n')

        f.write('组合性能排名（测试集）：\n')
        f.write('-' * 100 + '\n')

        combination_scores = []
        for i, test_result in enumerate(all_results['test']):
            best_acc = max(
                test_result['weighted_fusion']['accuracy'],
                test_result['ds_fusion']['accuracy'],
                test_result['majority_voting']['accuracy']
            )
            combination_scores.append((i, test_result['combination_name'], best_acc))

        combination_scores.sort(key=lambda x: x[2], reverse=True)

        for rank, (orig_idx, name, acc) in enumerate(combination_scores, 1):
            f.write(f"第{rank:2d}名: {name:<30} 最佳准确率: {acc:.4f}\n")

        f.write('\n' + '=' * 80 + '\n')

        f.write('模型选择建议:\n')
        f.write('-' * 80 + '\n')

        model_counts = defaultdict(int)
        top_n = min(5, len(combination_scores))

        for rank in range(top_n):
            _, name, _ = combination_scores[rank]
            parts = name.split('_')
            if len(parts) >= 5:
                for model_type in parts[2:5]:
                    model_counts[model_type] += 1

        f.write(f'前{top_n}名最佳组合中模型出现次数:\n')
        for model_type, count in model_counts.items():
            percentage = count / (top_n * 3) * 100
            f.write(f'  {model_type}: {count}次 ({percentage:.1f}%)\n')

        f.write('\n推荐的最佳模型组合:\n')
        for model_type in feature_types:
            feat_model_scores = defaultdict(list)
            for test_result in all_results['test']:
                for feat_name, model_type_in_combo in test_result['combination_info']:
                    if feat_name == model_type:
                        best_acc = max(
                            test_result['weighted_fusion']['accuracy'],
                            test_result['ds_fusion']['accuracy'],
                            test_result['majority_voting']['accuracy']
                        )
                        feat_model_scores[model_type_in_combo].append(best_acc)

            avg_scores = {}
            for model_type_in_combo, scores in feat_model_scores.items():
                avg_scores[model_type_in_combo] = np.mean(scores)

            if avg_scores:
                best_model = max(avg_scores.items(), key=lambda x: x[1])
                f.write(f'  {model_type}: {best_model[0]} (平均准确率: {best_model[1]:.4f})\n')

    print(f'\n汇总报告已保存至：{summary_file}')

    csv_file = os.path.join(args.report_dir, 'all_combinations_detailed.csv')
    rows = []

    for i, (val_result, test_result) in enumerate(zip(all_results['val'], all_results['test'])):
        row = {
            '组合编号': i+1,
            '组合名称': val_result['combination_name'],
            'Chirplet模型': val_result['combination_info'][0][1],
            'Mel模型': val_result['combination_info'][1][1],
            'STFT模型': val_result['combination_info'][2][1],
            '验证集_加权平均准确率': val_result['weighted_fusion']['accuracy'],
            '验证集_D-S准确率': val_result['ds_fusion']['accuracy'],
            '验证集_多数投票准确率': val_result['majority_voting']['accuracy'],
            '验证集_最佳准确率': max(val_result['weighted_fusion']['accuracy'],
                                     val_result['ds_fusion']['accuracy'],
                                     val_result['majority_voting']['accuracy']),
            '测试集_加权平均准确率': test_result['weighted_fusion']['accuracy'],
            '测试集_D-S准确率': test_result['ds_fusion']['accuracy'],
            '测试集_多数投票准确率': test_result['majority_voting']['accuracy'],
            '测试集_最佳准确率': max(test_result['weighted_fusion']['accuracy'],
                                     test_result['ds_fusion']['accuracy'],
                                     test_result['majority_voting']['accuracy'])
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f'详细结果CSV文件已保存至：{csv_file}')

    try:
        plot_combination_results(df, args.report_dir)
    except Exception as e:
        print(f"生成可视化图表时出错: {e}")

    print('\n全部评估完成！')


def plot_combination_results(df, report_dir):
    viz_dir = os.path.join(report_dir, 'visualization')
    os.makedirs(viz_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(df)), df['测试集_最佳准确率'].sort_values(ascending=False).values)
    plt.xlabel('组合排名')
    plt.ylabel('最佳准确率')
    plt.title('所有组合测试集最佳准确率分布')
    plt.xticks([])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'test_best_accuracy_distribution.png'), dpi=300)
    plt.close()

    model_types = ['resnet50', 'convnextv2', 'swin_transformer']
    model_performance = {model: [] for model in model_types}

    for model_type in model_types:
        mask = (df['Chirplet模型'] == model_type) | (df['Mel模型'] == model_type) | (df['STFT模型'] == model_type)
        model_performance[model_type] = df.loc[mask, '测试集_最佳准确率'].values

    plt.figure(figsize=(10, 6))
    positions = range(len(model_types))
    box_data = [model_performance[model] for model in model_types]

    plt.boxplot(box_data, positions=positions, labels=model_types)
    plt.xlabel('模型类型')
    plt.ylabel('测试集最佳准确率')
    plt.title('不同模型类型在组合中的表现')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'model_type_performance.png'), dpi=300)
    plt.close()

    top_10 = df.nlargest(10, '测试集_最佳准确率')

    plt.figure(figsize=(14, 8))
    x = range(len(top_10))
    width = 0.25

    plt.bar([i - width for i in x], top_10['测试集_加权平均准确率'], width, label='加权平均')
    plt.bar(x, top_10['测试集_D-S准确率'], width, label='D-S证据')
    plt.bar([i + width for i in x], top_10['测试集_多数投票准确率'], width, label='多数投票')

    plt.xlabel('组合')
    plt.ylabel('准确率')
    plt.title('前10名组合的测试集表现')
    plt.xticks(x, top_10['组合名称'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'top_10_combinations.png'), dpi=300)
    plt.close()

    print(f'可视化图表已保存至: {viz_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多模型决策级融合全组合评估')
    parser.add_argument('--chirplet_data_path', type=str,
                        default=r'E:\01_User\Dengyuxin\01_dataset\03_去噪—预处理—分割2s—有效音频_Chirplet语谱图_44100_384')
    parser.add_argument('--mel_data_path', type=str,
                        default=r'E:\01_User\Dengyuxin\01_dataset\03_去噪—预处理—分割2s—有效音频_Mel语谱图_44100_384')
    parser.add_argument('--stft_data_path', type=str,
                        default=r'E:\01_User\Dengyuxin\01_dataset\03_去噪—预处理—分割2s—有效音频_STFT语谱图_44100_384')
    parser.add_argument('--weights', type=str, default='0.25,0.35,0.40')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--input_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--backbone_type', type=str, default='convnextv2_tiny',
                        choices=['convnextv2_tiny', 'convnextv2_small', 'convnextv2_base', 'convnextv2_large',
                                 'convnextv2_huge'],
                        help='ConvNeXtV2变体类型')
    parser.add_argument('--model_save_path', type=str,
                        default=r'E:\01_User\Dengyuxin\07_论文提交程序\04 Multi-model decision-level fusion\model',
                        help='训练好的模型权重保存路径')
    parser.add_argument('--report_dir', type=str,
                        default=r'E:\01_User\Dengyuxin\07_论文提交程序\05 Result\07 Multi-model decision-level fusion\reports',
                        help='评估报告保存路径')
    parser.add_argument('--class_indict_path', type=str,
                        default=r'E:\01_User\Dengyuxin\07_论文提交程序\04 Multi-model decision-level fusion\class_indices.json')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    if not os.path.exists(args.report_dir):
        os.makedirs(args.report_dir)

    inference(args)