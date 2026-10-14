import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms
import json
import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from my_dataset_V1R6 import MyDataSet
from utils_convnextv2_Decision_Fusion import read_split_data_science
from convnextv2 import convnextv2_tiny, convnextv2_base, convnextv2_large, convnextv2_huge

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

        print(f"创建 {backbone_type} 模型，类别数: {num_classes}")

    def forward(self, x):
        out = self.backbone(x)
        if isinstance(out, tuple):
            return out[0]
        return out


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

def evaluate_dataset(models, loaders_list, dataset_type, args, class_names, feature_names, weights):
    print(f"\n{'=' * 80}\n开始 {dataset_type} 推理与融合评估\n{'=' * 80}")
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
        report = classification_report(labels, preds, labels=list(range(len(class_names))),
                                       target_names=class_names, output_dict=True, zero_division=0)

        individual_results[feat] = {
            'accuracy': acc,
            'report': report,
            'predictions': preds,
            'probabilities': probs.numpy()
        }

        all_probs.append(probs)
        if all_labels is None:
            all_labels = labels

    fused_w = FusionStrategy.weighted_average(all_probs, weights)
    fused_ds = FusionStrategy.ds_evidence_theory(all_probs)
    fused_mv = FusionStrategy.majority_voting(all_probs)
    preds_w = fused_w.argmax(dim=1).numpy()
    preds_ds = fused_ds.argmax(dim=1).numpy()
    preds_mv = fused_mv.argmax(dim=1).numpy()

    def calc_metrics(pred):
        acc = accuracy_score(all_labels, pred)
        report = classification_report(all_labels, pred, labels=list(range(len(class_names))),
                                       target_names=class_names, output_dict=True, zero_division=0)
        return acc, report

    acc_w, rep_w = calc_metrics(preds_w)
    acc_ds, rep_ds = calc_metrics(preds_ds)
    acc_mv, rep_mv = calc_metrics(preds_mv)

    os.makedirs(args.report_dir, exist_ok=True)
    ds_dir = os.path.join(args.report_dir, dataset_type)
    os.makedirs(ds_dir, exist_ok=True)

    def save_cm(pred, suffix):
        cm = confusion_matrix(all_labels, pred)
        plt.figure(figsize=(10, 9))

        abbrev_names = []
        for full_name in class_names:
            parts = full_name.split()
            if len(parts) >= 2:
                genus = parts[0][0].upper()
                species = parts[1][:3].lower()
                abbrev = genus + species
            else:
                abbrev = full_name[:4]
            abbrev_names.append(abbrev)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=abbrev_names, yticklabels=abbrev_names,
                    square=True)

        plt.title(f'{suffix} Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Pred')

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(ds_dir, f'{dataset_type}_{suffix}_CM.png'), dpi=300)
        plt.close()

    save_cm(preds_w, 'weighted')
    save_cm(preds_ds, 'DS')

    report_file = os.path.join(ds_dir, f'{dataset_type}_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f'{dataset_type} 评估报告  {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'类别映射：{dict(enumerate(class_names))}\n\n')

        f.write('【各模型单独评估结果】\n')
        for feat in feature_names:
            f.write(f'\n{feat}:\n')
            f.write(f'  Accuracy: {individual_results[feat]["accuracy"]:.4f}\n')

            formatted_report = format_classification_report(
                individual_results[feat]["report"],
                class_names,
                title=f"{feat} 分类报告"
            )
            f.write(f'\n{formatted_report}\n')

            f.write('\n' + '-' * 80 + '\n')

        f.write('\n' + '=' * 80 + '\n\n')

        f.write('【Weighted Average 融合】\n')
        f.write(f'Accuracy: {acc_w:.4f}\n')
        formatted_report_w = format_classification_report(rep_w, class_names, title="Weighted Average 分类报告")
        f.write(f'\n{formatted_report_w}\n')
        f.write('\n' + '-' * 80 + '\n\n')

        f.write('【D-S 证据理论融合】\n')
        f.write(f'Accuracy: {acc_ds:.4f}\n')
        formatted_report_ds = format_classification_report(rep_ds, class_names, title="D-S 证据理论分类报告")
        f.write(f'\n{formatted_report_ds}\n')
        f.write('\n' + '-' * 80 + '\n\n')

        f.write('【多数投票法融合】\n')
        f.write(f'Accuracy: {acc_mv:.4f}\n')
        formatted_report_mv = format_classification_report(rep_mv, class_names, title="多数投票法分类报告")
        f.write(f'\n{formatted_report_mv}\n')
        f.write('\n' + '-' * 80 + '\n')

    print(f'{dataset_type} 报告已保存至：{report_file}')

    return {
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

    feature_info = [
        ('Chirplet', args.chirplet_data_path),
        ('Mel', args.mel_data_path),
        ('STFT', args.stft_data_path)
    ]

    val_loaders, test_loaders, feature_names = [], [], []
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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
        val_loaders.append(DataLoader(val_set, batch_size=args.batch_size,
                                      sampler=SequentialSampler(val_set),
                                      num_workers=args.num_workers, pin_memory=True))
        test_loaders.append(DataLoader(test_set, batch_size=args.batch_size,
                                       sampler=SequentialSampler(test_set),
                                       num_workers=args.num_workers, pin_memory=True))
        feature_names.append(feat_name)
        print(f'- {feat_name}  验证{len(val_set)}  测试{len(test_set)}')

    models = []
    for feat_name, _ in feature_info:
        model = SimpleConvNeXtV2Model(
            backbone_type=args.backbone_type,
            num_classes=args.num_classes
        ).to(device)

        ckpt = os.path.join(args.model_save_path, f'best_convnextv2_{feat_name}.pth')

        if not os.path.exists(ckpt):
            alt_ckpt = os.path.join(args.model_save_path, f'best_convnextv2_{args.backbone_type}_{feat_name}.pth')
            if os.path.exists(alt_ckpt):
                ckpt = alt_ckpt
            else:
                raise FileNotFoundError(f"权重文件不存在：{ckpt}")

        print(f'加载 {feat_name} 模型权重: {ckpt}')

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

            if state_dict and list(state_dict.keys())[0].startswith('backbone.'):
                state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}

            model.backbone.load_state_dict(state_dict, strict=True)
            print(f"成功加载 {feat_name} 模型权重")

        except Exception as e:
            print(f"加载 {feat_name} 权重失败: {e}")
            raise e

        model.eval()
        models.append(model)
        print(f'- 已加载 {feat_name} {args.backbone_type} 模型')

    weights = None
    if args.weights:
        weights = list(map(float, args.weights.split(',')))
        if len(weights) != len(feature_names):
            print('警告：权重数量与模型数不符，采用等权融合')
            weights = None
        else:
            print('使用自定义权重：', weights)

    if models:
        total_params = sum(p.numel() for p in models[0].parameters())
        trainable_params = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
        print(f'\n模型参数信息:')
        print(f'  总参数量: {total_params:,}')
        print(f'  可训练参数量: {trainable_params:,}')
        print(f'  Backbone类型: {args.backbone_type}')
        print(f'  模型权重: 已加载训练好的完整模型权重')

    val_results = evaluate_dataset(models, val_loaders, '验证集', args, class_names, feature_names, weights)
    test_results = evaluate_dataset(models, test_loaders, '测试集', args, class_names, feature_names, weights)

    summary_file = os.path.join(args.report_dir, 'summary_report.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f'ConvNeXtV2 决策级融合汇总报告  {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('=' * 80 + '\n\n')
        f.write(f'模型配置:\n')
        f.write(f'  Backbone类型: {args.backbone_type}\n')
        f.write(f'  类别数: {args.num_classes}\n')
        f.write(f'  输入尺寸: {args.input_size}x{args.input_size}\n')
        f.write(f'  融合权重: {weights if weights else "等权融合"}\n\n')

        f.write('各特征单独模型在验证集上的表现:\n')
        for feat in feature_names:
            f.write(f'  {feat}: Accuracy={val_results["individual"][feat]["accuracy"]:.4f}\n')

        f.write('\n验证集详细分类报告（只展示最佳模型）:\n')
        best_val_model = max(feature_names, key=lambda x: val_results["individual"][x]["accuracy"])
        f.write(f'\n最佳单一模型 ({best_val_model}):\n')
        formatted_report = format_classification_report(
            val_results["individual"][best_val_model]["report"],
            class_names,
            title=f"{best_val_model} 验证集分类报告"
        )
        f.write(f'{formatted_report}\n')

        f.write('\n各特征单独模型在测试集上的表现:\n')
        for feat in feature_names:
            f.write(f'  {feat}: Accuracy={test_results["individual"][feat]["accuracy"]:.4f}\n')

        f.write('\n测试集详细分类报告（只展示最佳模型）:\n')
        best_test_model = max(feature_names, key=lambda x: test_results["individual"][x]["accuracy"])
        f.write(f'\n最佳单一模型 ({best_test_model}):\n')
        formatted_report = format_classification_report(
            test_results["individual"][best_test_model]["report"],
            class_names,
            title=f"{best_test_model} 测试集分类报告"
        )
        f.write(f'{formatted_report}\n')

        f.write('\n' + '=' * 80 + '\n')
        f.write('\n融合结果对比:\n')
        f.write('验证集:\n')
        f.write(f'  Weighted Fusion: Accuracy={val_results["weighted_fusion"]["accuracy"]:.4f}\n')
        f.write(f'  D-S Fusion: Accuracy={val_results["ds_fusion"]["accuracy"]:.4f}\n')
        f.write(f'  多数投票法: Accuracy={val_results["majority_voting"]["accuracy"]:.4f}\n')

        f.write('\n测试集:\n')
        f.write(f'  Weighted Fusion: Accuracy={test_results["weighted_fusion"]["accuracy"]:.4f}\n')
        f.write(f'  D-S Fusion: Accuracy={test_results["ds_fusion"]["accuracy"]:.4f}\n')
        f.write(f'  多数投票法: Accuracy={test_results["majority_voting"]["accuracy"]:.4f}\n')

        f.write('\n' + '=' * 80 + '\n')
        f.write('\n测试集最佳融合方法详细分类报告:\n')
        fusion_methods = ['Weighted Fusion', 'D-S Fusion', 'majority_voting']
        fusion_accuracies = [
            test_results["weighted_fusion"]["accuracy"],
            test_results["ds_fusion"]["accuracy"],
            test_results["majority_voting"]["accuracy"]
        ]
        best_method_idx = np.argmax(fusion_accuracies)
        best_method = fusion_methods[best_method_idx]
        f.write(f'\n最佳融合方法 ({best_method}):\n')

        if best_method_idx == 0:
            formatted_report = format_classification_report(
                test_results["weighted_fusion"]["report"],
                class_names,
                title="Weighted Fusion 测试集分类报告"
            )
        elif best_method_idx == 1:
            formatted_report = format_classification_report(
                test_results["ds_fusion"]["report"],
                class_names,
                title="D-S Fusion 测试集分类报告"
            )
        else:
            formatted_report = format_classification_report(
                test_results["majority_voting"]["report"],
                class_names,
                title="多数投票法 测试集分类报告"
            )
        f.write(f'{formatted_report}\n')

        best_single_acc = max([test_results["individual"][feat]["accuracy"] for feat in feature_names])
        best_fusion_acc = max(
            test_results["weighted_fusion"]["accuracy"],
            test_results["ds_fusion"]["accuracy"],
            test_results["majority_voting"]["accuracy"]
        )
        improvement = (best_fusion_acc - best_single_acc) * 100

        f.write(f'\n' + '=' * 80 + '\n')
        f.write(f'性能提升分析:\n')
        f.write(f'  最佳单模型准确率: {best_single_acc:.4f}\n')
        f.write(f'  最佳融合准确率: {best_fusion_acc:.4f}\n')
        f.write(f'  相对提升: {improvement:.2f}%\n')
        f.write(f'  最佳融合方法: {best_method}\n')

    print(f'\n汇总报告已保存至：{summary_file}')
    print('\n全部评估完成！结果保存在：', args.report_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ConvNeXtV2 决策级融合')
    parser.add_argument('--chirplet_data_path', type=str,
                        default=r'E:\01_User\Dengyuxin\01_dataset\03_去噪—预处理—分割2s—有效音频_Chirplet语谱图_44100_384')
    parser.add_argument('--mel_data_path', type=str,
                        default=r'E:\01_User\Dengyuxin\01_dataset\03_去噪—预处理—分割2s—有效音频_Mel语谱图_44100_384')
    parser.add_argument('--stft_data_path', type=str,
                        default=r'E:\01_User\Dengyuxin\01_dataset\03_去噪—预处理—分割2s—有效音频_STFT语谱图_44100_384')
    parser.add_argument('--weights', type=str, default='0.25,0.35,0.40')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--input_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--backbone_type', type=str, default='convnextv2_tiny',
                        choices=['convnextv2_tiny', 'convnextv2_small', 'convnextv2_base', 'convnextv2_large',
                                 'convnextv2_huge'],
                        help='ConvNeXtV2变体类型')
    parser.add_argument('--model_save_path', type=str,
                        default=r'E:\01_User\Dengyuxin\07_论文提交程序\01 ConvNeXt-V2\03_ConvNextV2_决策级融合\models',
                        help='训练好的模型权重保存路径')
    parser.add_argument('--report_dir', type=str,
                        default=r'E:\01_User\Dengyuxin\07_论文提交程序\05 Result\04 ConvNeXt-V2\03_ConvNextV2_决策级融合\reports1',
                        help='评估报告保存路径')
    parser.add_argument('--class_indict_path', type=str,
                        default=r'E:\01_User\Dengyuxin\07_论文提交程序\01 ConvNeXt-V2\03_ConvNextV2_决策级融合\class_indices.json')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    if not os.path.exists(args.report_dir):
        os.makedirs(args.report_dir)

    inference(args)