import warnings
warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

import torch
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import os
import argparse
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from utils_ResNet50_Feature_Fusion_Concat import (
    set_seed, read_split_data, MultiFeatureDataset,
    create_lr_scheduler, get_params_groups, ConcatFusionModel
)
from my_dataset_V1R6 import RandomTimeMasking, RandomFrequencyMasking
from early_stopping import EarlyStopping


class Trainer:
    def __init__(self, args, run_id=1):
        self.args = args
        self.run_id = run_id
        self.device = torch.device(args.device)
        self.run_log_dir = os.path.join(args.log_dir, f'run_{run_id}')
        self.model_best_path = os.path.join(args.model_save_dir, f'best_{run_id}.pth')
        self.model_last_path = os.path.join(args.model_save_dir, f'last_{run_id}.pth')

        os.makedirs(self.run_log_dir, exist_ok=True)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }

        self._setup_logging()

    def _setup_logging(self):
        log_file = os.path.join(self.run_log_dir, f'train_{self.run_id}.log')

        if os.path.exists(log_file):
            os.remove(log_file)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.tb_writer = SummaryWriter(log_dir=self.run_log_dir)

        self.logger.info(f"Training Configuration:")
        self.logger.info(f"  Epochs: {self.args.epochs}")
        self.logger.info(f"  Batch Size: {self.args.batch_size}")
        self.logger.info(f"  Learning Rate: {self.args.lr}")
        self.logger.info(f"  Device: {self.device}")

    def _get_data_transforms(self):
        return {
            'train': transforms.Compose([
                transforms.RandomApply([transforms.RandomAffine(0, translate=(0.3, 0.15))], p=0.3),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                RandomFrequencyMasking(freq_mask_param=15, min_masks=1, max_masks=3, apply_prob=0.3),
                RandomTimeMasking(time_mask_param=30, min_masks=1, max_masks=3, apply_prob=0.3),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    def _create_data_loaders(self):
        data_loaders = {}
        transforms_dict = self._get_data_transforms()

        for feature_type, data_path in self.args.data_paths.items():
            train_imgs, train_labs, val_imgs, val_labs, test_imgs, test_labs = read_split_data(
                data_path, test_scale=0.2, val_scale=0.2
            )

            train_dataset = MultiFeatureDataset(train_imgs, train_labs, transforms_dict['train'])
            val_dataset = MultiFeatureDataset(val_imgs, val_labs, transforms_dict['val'])
            test_dataset = MultiFeatureDataset(test_imgs, test_labs, transforms_dict['test'])

            train_sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(self.args.seed))
            val_sampler = SequentialSampler(val_dataset)

            data_loaders[f'train_{feature_type}'] = DataLoader(
                train_dataset, batch_size=self.args.batch_size,
                sampler=train_sampler, pin_memory=True,
                num_workers=2, persistent_workers=True
            )
            data_loaders[f'val_{feature_type}'] = DataLoader(
                val_dataset, batch_size=self.args.batch_size,
                sampler=val_sampler, pin_memory=True,
                num_workers=2, persistent_workers=True
            )
            data_loaders[f'test_{feature_type}'] = DataLoader(
                test_dataset, batch_size=self.args.batch_size,
                shuffle=False, pin_memory=True,
                num_workers=2
            )

            self.logger.info(f"{feature_type}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

        return data_loaders

    def _setup_model(self):
        self.model = ConcatFusionModel(
            num_classes=self.args.num_classes,
            feature_dim=2048,
        ).to(self.device)
        if self.args.freeze_backbone:
            frozen_count = 0
            for name, param in self.model.named_parameters():
                if 'backbone_c' in name or 'backbone_m' in name or 'backbone_s' in name:
                    param.requires_grad = False
                    frozen_count += 1

            self.logger.info(f"Frozen {frozen_count} backbone parameters")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: Total={total_params:,}, Trainable={trainable_params:,}")

    def _setup_optimizer(self, train_loader):
        params_groups = get_params_groups(self.model, weight_decay=self.args.wd)

        self.optimizer = optim.AdamW(
            params_groups,
            lr=self.args.lr,
            weight_decay=self.args.wd
        )

        self.scheduler = create_lr_scheduler(
            self.optimizer,
            num_step=len(train_loader),
            epochs=self.args.epochs,
            warmup=True,
            warmup_epochs=self.args.warmup_epochs
        )

    def train_epoch(self, train_loaders):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(enumerate(zip(*train_loaders)),
                    total=len(train_loaders[0]),
                    desc=f'Epoch {self.current_epoch}/{self.args.epochs}',
                    leave=False)

        for batch_idx, batches in pbar:
            (xc, yc), (xm, _), (xs, _) = batches
            xc, xm, xs = xc.to(self.device), xm.to(self.device), xs.to(self.device)
            y = yc.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(xc, xm, xs)
            loss = self.criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += batch_size

            if batch_idx % 20 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{(preds == y).float().mean().item():.3f}'
                })

        return total_loss / total, correct / total

    @torch.no_grad()
    def validate(self, val_loaders):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for batches in zip(*val_loaders):
            (xc, yc), (xm, _), (xs, _) = batches
            xc, xm, xs = xc.to(self.device), xm.to(self.device), xs.to(self.device)
            y = yc.to(self.device)

            logits = self.model(xc, xm, xs)
            loss = self.criterion(logits, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += batch_size

        return total_loss / total, correct / total

    def train(self):
        self.logger.info(f"Starting training run {self.run_id}")
        self.logger.info(f"Device: {self.device}")

        set_seed(self.args.seed)

        data_loaders = self._create_data_loaders()
        train_loaders = [data_loaders['train_chirplet'], data_loaders['train_mel'], data_loaders['train_stft']]
        val_loaders = [data_loaders['val_chirplet'], data_loaders['val_mel'], data_loaders['val_stft']]

        self._setup_model()
        self._setup_optimizer(train_loaders[0])

        early_stopper = EarlyStopping(
            patience=self.args.patience,
            mode='max',
            checkpoint_path=self.model_best_path
        )

        start_time = time.time()
        best_val_acc, best_epoch = 0.0, 0

        for epoch in range(1, self.args.epochs + 1):
            self.current_epoch = epoch

            train_loss, train_acc = self.train_epoch(train_loaders)
            val_loss, val_acc = self.validate(val_loaders)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            self.tb_writer.add_scalar("Loss/train", train_loss, epoch)
            self.tb_writer.add_scalar("Loss/val", val_loss, epoch)
            self.tb_writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.tb_writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.tb_writer.add_scalar("Learning_rate", self.optimizer.param_groups[0]['lr'], epoch)

            self.logger.info(
                f"Epoch {epoch:03d}/{self.args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            early_stopper(epoch, self.model, val_acc)
            if early_stopper.early_stop:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        torch.save(self.model.state_dict(), self.model_last_path)
        training_time = time.time() - start_time
        self.logger.info(f"Training completed. Best model saved to: {self.model_best_path}")
        self.logger.info(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
        self.logger.info(f"Training time: {training_time:.2f}s ({training_time / 60:.2f}min)")
        self._save_training_history()
        self._plot_training_curves(best_epoch, best_val_acc)
        self.tb_writer.close()

        return self.history

    def _save_training_history(self):
        history_path = os.path.join(self.run_log_dir, f'training_history_{self.run_id}.npz')
        np.savez(history_path,
                 train_loss=self.history['train_loss'],
                 val_loss=self.history['val_loss'],
                 train_acc=self.history['train_acc'],
                 val_acc=self.history['val_acc'],
                 learning_rates=self.history['learning_rates'])

    def _plot_training_curves(self, best_epoch, best_val_acc):
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        axes[0].plot(self.history['train_loss'], label='Train Loss', color='blue', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', color='red', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history['train_acc'], label='Train Accuracy', color='blue', linewidth=2)
        axes[1].plot(self.history['val_acc'], label='Val Accuracy', color='red', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(self.history['learning_rates'], label='Learning Rate', color='green', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Learning Rate', fontsize=12)
        axes[2].set_title('Learning Rate Schedule', fontsize=14)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        curve_path = os.path.join(self.run_log_dir, f'training_curves_{self.run_id}.png')
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Training curves saved to: {curve_path}")


class Evaluator:
    @staticmethod
    def evaluate_model(model, device, test_loaders, class_indices_path, output_dir, model_name):

        with open(class_indices_path, 'r', encoding='utf-8') as f:
            class_indices = json.load(f)
        class_indices = {int(k): v for k, v in class_indices.items()}

        with open('class_indices_en.json', 'r', encoding='utf-8') as f:
            class_names_en = json.load(f)

        sorted_indices = sorted(class_indices.keys())
        class_names = [class_names_en[str(idx)] for idx in sorted_indices]

        model.eval()
        all_true_labels, all_pred_labels = [], []

        with torch.no_grad():
            for batches in zip(*test_loaders):
                (xc, yc), (xm, _), (xs, _) = batches
                xc, xm, xs = xc.to(device), xm.to(device), xs.to(device)

                logits = model(xc, xm, xs)
                preds = logits.argmax(dim=1).cpu().numpy()

                all_true_labels.extend(yc.numpy())
                all_pred_labels.extend(preds)

        accuracy = np.mean(np.array(all_true_labels) == np.array(all_pred_labels))

        Evaluator._save_confusion_matrix(
            all_true_labels, all_pred_labels, class_names,
            os.path.join(output_dir, f'confusion_matrix_{model_name}.png'),
            accuracy
        )

        Evaluator._save_classification_report(
            all_true_labels, all_pred_labels, class_names,
            os.path.join(output_dir, f'classification_report_{model_name}.txt'),
            accuracy
        )

        return accuracy

    @staticmethod
    def _get_class_names(class_indices):
        sorted_indices = sorted(class_indices.items())
        return [name for idx, name in sorted_indices]

    @staticmethod
    def _save_confusion_matrix(true_labels, pred_labels, class_names,
                               output_dir, model_name, dataset_type='test'):

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

        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 9))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=abbrev_names, yticklabels=abbrev_names,
                    square=True)

        plt.title(f'Confusion Matrix ', fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)

        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)

        plt.tight_layout()

        cm_path = os.path.join(output_dir, f'{dataset_type}_{model_name}_CM.png')
        full_cm_path = os.path.join(output_dir, f'{dataset_type}_{model_name}_CM_full.png')

        os.makedirs(os.path.dirname(cm_path), exist_ok=True)

        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix with Full Names ',
                  fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        os.makedirs(os.path.dirname(full_cm_path), exist_ok=True)

        plt.savefig(full_cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to: {cm_path}")
        print(f"Full name confusion matrix saved to: {full_cm_path}")

    @staticmethod
    def _save_classification_report(true_labels, pred_labels, class_names, save_path, accuracy):

        report = classification_report(true_labels, pred_labels,
                                       target_names=class_names, digits=4, zero_division=0)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            f.write(report)

        print(f"Classification report saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='ResNet Concat Feature Fusion Training')

    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_runs', type=int, default=1)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=150)

    parser.add_argument('--weights', type=str,
                        default=r"E:\01_User\Dengyuxin\07_论文提交程序\02 ResNet\01 ResNet\resnet50.pth")
    parser.add_argument('--model_save_dir', type=str,
                        default=r'E:\01_User\Dengyuxin\07_论文提交程序\05 Result\05 ResNet\02_ResNet_特征级融合\02_Concat特征融合\weights\FeatureFusion')
    parser.add_argument('--log_dir', type=str,
                        default=r'E:\01_User\Dengyuxin\07_论文提交程序\05 Result\05 ResNet\02_ResNet_特征级融合\02_Concat特征融合\runs\FeatureFusion')
    parser.add_argument('--report_dir', type=str,
                        default=r'E:\01_User\Dengyuxin\07_论文提交程序\05 Result\05 ResNet\02_ResNet_特征级融合\02_Concat特征融合\reports\FeatureFusion')
    parser.add_argument('--freeze_backbone', action='store_true', help='冻结骨干网络')
    parser.add_argument('--device', type=str, default='cuda:0', help='训练设备')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--bottleneck_dim', type=int, default=2048, help='bottleneck dimension of feature')
    args = parser.parse_args()

    args.data_paths = {
        'chirplet': r'E:\01_User\Dengyuxin\01_dataset\03_去噪—预处理—分割2s—有效音频_Chirplet语谱图_44100_384',
        'mel': r'E:\01_User\Dengyuxin\01_dataset\03_去噪—预处理—分割2s—有效音频_Mel语谱图_44100_384',
        'stft': r'E:\01_User\Dengyuxin\01_dataset\03_去噪—预处理—分割2s—有效音频_STFT语谱图_44100_384'
    }

    for directory in [args.model_save_dir, args.log_dir, args.report_dir]:
        os.makedirs(directory, exist_ok=True)

    start_time = time.time()
    total_start = start_time

    train_start = time.time()
    all_histories = []

    print(f"\n{'=' * 60}")
    print(f"Starting Concat Fusion Training")
    print(f"{'=' * 60}")

    for run_id in range(1, args.num_runs + 1):
        print(f"\n{'=' * 60}")
        print(f"Run {run_id}/{args.num_runs}")
        print(f"{'=' * 60}")

        trainer = Trainer(args, run_id)
        history = trainer.train()
        all_histories.append(history)

    train_time = time.time() - train_start

    test_start = time.time()

    test_loaders = []
    for feature_type in ['chirplet', 'mel', 'stft']:
        data_path = args.data_paths[feature_type]
        _, _, _, _, test_imgs, test_labs = read_split_data(
            data_path, test_scale=0.2, val_scale=0.2
        )

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = MultiFeatureDataset(test_imgs, test_labs, transform)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)
        test_loaders.append(loader)

    device = torch.device(args.device)
    model_files = [f for f in os.listdir(args.model_save_dir) if f.endswith('.pth')]

    print(f"\n{'=' * 60}")
    print("Evaluation Results")
    print(f"{'=' * 60}")

    for model_file in model_files:
        print(f"\nEvaluating: {model_file}")

        model_path = os.path.join(args.model_save_dir, model_file)
        model_name = os.path.splitext(model_file)[0]


        model = ConcatFusionModel(
            num_classes=args.num_classes,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        accuracy = Evaluator.evaluate_model(
            model=model,
            device=device,
            test_loaders=test_loaders,
            class_indices_path='class_indices.json',
            output_dir=args.report_dir,
            model_name=model_name
        )

        print(f"Model {model_file} test accuracy: {accuracy:.4f}")

    test_time = time.time() - test_start
    total_time = time.time() - total_start

    print(f"\n{'=' * 60}")
    print("Training and Evaluation Completed!")
    print(f"{'=' * 60}")
    print(f"Training time: {train_time:.2f}s ({train_time / 60:.2f}min)")
    print(f"Evaluation time: {test_time:.2f}s")
    print(f"Total time: {total_time:.2f}s ({total_time / 60:.2f}min)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()