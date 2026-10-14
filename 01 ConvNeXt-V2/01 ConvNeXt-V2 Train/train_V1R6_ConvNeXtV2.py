import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler
from torchvision import transforms
import matplotlib.pyplot as plt

from my_dataset_V1R6 import MyDataSet, RandomTimeMasking, RandomFrequencyMasking
from early_stopping import EarlyStopping
from convnextv2 import convnextv2_tiny as create_model
from utils_V1R6 import read_split_data_science, create_lr_scheduler, get_params_groups, \
    train_one_epoch, evaluate, set_seed, count_parameters, evaluate_model, get_model_files, save_training_data
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
import numpy as np


def train(args):
    print(f'训练次数：{args.number}\n')
    print(args)
    if torch.cuda.is_available():
        device = torch.device(args.device)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    print(f"使用 {device} 进行模型训练.")
    print('使用Tensorboard命令：tensorboard --logdir=runs/... 查看模型训练日志')

    if os.path.exists(args.model_save_path) is False:
        os.makedirs(args.model_save_path)
    if os.path.exists(args.log_dir) is False:
        os.makedirs(args.log_dir)
    if not os.path.exists(args.report_directory):
        os.makedirs(args.report_directory)

    for number in range(1, args.number + 1):
        print(f'\n开始第 {number} 次训练...')
        foldename = 'training_curves_{}'.format(number)
        folder_path = os.path.join(args.log_dir, foldename)
        os.makedirs(folder_path, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=folder_path)

        set_seed(0)

        train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = \
            read_split_data_science(args.data_path, 'class_indices_' + os.path.basename(args.data_path) + '.json')

        data_transform = {
            "train": transforms.Compose([
                transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.3, 0.15), scale=None, shear=None)], p=0.3),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                RandomFrequencyMasking(freq_mask_param=15, min_masks=1, max_masks=3, apply_prob=0.3),
                RandomTimeMasking(time_mask_param=30, min_masks=1, max_masks=3, apply_prob=0.3),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

            "val": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

        train_dataset = MyDataSet(images_path=train_images_path,
                                  images_class=train_images_label,
                                  transform=data_transform["train"])
        val_dataset = MyDataSet(images_path=val_images_path,
                                images_class=val_images_label,
                                transform=data_transform["val"])
        train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=None,
                                      generator=torch.Generator().manual_seed(0))
        val_sampler = SequentialSampler(val_dataset)

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  pin_memory=True,
                                  persistent_workers=True,
                                  num_workers=2
                                  )
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=val_sampler,
                                pin_memory=True,
                                persistent_workers=True,
                                num_workers=2
                                )

        model = create_model(num_classes=args.num_classes)
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=device)
            weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))
        if args.freeze_layers:
            for name, para in model.named_parameters():
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))
        model = model.to(device)
        pg = get_params_groups(model, weight_decay=args.wd)
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                           warmup=True, warmup_epochs=args.warmup_epochs)

        weight_name_best = 'best_{}.pth'.format(number)
        weight_name_last = 'last_{}.pth'.format(number)

        acc_save_path = os.path.join(args.model_save_path, weight_name_best)
        last_path = os.path.join(args.model_save_path, weight_name_last)

        es_acc = EarlyStopping(patience=args.patience, mode='max', min_delta=0, checkpoint_path=acc_save_path)

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        learning_rates = []

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    epochs=args.epochs,
                                                    lr_scheduler=lr_scheduler)

            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch,
                                         epochs=args.epochs
                                         )

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            learning_rates.append(optimizer.param_groups[0]["lr"])

            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            es_acc(epoch, model, val_acc)
            if es_acc.early_stop:
                break
            torch.save(model.state_dict(), last_path)
        plt.figure(figsize=(6, 12))
        plt.subplot(3, 1, 1)
        plt.plot(range(1, epoch + 1), train_losses, label='Train Loss')
        plt.plot(range(1, epoch + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.ylim(0, plt.ylim()[1])
        plt.plot(range(1, epoch + 1), train_accuracies, label='Train Accuracy')
        plt.plot(range(1, epoch + 1), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        train_max_index = np.argmax(train_accuracies)
        train_max_value = train_accuracies[train_max_index]
        val_max_index = np.argmax(val_accuracies)
        val_max_value = val_accuracies[val_max_index]
        plt.scatter(train_max_index + 1, train_max_value, color='blue', marker='o', zorder=5)
        plt.text(train_max_index + 1, train_max_value, f'({train_max_index + 1}, {train_max_value:.4f})',
                 verticalalignment='bottom', horizontalalignment='right')
        plt.scatter(val_max_index + 1, val_max_value, color='red', marker='o', zorder=5)
        plt.text(val_max_index + 1, val_max_value, f'({val_max_index + 1}, {val_max_value:.4f})',
                 verticalalignment='top', horizontalalignment='right')

        plt.subplot(3, 1, 3)
        plt.plot(range(1, epoch + 1), learning_rates, label='Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        data_name = 'training_data_{}.npz'.format(number)
        data_path = os.path.join(folder_path, data_name)
        save_training_data(train_losses, val_losses, train_accuracies, val_accuracies, learning_rates, data_path)

        image_name = 'training_curves_{}.png'.format(number)
        image_path = os.path.join(folder_path, image_name)
        plt.savefig(image_path)
        plt.close()


def predict(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n使用 {device} 进行模型推理.")
    set_seed(0)

    train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = \
        read_split_data_science(args.data_path, 'class_indices_' + os.path.basename(args.data_path) + '.json')
    class_indict_path = 'class_indices_' + os.path.basename(args.data_path) + '.json'
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_dataset = MyDataSet(images_path=test_images_path, images_class=test_images_label, transform=data_transform)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             sampler=test_sampler,
                             pin_memory=True,
                             persistent_workers=True,
                             num_workers=1
                             )

    model_list = get_model_files(args.model_save_path)

    for model_path in model_list:
        model_name = os.path.basename(model_path)
        print(f'\n推理模型：{model_name}')
        name_without_extension, _ = model_name.split('.')

        if not os.path.exists(os.path.join(args.report_directory, name_without_extension)):
            os.makedirs(os.path.join(args.report_directory, name_without_extension))

        confusion_matrix_path = os.path.join(args.report_directory, name_without_extension, 'confusion_matrix.png')
        classification_report_path = os.path.join(args.report_directory, name_without_extension,
                                                  'classification_report.txt')

        model = create_model(num_classes=args.num_classes)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        evaluate_model(model, device, test_loader, class_indict_path, confusion_matrix_path, classification_report_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=20)  #
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--number', type=int, default=1)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--wd', type=float, default=4e-2)
    parser.add_argument('--warmup_epochs', type=int, default=15)
    parser.add_argument('--data_path', type=str,
                        default=r"E:\01_User\Dengyuxin\01_dataset\03_去噪—预处理—分割2s—有效音频_Mel语谱图_44100_384")
    parser.add_argument('--weights', type=str, default=r'E:\01_User\Dengyuxin\07_论文提交程序\01 ConvNeXt-V2\01_ConvNextV2/convnextv2_tiny_22k_384_ema.pt',
                        help='initial weights path')
    parser.add_argument('--model_save_path', type=str,
                        default=r"E:\01_User\Dengyuxin\07_论文提交程序\05 Result\04 ConvNeXt-V2\01 ConvNeXtV2\weights\Mel",
                        help='weights save path')
    parser.add_argument('--log_dir', type=str,
                        default=r"E:\01_User\Dengyuxin\02_Pytorch\Pytorch\ConvNeXt-V2\01 ConvNeXtV2\runs\Mel")
    parser.add_argument('--report_directory', type=str,
                        default=r"E:\01_User\Dengyuxin\02_Pytorch\Pytorch\ConvNeXt-V2\01 ConvNeXtV2\reports\Mel")

    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--patience', type=int, default=150, help='设置为 epochs 时不使用 patience')

    opt = parser.parse_args()
    start_time = time.time()
    train(opt)
    train_time = time.time()
    print(f'训练用时：{int(train_time - start_time)}s')
    predict(opt)
    predict_time = time.time()
    print(f'总用时：{int(predict_time - start_time)}s')