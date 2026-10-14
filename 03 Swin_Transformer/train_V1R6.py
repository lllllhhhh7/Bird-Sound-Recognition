'''
    作者：罗创
    版本：V1R6
    特性：
        1.总共保存2个模型，分别是最好的acc模型、最后的模型
        2.设置随机种子，保证洗牌顺序、数据增强顺序保持一致
        3.可以计算模型参数量和计算量（可选）
        4.预热轮次设置为5轮
        5.先加载权重、冻结参数，再加载到GPU中，可以避免权重在 CPU 训练
        6.整合了推理程序
        7.新增参数量、计算量的计算方式
        8.修复早停后，不能画图的问题
        9.ConvNextV2 加载权重会删除不期望的键
        10.新增运行次数选项
        11.修复训练结果不一致的问题(2024.07.15)
        12.log文件夹内自动创建文件夹，区分log文件
        13.新增权重采样器(2024.07.24)，效果不明显，不使用
        14.学习率、loss等训练数据会保存至本地，对应函数save_training_data(2024.07.27)
    时间：2024.07.27
'''

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
from model import swin_base_patch4_window12_384 as create_model
# from MobileViT.mobilevit_v1_v2_v3 import mobilevit_v3_w0_5 as create_model
from utils_V1R6 import read_split_data_science, create_lr_scheduler, get_params_groups, \
    train_one_epoch, evaluate, set_seed, count_parameters, evaluate_model, get_model_files, save_training_data
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
import numpy as np


def train(args):
    print(f'训练次数：{args.number}\n')
    print(args)  # 输出训练参数设置
    if torch.cuda.is_available():
        device = torch.device(args.device)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    print(f"使用 {device} 进行模型训练.")
    print('使用Tensorboard命令：tensorboard --logdir=runs/... 查看模型训练日志')

    # 创建模型保存路径
    if os.path.exists(args.model_save_path) is False:
        os.makedirs(args.model_save_path)
    # 创建 log 保存路径
    if os.path.exists(args.log_dir) is False:
        os.makedirs(args.log_dir)
    # 创建报告保存目录
    if not os.path.exists(args.report_directory):
        os.makedirs(args.report_directory)

    for number in range(1, args.number + 1):
        print(f'\n开始第 {number} 次训练...')
        # 新建文件夹
        foldename = 'training_curves_{}'.format(number)
        folder_path = os.path.join(args.log_dir, foldename)
        os.makedirs(folder_path, exist_ok=True)
        # 保存日志文件
        tb_writer = SummaryWriter(log_dir=folder_path)

        # 设置随机种子为0
        set_seed(0)

        # 获取训练集、验证集、测试集的图片路径和标签
        # train_data_path = os.path.join(args.data_path, 'train_noise')
        # val_data_path = os.path.join(args.data_path, 'val')
        #
        # train_images_path, train_images_label = read_data(train_data_path)
        # val_images_path, val_images_label = read_data(val_data_path)
        train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = \
            read_split_data_science(args.data_path, 'class_indices_' + os.path.basename(args.data_path) + '.json')

        # 不使用数据增强
        # data_transform = {
        #     "train": transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        #     "val": transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

        # 频谱数据增强
        data_transform = {
            "train": transforms.Compose([
                # 时间位移、频率位移
                transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.3, 0.15), scale=None, shear=None)], p=0.3),
                # 水平翻转
                transforms.RandomHorizontalFlip(p=0.3),
                # 转换为Tensor
                transforms.ToTensor(),
                # 频率遮掩
                RandomFrequencyMasking(freq_mask_param=15, min_masks=1, max_masks=3, apply_prob=0.3),
                # 时间遮掩
                RandomTimeMasking(time_mask_param=30, min_masks=1, max_masks=3, apply_prob=0.3),
                # 归一化
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

            "val": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

        # 实例化训练数据集
        train_dataset = MyDataSet(images_path=train_images_path,
                                  images_class=train_images_label,
                                  transform=data_transform["train"])

        # 实例化验证数据集
        val_dataset = MyDataSet(images_path=val_images_path,
                                images_class=val_images_label,
                                transform=data_transform["val"])

        # # 数据权重平衡，并设置随机种子为0
        # class_counts = np.bincount(train_images_label)  # 统计每个类别的数量
        # class_weights = 1. / class_counts  # 计算每个类别的权重
        # sample_weights = class_weights[train_images_label]  # 生成每个样本的权重
        # train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True, generator=torch.Generator().manual_seed(0))

        # 数据洗牌，并设置随机种子为0
        train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=None,
                                      generator=torch.Generator().manual_seed(0))

        # 为 DataLoader 的 sampler 参数提供一个 SequentialSampler。每个epoch 不会重新洗牌
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

        # 创建模型
        model = create_model(num_classes=args.num_classes)

        # x = torch.randn(1, 3, 224, 224)
        # # 计算模型参数量和计算量
        # flop_analyzer = FlopCountAnalysis(model, x)
        # print(f'\n{flop_count_table(flop_analyzer)}')

        # 加载预训练权重
        # 加载预训练权重
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=device)
            weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict

            # 修正：删除分类头（head）相关的权重（SwinTransformer的分类头命名为head）
            for k in list(weights_dict.keys()):
                if "head" in k:  # 将"classifier"改为"head"，匹配SwinTransformer的命名
                    del weights_dict[k]

            # 加载过滤后的权重（仅保留特征提取部分）
            print(model.load_state_dict(weights_dict, strict=False))

        # 参数冻结
        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

        # 将模型移到设备上
        model = model.to(device)

        # 优化器和学习率调度策略
        pg = get_params_groups(model, weight_decay=args.wd)
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
        # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=args.wd)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                           warmup=True, warmup_epochs=args.warmup_epochs)

        # 早停策略
        weight_name_best = 'best_{}.pth'.format(number)
        weight_name_last = 'last_{}.pth'.format(number)

        acc_save_path = os.path.join(args.model_save_path, weight_name_best)
        last_path = os.path.join(args.model_save_path, weight_name_last)

        es_acc = EarlyStopping(patience=args.patience, mode='max', min_delta=0, checkpoint_path=acc_save_path)

        # 在训练循环开始前初始化 5 个列表
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        learning_rates = []

        # 模型训练的每一步过程显示
        for epoch in range(1, args.epochs + 1):
            # train
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    epochs=args.epochs,
                                                    lr_scheduler=lr_scheduler)

            # validate
            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch,
                                         epochs=args.epochs
                                         )

            # 在每个epoch结束时，将训练和验证的损失和准确率添加到相应的列表中
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            learning_rates.append(optimizer.param_groups[0]["lr"])

            # 使用tensorboard记录数据
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            # mode 设置为'min'，使用 val_loss 作为指标；设置为'max'，使用 val_acc 作为指标
            es_acc(epoch, model, val_acc)
            # 达到patience，触发早停，跳出循环，这里只使用acc作为结束程序的条件
            if es_acc.early_stop:
                break
            # 保存最后的模型
            torch.save(model.state_dict(), last_path)

        # 在训练循环结束后，使用matplotlib来绘制损失、准确率和学习率曲线
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
        # 找到训练准确率曲线的最高点
        train_max_index = np.argmax(train_accuracies)
        train_max_value = train_accuracies[train_max_index]
        # 找到验证准确率曲线的最高点
        val_max_index = np.argmax(val_accuracies)
        val_max_value = val_accuracies[val_max_index]
        # 标记训练准确率曲线的最高点
        plt.scatter(train_max_index + 1, train_max_value, color='blue', marker='o', zorder=5)
        plt.text(train_max_index + 1, train_max_value, f'({train_max_index + 1}, {train_max_value:.4f})',
                 verticalalignment='bottom', horizontalalignment='right')
        # 标记验证准确率曲线的最高点
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

        # 保存数据和图像到本地
        data_name = 'training_data_{}.npz'.format(number)
        data_path = os.path.join(folder_path, data_name)
        save_training_data(train_losses, val_losses, train_accuracies, val_accuracies, learning_rates, data_path)

        image_name = 'training_curves_{}.png'.format(number)
        image_path = os.path.join(folder_path, image_name)
        plt.savefig(image_path)

        # 关闭图像，释放资源
        plt.close()


def predict(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n使用 {device} 进行模型推理.")

    # 设计随机种子为0
    set_seed(0)

    # 加载测试数据集
    # test_data_path = os.path.join(args.data_path, 'test')
    # test_images_path, test_images_label = read_data(test_data_path)
    train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = \
        read_split_data_science(args.data_path, 'class_indices_' + os.path.basename(args.data_path) + '.json')

    # 加载类别索引字典
    class_indict_path = 'class_indices_' + os.path.basename(args.data_path) + '.json'

    # 仅归一化
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 实例化谱图测试集
    test_dataset = MyDataSet(images_path=test_images_path, images_class=test_images_label, transform=data_transform)

    # 为 DataLoader 的 sampler 参数提供一个 SequentialSampler。每个epoch 不会重新洗牌
    test_sampler = SequentialSampler(test_dataset)

    # 加载语谱图测试集
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             sampler=test_sampler,
                             pin_memory=True,
                             persistent_workers=True,
                             num_workers=1
                             )

    # 检索路径中存在的模型
    model_list = get_model_files(args.model_save_path)

    for model_path in model_list:
        # 获取不带后缀的模型名称
        model_name = os.path.basename(model_path)
        print(f'\n推理模型：{model_name}')
        name_without_extension, _ = model_name.split('.')

        # 创建报告保存目录
        if not os.path.exists(os.path.join(args.report_directory, name_without_extension)):
            os.makedirs(os.path.join(args.report_directory, name_without_extension))

        # 构建混淆矩阵和分类报告的路径
        confusion_matrix_path = os.path.join(args.report_directory, name_without_extension, 'confusion_matrix.png')
        classification_report_path = os.path.join(args.report_directory, name_without_extension,
                                                  'classification_report.txt')

        # 创建模型
        model = create_model(num_classes=args.num_classes)
        # 加载权重
        state_dict = torch.load(model_path, map_location=device)
        # # 删除模型参数和操作计数相关的额外信息，训练时调用 flops_params 函数，会保存这些信息，需要删除，否则权重加载会报错
        # for key in list(state_dict.keys()):
        #     if 'total_ops' in key or 'total_params' in key:
        #         del state_dict[key]
        # 加载筛选后的权重，
        model.load_state_dict(state_dict)
        # 将模型移到设备上
        model.to(device)

        # 执行模型评估
        evaluate_model(model, device, test_loader, class_indict_path, confusion_matrix_path, classification_report_path)


if __name__ == '__main__':
    '''
    修改点：
        1.模型导入
        2.是否数据增强
        3.预训练权重加载
        4.CUDA选择
    '''

    parser = argparse.ArgumentParser()
    # 参数设置
    parser.add_argument('--num_classes', type=int, default=20)  #
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--number', type=int, default=1)
    # AdamW
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    # # SGD
    # parser.add_argument('--lr', type=float, default=1e-2)
    # parser.add_argument('--wd', type=float, default=4e-5)

    # 预热轮次，一般设置为总轮次的5%-20%
    parser.add_argument('--warmup_epochs', type=int, default=15)
    # 数据集参数
    parser.add_argument('--data_path', type=str,
                        default=r"E:\01_User\Dengyuxin\01_dataset\03_去噪—预处理—分割2s—有效音频_STFT语谱图_44100_384")

    # 预训练权重路径、模型保存路径、log文件路径。
    parser.add_argument('--weights', type=str, default=r'E:\01_User\Dengyuxin\07_论文提交程序\03 Swin_Transformer\swin_base_patch4_window12_384_22k.pth',
                        help='initial weights path')
    parser.add_argument('--model_save_path', type=str,
                        default=r"E:\01_User\Dengyuxin\07_论文提交程序\05 Result\06 Swin_Transformer\weights\STFT",
                        help='weights save path')
    parser.add_argument('--log_dir', type=str,
                        default=r"E:\01_User\Dengyuxin\07_论文提交程序\05 Result\06 Swin_Transformer\runs\STFT")
    parser.add_argument('--report_directory', type=str,
                        default=r"E:\01_User\Dengyuxin\07_论文提交程序\05 Result\06 Swin_Transformer\reports\STFT")

    # 迁移学习是否冻结参数
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--patience', type=int, default=150, help='设置为 epochs 时不使用 patience')

    opt = parser.parse_args()

    # 记录开始时间
    start_time = time.time()
    # 训练
    train(opt)
    train_time = time.time()
    print(f'训练用时：{int(train_time - start_time)}s')
    # 推理
    predict(opt)
    predict_time = time.time()
    print(f'总用时：{int(predict_time - start_time)}s')