# 基于多特征语谱图与多模型决策级融合的鸟类声音识别系统

> 论文配套程序 | 20 类鸟声识别 | Chirplet / Mel / STFT × ResNet50 / ConvNeXt-V2 / Swin Transformer

---

## 1. 项目简介

本项目实现了完整的鸟类声音识别流水线：将原始音频预处理后转换为 **Chirplet、Mel、STFT** 三种时频语谱图，分别训练 **ResNet50、ConvNeXt-V2、Swin Transformer** 三种骨干网络，并研究 **特征级融合（Add / Concat）** 与 **决策级融合（加权平均 / D-S 证据理论 / 多数投票 / 自适应加权）** 对识别性能的提升。

- **任务**：20 类鸟类声音分类（物种清单见 [第 9 节](#9-类别清单20-类)）
- **输入**：44.1 kHz 音频 → 2 s 有效片段 → 384×384 语谱图（RGB，3 通道）
- **训练范式**：每种特征 × 每种骨干独立训练一个基线模型，共 9 个模型（权重命名 `best_{架构}_{特征}.pth`）
- **核心结果**：27 种跨架构组合、多种融合方法系统评估，验证集准确率 **97.55% ~ 98.27%**，其中组合13（ConvNeXtV2-Chirplet + ConvNeXtV2-Mel + ResNet50-STFT）在 **D-S 证据融合**下取得最高 **98.27%**

## 2. 总体流程

```
原始音频
   │  audio_processing_01.py   整理归档、剔除非音频文件
   ▼
按类别存放的音频文件
   │  audio_processing_02.py   带通滤波去噪、预处理、分割 2 s 有效音频
   ▼
有效音频片段（44.1 kHz）
   │  audio_processing_03_{chirplet|mel|stft}.py
   ▼
三种 384×384 语谱图数据集 ──┬── ResNet50        ──┐
                            ├── ConvNeXt-V2-Tiny ──┼── 特征级融合（Add / Concat）
                            └── Swin-B (384)     ──┘
                                     │
                                     ▼
                        决策级融合（加权平均 / D-S / 多数投票 / 自适应）
                                     │
                                     ▼
                    评估报告（Accuracy / Precision / Recall / F1 / 混淆矩阵 / 可视化）
```

## 3. 目录结构

```
07_论文提交程序/
├── 00 Audio Preprocessing/                      # 音频预处理与语谱图生成
│   ├── audio_processing_01.py                   #   ① 音频收集整理（清理非音频、按类归档）
│   ├── audio_processing_02.py                   #   ② 带通滤波去噪、预处理、分割 2 s 有效音频
│   ├── audio_processing_03_chirplet.py          #   ③a 生成 Chirplet 语谱图
│   ├── audio_processing_03_mel.py               #   ③b 生成 Mel 语谱图
│   ├── audio_processing_03_stft.py              #   ③c 生成 STFT 语谱图
│   └── chirplet.py                              #   快速 Chirplet 变换（FCT）实现
│
├── 01 ConvNeXt-V2/
│   ├── 01_ConvNextV2/                           # ConvNeXt-V2 单特征基线训练
│   ├── 02_ConvNextV2_特征级融合/
│   │   ├── 01_Add特征融合/                      #   特征级 Add 融合训练
│   │   └── 02_Concat特征融合/                   #   特征级 Concat 融合训练
│   └── 03_ConvNextV2_决策级融合/                # 单架构三特征决策级融合（含 models/ 权重）
│
├── 02 ResNet/
│   ├── 01 ResNet/                               # ResNet50 单特征基线训练
│   ├── 02_ResNet_特征级融合/                    #   01_Add / 02_Concat
│   └── 03_ResNet_决策级融合/                    # end.py（加权/D-S/投票）+ 多数投票专版（含 model/ 权重）
│
├── 03 Swin_Transformer/                         # Swin-B 基线训练（本地模型实现）
│
└── 04 Multi-model decision-level fusion/        # 跨架构全组合决策级融合评估（核心）
    ├── Train_MultiModelDecisionFusion.py        #   27 组合 × 多融合方法 全量评估
    ├── model/                                   #   跨架构权重（resnet50 / convnextv2 × 3 特征）
    ├── reports_all_combinations/                #   评估报告：CSV + 汇总 + 可视化
    └── reports_all_combinations_adaptive/       #   含自适应加权的评估报告
```

> 各目录中的 `my_dataset_V1R6.py` 为数据集类（含 SpecAugment 式时间/频率遮掩增强、过/欠采样支持），`early_stopping.py` 为早停模块，`utils*.py` 为数据划分与训练/评估工具，`class_indices*.json` 为类别索引。

## 4. 环境依赖

- **Python 3.8**，PyTorch（建议 CUDA 版本）
- 显卡显存建议 ≥ 8 GB（384×384 输入、batch 32；Swin-B 需求更高）

```bash
pip install torch torchvision torchaudio \
            numpy scipy librosa soundfile \
            scikit-learn matplotlib seaborn pandas \
            tqdm fvcore tensorboard pypinyin pillow
```

## 5. 数据准备与预处理（00 Audio Preprocessing）

按顺序执行三步脚本，将原始音频转为三类语谱图数据集：

| 步骤 | 脚本 | 功能 |
|---|---|---|
| ① | `audio_processing_01.py` | 复制原始数据，删除非音频文件，将音频按类别归档到统一目录 |
| ② | `audio_processing_02.py` | 带通滤波（Butterworth）去噪等预处理，并将音频**分割为 2 s 有效片段** |
| ③ | `audio_processing_03_chirplet.py` / `_mel.py` / `_stft.py` | 以 44.1 kHz 重采样后分别生成 **Chirplet / Mel / STFT** 语谱图 PNG（384×384，多进程加速） |

其中 `chirplet.py` 为快速 Chirplet 变换（FCT）实现，源自 [DYNI-TOULON/fastchirplet](https://github.com/DYNI-TOULON/fastchirplet)（Tassan 等）。

**数据集组织**（训练脚本通过 `--data-path` 指向某一特征的语谱图根目录）：

```
03_去噪—预处理—分割2s—有效音频_Mel语谱图_44100_384/     # 根目录命名约定：数据版本_特征_采样率_尺寸
├── Accipiter gentilis/          # 每类一个子文件夹（物种学名）
│   ├── xxxxx.png                # 该类的语谱图（仅支持 .png / .PNG）
│   └── ...
└── Vanellus vanellus/
```

- **数据划分**：`read_split_data_science()` 按类别目录划分 train / val / test ≈ **60% / 20% / 20%**（固定随机种子 0，保证可复现）。
- **类别索引**：脚本自动生成 `class_indices_*.json`（类别顺序由文件夹名排序决定，生成后请勿重命名文件夹）；`class_indices_en.json` 为学名版索引，供融合评估脚本输出报告使用。

## 6. 模型训练

### 6.1 基线训练命令

分别进入对应目录运行（`--data-path` 等路径参数按本机实际情况修改，**建议使用绝对路径**）：

```bash
# ConvNeXt-V2 基线（默认加载 convnextv2_tiny_22k_384_ema.pt 预训练权重）
cd "01 ConvNeXt-V2/01_ConvNextV2"
python train_V1R6_ConvNeXtV2.py --data-path <Mel语谱图数据集路径> --device cuda:0

# ResNet50 基线
cd "02 ResNet/01 ResNet"
python train_V1R6_ResNet.py --data-path <Mel语谱图数据集路径> --device cuda:0

# Swin Transformer 基线
cd "03 Swin_Transformer"
python train_V1R6.py --data-path <Mel语谱图数据集路径> --device cuda:0
```

训练输出：最优权重（`best_*.pth`）、TensorBoard 日志（`runs/`）、训练曲线（`training_curves_*`）与评估报告。

### 6.2 关键超参数

| 参数 | ResNet50 | ConvNeXt-V2-Tiny | Swin-B (384) |
|---|---|---|---|
| epochs | 150 | 150 | 100 |
| batch size | 32 | 32 | 32 |
| 学习率 | 1e-4 | 4e-4 | 5e-4 |
| weight decay | 1e-4 | 4e-2 | 5e-2 |
| warmup epochs | 10 | 15 | — |
| 优化器 | AdamW | AdamW | AdamW |
| 预训练权重 | `resnet50.pth`（torchvision 结构） | `convnextv2_tiny_22k_384_ema.pt`（ImageNet-22K） | 本地实现 `swin_base_patch4_window12_384` |
| 早停 | `--patience`（设为 epochs 值即不启用） | 同左 | 同左 |

### 6.3 数据增强

- 几何增强：随机仿射平移（p=0.3）+ 随机水平翻转（p=0.3）
- 频谱增强（`my_dataset_V1R6.py` 自定义）：**RandomTimeMasking / RandomFrequencyMasking**（SpecAugment 式时间/频率遮掩，各 1~3 段，p=0.3）
- 标准化：ImageNet mean/std
- 类别不均衡：支持 `WeightedRandomSampler` 过采样 / 欠采样

## 7. 融合模型

### 7.1 特征级融合

位于 `01 ConvNeXt-V2/02_ConvNextV2_特征级融合/` 与 `02 ResNet/02_ResNet_特征级融合/`：

- **Add 融合**：三种语谱图的特征图逐元素相加后分类
- **Concat 融合**：特征图通道维拼接后分类

### 7.2 单架构决策级融合

- `01 ConvNeXt-V2/03_ConvNextV2_决策级融合/`：`Train_convnextv2_Decision_Fusion(end).py`，加载该架构的 Chirplet / Mel / STFT 三份权重（`models/best_convnextv2_*.pth`）
- `02 ResNet/03_ResNet_决策级融合/`：`Train_ResNet50_Decision_Fusion(end).py`（加权平均 + D-S + 多数投票）与 `Train_ResNet50_Decision_Fusion(majority voting).py`（多数投票专版）、`end.py`（完整评估入口），权重位于 `model/best_resnet_*.pth`

流程：三种特征的验证/测试集概率输出 → 融合 → 输出 Accuracy / Precision / Recall / F1 / 混淆矩阵与评估报告。

### 7.3 跨架构全组合决策级融合（04，核心）

`Train_MultiModelDecisionFusion.py` 对 **Chirplet、Mel、STFT 三个特征位置分别选择 {resnet50, convnextv2, swin_transformer}**，共 **3³ = 27 种组合**逐一评估：

| 项目 | 说明 |
|---|---|
| 组合命名 | `组合XX_{Chirplet位模型}_{Mel位模型}_{STFT位模型}`，如 `组合13_convnextv2_convnextv2_resnet50` |
| 融合方法 | **加权平均**（默认权重 `0.25,0.35,0.40`，依次对应 Chirplet/Mel/STFT，可用 `--weights` 覆盖）、**D-S 证据理论**、**多数投票** |
| 自适应加权 | 自适应版额外支持 `confidence` / `entropy` / `margin` / `ensemble` 四种自适应权重策略 |
| 权重加载 | 从 `--model_save_path` 读取 `best_{架构}_{特征}.pth`（架构前缀：`resnet50` / `convnextv2` / `swin`） |
| 输入尺寸 | 384×384，`--batch_size 32` |

```bash
cd "04 Multi-model decision-level fusion"
python Train_MultiModelDecisionFusion.py \
    --chirplet_data_path <Chirplet语谱图数据集> \
    --mel_data_path      <Mel语谱图数据集> \
    --stft_data_path     <STFT语谱图数据集> \
    --model_save_path    ./model \
    --report_dir         ./reports_all_combinations_adaptive \
    --device cuda:0
```

**输出报告**（`reports_all_combinations[_adaptive]/`）：

```
├── all_combinations_detailed.csv        # 每组合 × 每融合方法 的完整指标
├── all_combinations_summary.txt         # 27 组合验证集准确率汇总与最佳方法
└── visualization/
    ├── top_10_combinations.png          # Top-10 组合排名
    ├── fusion_methods_comparison.png    # 融合方法对比
    └── test_best_accuracy_distribution.png
```

## 8. 实验结果摘要（验证集）

27 种组合 × 多种融合方法的准确率分布：**97.55% ~ 98.27%**。

| 方案 | 验证集准确率 | 说明 |
|---|---|---|
| 单模型基线（ResNet50，三特征一致） | ≈ 97.6% | 组合01，未融合 |
| 最佳组合 | **98.27%** | 组合13（ConvNeXtV2-Chirplet + ConvNeXtV2-Mel + ResNet50-STFT），**D-S 证据融合** |
| 自适应加权最佳 | 98.23% | 组合05（ResNet50-Chirplet + ConvNeXtV2-Mel + ConvNeXtV2-STFT），自适应 entropy |

主要结论：

1. **D-S 证据理论在多数组合中表现最佳**，其次是自适应 entropy 加权；
2. 跨架构异构融合（如 组合13）优于同架构重复；
3. 测试集完整指标（含每类 Precision / Recall / F1、混淆矩阵）见 `all_combinations_detailed.csv`。

## 9. 类别清单（20 类）

| # | 学名 | 中文名 | # | 学名 | 中文名 |
|---|---|---|---|---|---|
| 0 | Accipiter gentilis | 苍鹰 | 10 | Himantopus himantopus | 黑翅长脚鹬 |
| 1 | Anas crecca | 绿翅鸭 | 11 | Passer domesticus | 家麻雀 |
| 2 | Anas platyrhynchos | 绿头鸭 | 12 | Perdix perdix | 灰山鹑 |
| 3 | Anser anser | 灰雁 | 13 | Phalacrocorax carbo | 普通鸬鹚 |
| 4 | Ardea cinerea | 苍鹭 | 14 | Phasianus colchicus | 雉鸡 |
| 5 | Buteo buteo | 普通鵟 | 15 | Rallus aquaticus | 普通秧鸡 |
| 6 | Coturnix coturnix | 鹌鹑 | 16 | Tringa glareola | 林鹬 |
| 7 | Cygnus cygnus | 大天鹅 | 17 | Tringa ochropus | 矶鹬 |
| 8 | Fulica atra | 白骨顶 | 18 | Tringa totanus | 红脚鹬 |
| 9 | Gavia stellata | 红喉潜鸟 | 19 | Vanellus vanellus | 凤头麦鸡 |

## 10. 注意事项（FAQ）

1. **路径参数**：多数脚本的 argparse 默认值为作者本机的绝对/相对路径（如 `E:\...`、`'../class_indices_en.json'`），跨机器运行前务必核对 `--data-path`、`--weights`、`--model_save_path`、`--report_directory` 等参数；**相对路径随启动目录变化，建议统一改用绝对路径或始终从脚本所在目录启动**，否则会报 `FileNotFoundError`。
2. **权重命名约定**：`best_{架构}_{特征}.pth`，架构前缀为 `resnet50` / `convnextv2` / `swin`（如 `best_swin_Chirplet.pth`）。`04/model/` 收录了 ResNet50 与 ConvNeXt-V2 的权重，Swin 的三份权重请放置到同一目录下方可运行全组合评估。
3. **中文字体**：绘图使用 `SimHei`，Linux 服务器需替换为系统已有中文字体（如 `Noto Sans CJK SC`），否则中文标签显示为方块。
4. **显存不足**：优先调小 `--batch_size`；Swin-B（384 输入）显存占用最高。
5. **复现性**：数据划分与训练固定随机种子 0；更换数据集后请删除旧 `class_indices_*.json` 重新生成。
6. **评估脚本修订版**：工作区另有删除 AUC 输出的评估脚本与加固版工具（含文件排序、D-S 公式修正、三特征对齐校验等），可直接替换 `04/` 中对应文件使用，接口不变。

---

*本项目为论文配套代码，仅用于学术研究。*
