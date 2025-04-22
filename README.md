# VGG16 复现 - PaddlePaddle 实现（支持 CIFAR10 / 自定义数据集）

本项目使用 **PaddlePaddle** 深度学习框架，**完整复现经典卷积神经网络 VGG16** 架构，支持在 CIFAR10、ImageNet 以及自定义数据集上进行训练与评估。

项目结构清晰，模块解耦，具备以下特性：

- ✅ 纯手写实现 VGG16 网络结构（非调用预训练模型）
- ✅ 支持多种图像分类数据集（通过命令行参数指定）
- ✅ 支持断点保存与加载
- ✅ 结构可扩展，可集成更复杂训练流程

---

## 📦 项目结构

```
vgg16_cifar10_flexible/
├── model.py          # VGG16 模型定义
├── dataset.py        # 数据加载与 transform 定义
├── train.py          # 通用训练脚本（支持 CLI 参数）
├── evaluate.py       # 模型验证脚本
├── utils.py          # 保存/加载模型工具函数
├── config.py         # 训练默认配置项（可选）
└── README.md
```

---

## 🚀 快速开始

### 1. 安装 PaddlePaddle

```bash
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

### 2. 训练模型（默认使用 CIFAR10）

```bash
python train.py
```

或者使用命令行参数控制：

```bash
python train.py \
    --dataset cifar10 \
    --input_size 32 \
    --num_classes 10 \
    --epochs 20 \
    --lr 0.001 \
    --batch_size 64
```

### 3. 自定义数据集使用方式（建议自行扩展 `dataset.py`）：

```python
# dataset.py 中添加自定义数据集逻辑
if name == 'custom':
    return CustomDataset(...)
```

---

## 📊 模型验证

训练过程中会自动在验证集上进行评估，查看准确率。

---

## 💾 模型保存

训练每个 epoch 后，会自动将模型参数保存至 `./checkpoints/` 目录下：

- `vgg16_epochX.pdparams`: 模型参数
- `vgg16_epochX.pdopt`: 优化器状态

---

## 🔍 单图预测（可选）

你可以在 `predict.py` 中加载模型并进行图像预测，具体见脚本说明。

---

## 📚 关于 VGG16

VGG16 是由 Oxford VGG 团队提出的经典 CNN 架构，特点是使用多个 3x3 卷积堆叠代替大卷积核，结构简单但效果出色，广泛应用于图像分类、目标检测等任务。

本项目复现版本遵循原始结构（13个卷积层 + 3个全连接层），使用全局平均池化简化模型计算。

---

## ✅ 示例截图（CIFAR-10）

```
[Epoch 3] Batch 100, Loss: 1.4253
Validation Accuracy: 0.6324
```

---

## 📌 TODO（可选扩展）

- [ ] 加入 TensorBoard 可视化
- [ ] 加入 EarlyStopping / 学习率调度器
- [ ] 支持多GPU训练（DataParallel）
- [ ] 更丰富的数据增强

---

## 🧠 作者 

作者：Yuxu Ge（葛于旭）  
框架：基于 PaddlePaddle 2.x  
数据集：CIFAR10 

---

## 📝 License

MIT