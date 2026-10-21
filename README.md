# Kaggle 叶子分类（Classify Leaves）

Kaggle 图像分类入门比赛：[Classify Leaves](https://www.kaggle.com/c/classify-leaves)。
任务：把 27,000+ 张叶子照片分到 **176 个植物类别**（训练集 18,353 张 / 测试集 8,800 张）。

## 方法与结果

| 方案 | 模型 | 说明 | 验证集准确率 |
|------|------|------|-------------|
| `fine-tuning.ipynb` | ResNet-18（ImageNet 预训练） | d2l 13.2 节迁移学习范式：冻结主干用小学习率微调，fc 层 10× 学习率，80/20 划分 | 见 notebook 输出 |
| `simple-resnet-baseline-noconflict.ipynb` | ResNet-34 | 独立重写版：完整 Dataset/DataLoader + 训练循环，50 epoch 保存最优权重 | **0.825** |
| `autogluon.ipynb` | AutoGluon | 自动化基线对照 | — |

## 文件说明

```
├── download_leaves_data.py              # Kaggle 数据集 16 线程分块下载器（需 KAGGLE_API_TOKEN）
├── fine-tuning.ipynb                    # d2l 风格迁移学习（ResNet-18 微调）
├── simple-resnet-baseline.ipynb         # ResNet-34 baseline（无输出的干净模板）
├── simple-resnet-baseline-noconflict.ipynb  # 同上的完整执行版（含训练日志）
├── autogluon.ipynb                      # AutoGluon 对照
├── train.csv / test.csv / sample_submission.csv   # 比赛数据清单
└── submission.csv                       # 提交文件示例
```

## 复现步骤

1. 设置环境变量 `KAGGLE_API_TOKEN`（kagglehub 的 token），运行
   `python download_leaves_data.py` 下载 `classify-leaves.zip` 并解压出 `images/`；
2. 依次运行 `fine-tuning.ipynb` 或 `simple-resnet-baseline-noconflict.ipynb`；
3. 生成 `submission.csv` 后提交到 Kaggle。

环境：PyTorch 2.x + CUDA（RTX 5060），torchvision 预训练权重。

## 致谢

方法论与代码风格参考《动手学深度学习》（[d2l-zh](https://github.com/d2l-ai/d2l-zh)）13.2 节微调实战。
