# 方案
将第c类图像数据(BS, W, H, C)变成(BS, D), 使用该项目将所有数据压缩成M个码本，每个码本有K个D维的码字。
最终将(M * K)个码字用于训练下游模型。

# 实验设计
## 探索实验
###  训练效应是否可逐级压缩
把每个真实样本先映射成一个向量，再把这个向量喂给 Qinco 训练。

需要验证的是：
1. 第一层码本是否解释大部分量
2. 后续码本是否在补余量

需要得到
1. 每层码本带来的 marginal gain
2. 各层码字对应真实样本的最近邻可视化
3. 只用前 m 个码本时的 residual norm
4. explained variance / explained norm ratio

预期结果
前 1 个码本解释了大部分 action variance
后 1-2 个码本主要补边界/难样本/长尾模式

### prefix 可用性
训练一个 M 个码本的模型，然后分别评估：

只用第 1 个码本
用前 2 个码本
用前 3 个码本
...
用全部 M 个码本

预期结果：
前缀越长，action residual 单调下降
第一个码本收益最大
后续码本收益变小但更偏难模式/细节
prefix 长度和最终 student acc 有相关性

# 修改
## VAE
由于图像展平后维度过高，会导致梯度爆炸

# 实施命令
## 1) ImageNet 按类转向量
将完整 ImageNet 根目录中的图像按 `spec` 过滤后，按类导出到子目录。

```bash
python scripts/convert_imagenet_to_vecs.py \
  --imagenet-root /path/to/imagenet \
  --misc-dir /data/wlf/IGD/misc \
  --spec nette \
  --split train \
  --image-size 256 \
  --out-dir data/imagenet_vectors_nette
```

说明：
- `spec` 支持：`nette`、`woof`、`1k`、`100`
- `split` 支持：`train`、`val`、`all`
- 默认 `image-size=256`
- 可选 `--max-samples N` 做快速试跑
- 结果结构：`<out-dir>/<class-name>/vectors.npy` + 每张图的 `*.npy`

## 2) 训练并导出主码本
使用新增入口 `run_export_codebooks.py` 训练。训练结束后会自动导出主码本和可视化。

```bash
python run_export_codebooks.py \
  task=train \
  db=none \
  trainset=data/imagenet_vectors_nette/n01440764/vectors.npy \
  output=results/imagenet_nette_qinco_n01440764.pt
```

## 3) 产物位置
输出目录为 `output` 对应 `.pt` 文件父目录下的类名子目录（自动推断），包含：
- 模型权重：`*.pt`
- 主码本张量：`<class-name>/codebooks_main.pt`
- 每层码本可视化：`<class-name>/codebook_step*.png`

