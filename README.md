# CNN + 单向 RNN：IMU 睡眠分期

独立的 PyTorch 基线。六轴 IMU 经共享 CNN 提取窗口特征，再用普通单向 tanh RNN 对当前窗口四分类。默认映射为 `0=Wake, 1=Light, 2=Deep, 3=REM`；原始标签语义及 PSG 同步尚未核实，不能把内部测试通过当作原始标注核验。

## 环境与入口

已验证环境：Windows，Python 3.9，`D:/Anaconda/envs/attnsleep/python.exe`，PyTorch 2.7.1+cu118，NumPy 2.0.2，RTX 3060 Laptop GPU（6 GB）。以下命令在本项目目录执行；若没有激活该环境，将 `python` 替换成该解释器完整路径。`requirements.txt` 固定 Python 包版本；CUDA 构建需与机器匹配，当前环境无需重装 torch。

```powershell
python -m pytest -q
python train_loso.py --smoke --device cuda --output-dir results/smoke_gpu
python train_loso.py --smoke --device cuda --use-class-weight --output-dir results/weighted_smoke_gpu
python train_loso.py --smoke --device cuda --use-sqrt-class-weight --output-dir results/sqrt_weighted_smoke_gpu
python train_loso.py --test-subject LIHongmei --device cuda --output-dir results/single
python -u train_loso.py --all-subjects --device cuda --output-dir results/loso_live
python -u train_loso.py --all-subjects --device cuda --use-class-weight --output-dir results/loso_weighted
python -u train_loso.py --all-subjects --device cuda --use-sqrt-class-weight --output-dir results/loso_sqrt_weighted
python -u train_loso.py --all-subjects --device cuda --output-dir results/loso_live --resume
python predict.py --checkpoint results/loso/LIHongmei/best.pt --data-root ../split-data --subjects LIHongmei --output-csv results/inference.csv --device cuda
```

`--data-root` 默认指向相邻 `split-data`，不依赖调用时的相对目录。首次全量运行会检查并哈希所有窗口，建立约 3.5 GB 的 float32 磁盘映射缓存。`--cache-dir` 默认 `results/cache`。缓存写入本项目，不修改源数据。空间紧张时可在没有运行训练的情况下删除派生缓存，下一次会重建。

`--smoke` 默认选择排序后第一人测试，只训练一轮；从每位训练者整晚等距选 8 个目标，验证、测试各最多 32 个目标。归一化仍仅使用全部训练受试者。也可同时指定 `--all-subjects` 检查所有折。冒烟结果仅验证流程，不能用作正式性能结论。

训练控制台参照同工作区 DeepSleepNet 的显示方式：每个 batch 通过 `tqdm` 动态更新当前/平均 loss、累计准确率、学习率、速度和预计剩余时间；验证与测试分别显示窗口编码和预测进度。每轮结束打印训练 Loss/Accuracy、验证 Loss/Accuracy/Balanced Accuracy/Macro-F1/Kappa、最佳轮次与早停计数。每折测试完成后打印各阶段 Precision、Recall（Sensitivity）、Specificity、F1、Support 和混淆矩阵；全部折结束再显示折间均值与合并指标。建议通过 `python -u` 运行，确保 PowerShell 及时刷新输出。

## 数据与时序协议

- 输入目录为 `受试者/10s-step/npy_N.npy`，按数字编号排序。列 0 是时间，1–6 是六轴，7 是 sleep；其他列不作为输入。独立推理目前同样接收这一带标签格式，标签仅写入输出及计算指标，不作为模型特征。
- 窗口长 30 秒、步长 10 秒、采样率 125 Hz。仅允许 3749、3750、3751 点；少一点复制最后值、多一点截尾。六轴/时间须有限，时间间隔须为 0.008 秒（绝对容差 `1e-5`），标签须是 0–3 的有限整数。
- 目标标签采用整个原窗口多数表决，并列取最先出现值。训练数据统计使用修正长度之前的六轴原始采样点；重叠部分按各窗口出现次数计入统计，延续窗口数据口径。
- 每个目标使用当前及此前最多 20 窗，不包含未来窗口。编号缺失或相邻起点偏离 10 秒超过 0.02 秒时开始新段。序列不跨受试者或断点；每段开始使用实际短序列。
- 模型在 CNN/BatchNorm 之前剔除填充窗口，RNN 使用 packed sequences，只监督最后有效位置。预测可在当前 30 秒窗口结束后生成；这里的“因果”不表示能在窗口开始时输出。
- 每个测试窗口只预测一次，包含每晚开头。重叠窗口并非统计独立的 30 秒 PSG epochs，不能与旧版重复计分或丢弃边界的结果直接比较。
- 不自动剔除恒定轴、不滤波、不改标、不重采样类别。

## 训练和复现

受试者名称采用 Python 默认区分大小写排序。每折一人测试，循环取排序后下一人为验证，其余人训练；13 人数据为 11/1/1。每折保存明确划分。

归一化仅使用该折训练受试者的通道均值和标准差；标准差小于 `1e-8` 的通道使用 1。所有训练目标每轮遍历一次，随机打乱；不做前缀截取或均衡采样。

类别加权只由该折完整训练受试者统计；验证和测试标签不参与。`--use-class-weight` 使用完整逆频率权重 `N / (4 * n_c)`；`--use-sqrt-class-weight` 使用弱化权重 `sqrt(N / (4 * n_c))`，两者互斥。完整逆频率实验提高了 Deep/REM，但合并 Macro-F1 从 `0.21635` 降至 `0.20545`，因此后续性能配置采用平方根权重。冒烟模式也用完整训练受试者统计权重，而不是只统计等距抽取的训练目标。若训练折缺少任一类别，程序会停止并明确报错。类别计数、公式、指数、权重和来源保存到每折的 `training_distribution.json`、`best.pt` 与 `result.json`。不同权重策略必须使用不同输出目录。

默认参数：序列上限 21，CNN 特征 32，单层单向 RNN 隐藏维度 32，batch 8，Adam lr `0.001`，交叉熵，梯度范数裁剪 1，seed 42，最多 30 轮，5 轮验证 Macro-F1 不提升早停，同分保留较早模型。默认 0 个 DataLoader worker、2 个 CPU 线程以控制内存。可通过 `--help` 查看参数。

开启 PyTorch 确定性算法并记录环境；相同软件、硬件和数据环境下复现，跨硬件/版本不承诺逐位一致。验证和推理的 CNN 特征仅在单次评估内缓存，模型 eval 模式，下一轮重新编码。训练始终端到端计算。

测试集只在最佳验证模型恢复后预测。`--resume` 复用已完成折，并检查配置、代码哈希、数据哈希及保存预测的指标；未完成折从 seed 42 重新训练。它不是逐 batch 断点恢复。源代码、训练参数或数据变化后应使用新输出目录。不要同时启动两个进程写入同一输出目录。

## 产物

- `run.json`：配置、源码哈希、数据摘要、环境；`manifest.json`：逐文件 SHA256 与窗口标识。
- `status.json`：当前折、轮次、训练步骤或完成/失败状态；`summary.json`、`REPORT.md`：逐折与合并结果，训练期间逐折更新。
- 每折目录：`split.json`、`normalizer.json`、`training_distribution.json`、`history.json`、`best.pt`、`predictions.csv`、`result.json`。
- CSV 包含受试者、窗口编号、首末采样时间、真实标签、预测类别及 `p0`–`p3`。`end` 是最后一个原始采样点时间，不是半开窗口右端点。
- 固定四类计算 Accuracy、Balanced Accuracy、Macro-F1、Cohen κ、每类 Precision/Recall（Sensitivity）/Specificity/F1/支持数及混淆矩阵（行真值、列预测）。固定四类平均时缺失类 Recall/F1 为 0；κ 分母为 0 时记 null。折间统计采用总体标准差，κ 忽略不可定义折并报告有效折数。合并指标从所有逐窗口预测重新计算。

数据、缓存、模型和运行结果都在 Git 忽略规则内；不自动提交训练数据。当前仓库位于父级 Git 仓库内，保持该结构。
