# Results

## 1. Problem Setup

本实验面向二维翼型 CFD 代理建模，目标是学习如下映射：

`geometry representation + flow conditions + query coordinates -> local flow fields + surface pressure coefficient + global aerodynamic coefficients`

在当前工程中，这个任务被具体化为：

- 输入
  - 几何表示
  - 流动工况
  - 查询点坐标 `(x, y)`
- 输出
  - 查询点流场变量 `[u, v, p, rho]`
  - 表面压力系数 `Cp`
  - 全局升阻力系数 `Cl, Cd`

本轮实验的主目标不是追求最终工业级精度，而是验证以下完整链路已经能够在真实原始数据上闭环运行：

1. 原始 CFD 数据读取
2. 数据标准化与切分
3. 神经算子训练
4. test / unseen geometry 评测
5. 结果可视化与误差分析

## 2. Engineering Pipeline

### 2.1 Code Organization

本项目围绕一个标准工程链路组织，而不是围绕 notebook：

- 数据准备：[scripts/prepare_dataset.py](/Users/jason/Documents/CFD/scripts/prepare_dataset.py)
- 训练入口：[scripts/train.py](/Users/jason/Documents/CFD/scripts/train.py)
- 评测入口：[scripts/evaluate.py](/Users/jason/Documents/CFD/scripts/evaluate.py)
- 数据模块：[src/cfd_operator/data](/Users/jason/Documents/CFD/src/cfd_operator/data)
- 模型模块：[src/cfd_operator/models](/Users/jason/Documents/CFD/src/cfd_operator/models)
- 训练器：[src/cfd_operator/trainers/trainer.py](/Users/jason/Documents/CFD/src/cfd_operator/trainers/trainer.py)
- 评测器：[src/cfd_operator/evaluators/evaluator.py](/Users/jason/Documents/CFD/src/cfd_operator/evaluators/evaluator.py)

### 2.2 Model Principle

当前主模型为 DeepONet 风格算子代理：

- `branch net`
  - 编码几何和流动工况
- `trunk net`
  - 编码查询点坐标
- `field head`
  - 输出局部流场
- `scalar head`
  - 输出 `Cl, Cd`

对应实现位于：
- [deeponet.py](/Users/jason/Documents/CFD/src/cfd_operator/models/deeponet.py)

当前默认模型参数量约为 `80,326`，属于轻量级研究原型。

## 3. Data Source and Representation

### 3.1 Raw Data Source

本轮实验使用的真实数据来源于：

- [AirfRANS_original.tar](/Users/jason/Documents/CFD/outputs/data/AirfRANS_original.tar)

这是一个原始数据包，不是官方预处理后的 `Dataset.zip`。其内部结构为：

```text
AirfRANS_original/
  dataset/
    samples/
      sample_xxxxxxx/
        scalars.csv
        meshes/
          mesh_000000000.cgns
```

其中：

- `scalars.csv`
  - 保存全局标量
  - 包括 `C_D`, `C_L`, `angle_of_attack`, `inlet_velocity`
- `mesh_000000000.cgns`
  - 保存二维网格与场变量
  - 本实验实际读取的字段包括：
    - `CoordinateX`
    - `CoordinateY`
    - `VertexFields/Ux`
    - `VertexFields/Uy`
    - `VertexFields/p`
    - `VertexFields/implicit_distance`

### 3.2 Global Dataset Statistics

对全部 `1000` 个样本扫描后，标量分布如下：

- `Cl`
  - min = `-0.5336`
  - p50 = `0.6844`
  - p95 = `1.5849`
  - max = `1.8933`
- `Cd`
  - min = `0.0069`
  - p50 = `0.0107`
  - p95 = `0.0231`
  - max = `0.0459`
- `AoA`
  - min = `-4.94 deg`
  - p50 = `3.999 deg`
  - p95 = `13.513 deg`
  - max = `14.926 deg`
- `inlet_velocity`
  - min = `31.283`
  - p50 = `62.439`
  - p95 = `89.976`
  - max = `93.592`

从 `inlet_velocity / 340` 的粗略估算看，这批数据主要处于低速到中低速亚声速区间。

## 4. Data Processing and Split Strategy

### 4.1 Raw-to-NPZ Conversion

为了让原始 `CGNS` 数据能够进入现有代理模型工程，本轮新增了 raw 转换器：

- [airfrans_original.py](/Users/jason/Documents/CFD/src/cfd_operator/data/airfrans_original.py)

转换流程如下：

1. 从 `tar` 中流式读取每个样本
2. 读取 `scalars.csv`
3. 读取 `CGNS` 顶点坐标与顶点场
4. 基于 `implicit_distance` 识别翼型表面附近点
5. 采样 query points 和 surface points
6. 构造统一的训练字段：
   - `branch_inputs`
   - `query_points`
   - `field_targets`
   - `surface_points`
   - `surface_cp`
   - `scalar_targets`
7. 保存为统一 `NPZ`

本轮生成的全量训练文件为：

- [airfrans_original_full.npz](/Users/jason/Documents/CFD/outputs/data/airfrans_original_full.npz)

其配置为：

- 样本数：`1000`
- 每个样本 `128` 个 query points
- 每个样本 `64` 个 surface points

### 4.2 A Concrete Sample Through the Pipeline

为了说明原始数据如何进入训练，下面选取一条具体样本：

- 原始样本：`sample_000000006`

其原始 `scalars.csv` 为：

```csv
C_D,C_L,angle_of_attack,inlet_velocity
9.291170125372734401e-03,1.032352493088077583e-01,1.614429558094754649e-02,3.167800000000000082e+01
```

对应物理量为：

- `Cd = 0.00929`
- `Cl = 0.10324`
- `AoA ≈ 0.925 deg`
- `inlet_velocity = 31.678`

转换到训练集后，该样本对应：

- `source = airfrans_original:sample_000000006`

转换后的关键字段为：

- `flow_conditions = [0.09317, 0.92500, 31.67800]`
  - 对应 `Mach` 近似、`AoA(deg)`、`Reynolds proxy`
- `scalar_targets = [0.103235, 0.009291]`
  - 对应 `[Cl, Cd]`
- `query_point_0 = [0.01323, -0.01762]`
- `field_target_0 = [0.50218, -0.31284, 0.19500, 1.0]`
- `surface_point_0 = [0.00015, 0.00201]`
- `surface_cp_0 = 0.82790`

### 4.3 Nondimensionalization

这一轮最重要的质量提升在于 raw 场变量的无量纲化：

- `u, v`
  - 用来流速度归一化
- `p`
  - 用动态压做无量纲化
- `rho`
  - 当前使用常数近似 `1.0`

这一步非常关键，因为 raw 数据中的压力量级原本跨度很大。改进后，全量训练集的 `field_targets` 统计量变为：

- min = `-26.77`
- max = `4.86`
- mean = `0.406`
- std = `0.850`

相比前一版 raw 转换，场变量尺度已经显著干净。

### 4.4 Geometry Encoding

当前 raw 版本的几何编码策略是：

1. 从近壁点中抽取 `64` 个表面点
2. 按 chord 做几何归一化
3. 将 `64 x 2` 的表面点展平
4. 再拼接 `Mach` 和 `AoA`

因此 branch 输入维度为：

- `64 * 2 + 2 = 130`

### 4.5 Split Design

本轮还修复了原始数据上的分组逻辑，不再只做随机切分。

当前全量 split 为：

- `train = 594`
- `val = 127`
- `test = 128`
- `test_unseen_geometry = 150`
- `test_unseen_condition = 1`

解释如下：

- `test`
  - seen pool 上的常规测试集
- `test_unseen_geometry`
  - 基于几何 ID holdout 的未见翼型测试集
- `test_unseen_condition`
  - 由于原始数据中“同几何多工况重复”不足，目前只有 `1` 个样本

因此，本轮有意义的泛化评测重点是：

- `test`
- `test_unseen_geometry`

而 `test_unseen_condition` 当前不具统计意义。

## 5. Experimental Setup

### 5.1 Training Setup

训练入口：

- [scripts/train.py](/Users/jason/Documents/CFD/scripts/train.py)

本轮全量实验使用：

- 数据：`airfrans_original_full.npz`
- 模型：DeepONet
- epoch：`8`
- batch size：`16`
- 训练模式：纯监督
- physics-informed loss：关闭

训练结果目录：

- [airfrans_original_full_run_v2](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2)

### 5.2 Evaluation Setup

评测入口：

- [scripts/evaluate.py](/Users/jason/Documents/CFD/scripts/evaluate.py)

本轮实际评测了：

1. `test`
2. `test_unseen_geometry`

评测输出目录：

- [test](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2/eval/test)
- [test_unseen_geometry](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2/eval/test_unseen_geometry)

自动生成内容包括：

- 指标 JSON
- Markdown 报告
- 损失曲线
- `Cl/Cd` 散点图
- `Cp` 对比图
- 2D 场真值/预测图

## 6. Training Dynamics

训练过程中 loss 的主要变化为：

- epoch 1
  - `train_total = 5.359`
  - `val_total = 5.667`
  - `train_field = 2.668`
  - `val_field = 2.563`
- epoch 4
  - `train_total = 3.421`
  - `val_total = 3.978`
  - `train_field = 1.104`
  - `val_field = 1.292`
- epoch 8
  - `train_total = 3.200`
  - `val_total = 3.558`
  - `train_field = 0.785`
  - `val_field = 0.912`

这表明：

- 场变量误差明显下降
- 标量误差同步下降
- 训练过程稳定，没有出现明显发散

## 7. Quantitative Results

### 7.1 Test Split

来自 [metrics.json](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2/eval/test/metrics.json)：

- `field_mse = 0.1816`
- `field_rmse = 0.4262`
- `field_relative_error = 0.4199`
- `cp_mae = 1.1102`
- `cp_relative_error = 1.0009`
- `cl_mae = 0.1299`
- `cl_relative_error = 0.1724`
- `cd_mae = 0.001406`
- `cd_relative_error = 0.1638`

### 7.2 Unseen Geometry Split

来自 [metrics.json](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2/eval/test_unseen_geometry/metrics.json)：

- `field_mse = 0.1229`
- `field_rmse = 0.3506`
- `field_relative_error = 0.3847`
- `cp_mae = 0.8698`
- `cp_relative_error = 1.0007`
- `cl_mae = 0.1245`
- `cl_relative_error = 0.1868`
- `cd_mae = 0.001015`
- `cd_relative_error = 0.1126`

## 8. Visualization

### 8.1 Test Figures

- [loss_curve.png](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2/eval/test/loss_curve.png)
- [cl_scatter.png](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2/eval/test/cl_scatter.png)
- [cd_scatter.png](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2/eval/test/cd_scatter.png)
- [cp_00.png](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2/eval/test/cp_00.png)
- [field_pred_00.png](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2/eval/test/field_pred_00.png)

![loss](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2/eval/test/loss_curve.png)

![cp](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2/eval/test/cp_00.png)

![field](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2/eval/test/field_pred_00.png)

### 8.2 Unseen Geometry Figures

- [report.md](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2/eval/test_unseen_geometry/report.md)
- [cp_00.png](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2/eval/test_unseen_geometry/cp_00.png)
- [field_pred_00.png](/Users/jason/Documents/CFD/outputs/airfrans_original_full_run_v2/eval/test_unseen_geometry/field_pred_00.png)

## 9. Discussion

### 9.1 What Improved

与上一版 raw 转换相比，本轮最重要的提升是数据表示质量。

一方面，场变量无量纲化显著提升了训练稳定性。此前 `field_targets` 跨越多个数量级，导致场监督误差主导训练；本轮处理后，场变量尺度被压缩到合理范围，模型能更专注于结构性关系而不是数值量级。

另一方面，几何分组逻辑被修复，`test_unseen_geometry` 终于可以真正生成出来。这使得本轮实验不再只是 IID test，而是能够初步评估模型在未见翼型上的泛化能力。

### 9.2 What the Metrics Suggest

这轮实验最可靠的结论是：

- `Cl` 和 `Cd` 已经进入可学习区间
- 局部场变量已经可以学到稳定趋势
- `Cp` 仍然是当前最难的部分

从 test 指标看：

- `cl_relative_error ≈ 0.17`
- `cd_relative_error ≈ 0.16`

这说明全局气动系数已经具备了研究原型级别的可用性。

从场变量看：

- `field_relative_error ≈ 0.42`

这说明局部场已经能被模型部分重建，但距离高保真代理仍有明显差距。

最薄弱的是 `Cp`：

- `cp_relative_error ≈ 1.0`

这意味着当前表面压力分布的恢复质量仍不足。

### 9.3 Why Cp Is Still Weak

当前 `Cp` 的瓶颈主要来自数据构造层，而不是网络层：

- 表面点来自 `implicit_distance` 近壁近似抽样
- `Cp` 参考压力由近似 farfield 压力估计给出
- 原始数据中没有直接给出更稳的表面离散结构给当前转换器使用

因此，`Cp` 的误差高并不意味着 DeepONet 本身无效，而是说明原始数据到表面监督信号的映射还需进一步改进。

### 9.4 Interpretation of Unseen Geometry Results

本轮 `unseen geometry` 的结果没有比 `test` 更差，甚至略好：

- `field_relative_error`
  - test: `0.4199`
  - unseen geometry: `0.3847`

这不应该被简单解读为“泛化优于 IID”。更合理的解释是：

- 当前 raw 几何 signature 仍较粗糙
- `test` 集与 `unseen geometry` 集的样本难度分布并不完全一致
- 当前数据构造方式还没有把几何难度严格排序

因此，这个结果说明：

- 分组机制已经生效
- 但几何编码和 split 难度控制仍有继续改进空间

## 10. Limitations

本轮实验已经完成了真实原始数据上的完整闭环，但仍有明确限制：

1. `rho` 当前仍是常数近似
2. `Cp` 构造仍是近似方案
3. `test_unseen_condition` 只有 `1` 个样本，不具统计意义
4. 本轮未启用 physics-informed 训练
5. 当前结果属于研究原型验证，不应解释为工业级 CFD surrogate 精度

## 11. Conclusion

本轮实验的核心结论是：

1. 原始 `AirfRANS_original.tar` 已经被成功接入当前代理模型工程。
2. 工程链路已经在真实原始数据上完成了：
   - 原始数据读取
   - 全量转换
   - 训练
   - test 评测
   - unseen geometry 评测
   - 可视化输出
3. 经过无量纲化与分组修复后，模型在：
   - 局部场重建
   - `Cl/Cd` 预测
   上已经表现出稳定的可学习性。
4. 当前最主要的短板集中在 `Cp` 构造质量，而不是模型参数量本身。

## 12. Next Steps

如果继续推进，优先建议如下：

1. 优化 raw 表面点与压力参考构造，提高 `Cp` 质量
2. 继续强化几何 canonicalization，提升 `unseen geometry` split 的可信度
3. 在全量数据上重新启用 physics-informed loss 做对比实验
4. 提高 query / surface 采样分辨率，验证结果是否继续改善
5. 在数据表示稳定后，再评估是否需要更强的 operator backbone

