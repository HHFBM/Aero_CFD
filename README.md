# CFD Operator Surrogate

面向飞行器 CFD 加速分析的神经算子代理工程。当前仓库已经从“AirfRANS 适配的分析型输出工程”进一步整理为一个**以 AirfRANS 为默认数据适配器的通用 CFD surrogate 框架雏形**，并保持现有训练主线不被推翻：

- 不删除现有接口
- 不重写训练主线
- 继续基于 2D / 简化翼型代理主线
- 严格按 AirfRANS 实际可提供的数据设计监督与输出
- 保持 AirfRANS 作为当前默认训练/测试数据源
- 通过统一 schema / adapter / capability 降低对单一数据集的绑定

当前工程支持两条主要模型路线：

- `DeepONet` 主线
- `Geo-FNO` 对比基线

并已经打通：

- 训练
- 评测
- 推理
- analysis bundle 导出
- 可视化
- Colab notebook 运行入口

同时，当前工程结构上已经新增：

- 统一 `CFDSurrogateSample` schema
- `Synthetic / AirfRANS / AirfRANSOriginal / NPZFile` adapter
- `DatasetCapability` / `TaskRequest`
- decoder heads 抽象
- capability-aware evaluator / infer / export 兼容层
- future native geometry backbone 的接口预留与 contract metadata

## 1. 项目定位

这个项目不是工业级三维 CFD 求解器替代品，而是一个“可训练、可评测、可解释、可导出”的 2D 气动代理系统。

当前更准确的定位是：

- **默认数据源**：AirfRANS 2D 不可压缩 RANS 数据
- **默认任务**：field + scalar + surface + slice + feature 分析输出
- **当前工程目标**：在不推翻 AirfRANS 主链路的前提下，形成一个可以继续接入其他 2D CFD 数据的通用 surrogate 框架骨架

也就是说：

- 现在依然是 AirfRANS 主线工程
- 但 trainer / evaluator / infer / export 已经不再把 AirfRANS 当作唯一隐式前提
- 后续新增数据源时，优先新增 adapter，而不是重写训练主框架

适用场景：

- 二维翼型快速筛选
- 设计空间探索前置过滤
- CFD surrogate / operator learning baseline
- AirfRANS 数据工程接入与多任务分析输出验证
- 结果汇报、曲线对比和剖面分析

当前不适用的场景：

- 工业级 3D 全机高保真流场
- 热流定量设计
- 强激波专用高精度定位
- 复杂壁面摩擦阻力工程定量替代

## 2. 当前已经实现的能力

### 2.0 框架抽象

当前仓库已经具备以下通用化抽象，但默认行为仍兼容现有 AirfRANS 主路径：

- 统一样本 schema
  - [src/cfd_operator/schema.py](/Users/jason/Documents/CFD/src/cfd_operator/schema.py)
- 数据 adapter
  - [src/cfd_operator/data/adapters.py](/Users/jason/Documents/CFD/src/cfd_operator/data/adapters.py)
- dataset capability / task request
  - [src/cfd_operator/tasks/capabilities.py](/Users/jason/Documents/CFD/src/cfd_operator/tasks/capabilities.py)
- decoder heads
  - [src/cfd_operator/models/heads.py](/Users/jason/Documents/CFD/src/cfd_operator/models/heads.py)
- metric registry
  - [src/cfd_operator/evaluators/registry.py](/Users/jason/Documents/CFD/src/cfd_operator/evaluators/registry.py)

这些抽象的作用是：

- 让 AirfRANS 成为“默认 adapter”，而不是整套工程的隐式硬编码前提
- 让 trainer / evaluator / infer 优先依赖 schema + capability
- 在保持旧 payload / 旧 checkpoint 兼容的前提下，为后续接入其他 2D CFD 数据源做准备

### 2.1 数据

支持以下数据路径：

- `synthetic` toy 数据
- `airfrans` 转换数据
- `airfrans_original` 原始 AirfRANS 离线转换数据
- `file` 类型 `.npz` 数据集

当前默认主训练/测试数据路径仍然是 AirfRANS heavy 数据的统一 `.npz` 格式。

同时当前 loader 侧已经改成 adapter 风格，支持：

- `SyntheticAdapter`
- `AirfRANSAdapter`
- `AirfRANSOriginalAdapter`
- `NPZFileAdapter`

当前默认行为是：

- 旧配置不改时，仍然走 AirfRANS 主路径
- datamodule 会先通过 adapter 生成统一 schema
- 然后桥接回当前 trainer/evaluator 仍在使用的 legacy payload

这意味着：

- 当前 AirfRANS 主链路不回退
- 新增 2D CFD 数据源时，主要新增 adapter 即可

### 2.2 模型

已可用模型：

- `DeepONet`
- `Geo-FNO`

当前 `Geo-FNO` 已经不再是 placeholder，而是一个可训练、可评测的非规则点云几何感知频域基线。

标准 `FNO` 仍未作为当前 AirfRANS 主路径的正式训练模型，因为现有 `query_points` 是样本级非规则点云，不是共享规则网格。若后续要做标准 FNO，需要先统一网格化。

### 2.3 输出

当前优先真实打通的输出：

- pointwise fields
  - `u`
  - `v`
  - `p`
  - `nut`
- scalar outputs
  - `cl`
  - `cd`
- surface outputs
  - `pressure_surface`
  - `cp_surface`
- slice outputs
  - line slice 上的 `u / v / p / nut`
- feature / analysis outputs
  - `high_gradient_mask`
  - `pressure_gradient_indicator`
  - `high_gradient_region_summary`

### 2.4 推理与导出

当前推理链路已支持：

- `infer.py` 单样本推理
- FastAPI 接口
- `analysis_bundle` 文件导出
- PNG 图导出

导出文件包括：

- `predictions.json`
- `scalar_summary.json`
- `surface_values.csv`
- `slice_values.csv`
- `feature_summary.json`
- 图像文件

### 2.5 评测与测试

当前评测已支持：

- field RMSE / relative error
- `cl/cd` MAE / relative error
- `cp_surface_rmse`
- `pressure_surface_rmse`
- `slice_rmse`
- feature accuracy / IoU / F1

测试已覆盖：

- 数据与 schema
- postprocess
- infer bundle
- scalar outputs
- surface outputs
- slice outputs
- feature outputs
- loss
- physics
- smoke train / eval
- Geo-FNO smoke train

最近一次全量测试结果：

- `57 passed`

## 3. AirfRANS 适配原则

AirfRANS 当前仍然是默认数据适配器，而不是被移除或降级掉的数据路径。

当前实现严格遵守 AirfRANS 的真实边界：

- AirfRANS 是 2D、不可压缩、亚声速 RANS 数据
- 主字段围绕 `velocity / pressure / turbulent viscosity`
- 可真实监督的主线是 `u / v / p / nut / cl / cd / pressure_surface`

因此当前工程明确区分三类量。

### 3.1 真实监督 / 真实打通

- `u`
- `v`
- `p`
- `nut`
- `cl`
- `cd`
- `pressure_surface`
- `slice u / v / p / nut`

### 3.2 Derived outputs

- `cp_surface`
- `high_gradient_mask`
- `pressure_gradient_indicator`
- `high_gradient_region_summary`
- `wall_shear_surface`

### 3.3 Placeholder / TODO

- `rho`
- `heat_flux_surface`
- `shock_indicator`
- `shock_location`
- `cm`
- `cdp / cdv / clp / clv`

这些 placeholder 量当前不会被伪装成 AirfRANS 正式 benchmark 结果。

## 4. 输出的物理意义

### 4.1 Pointwise fields

- `u`
  - x 方向速度分量
- `v`
  - y 方向速度分量
- `p`
  - 压力场
- `nut`
  - 湍流运动粘性 / 涡粘性，用于刻画湍流导致的等效扩散能力

### 4.2 Scalars

- `cl`
  - 升力系数
- `cd`
  - 阻力系数

### 4.3 Surface outputs

- `pressure_surface`
  - 翼型表面压力分布
- `cp_surface`
  - 表面压力系数，定义为 `Cp = (p - p_ref) / q_ref`

### 4.4 Slice outputs

line slice 表示沿指定二维直线抽取的局部剖面，当前支持：

- `x = const`
- `y = const`
- 任意线段

### 4.5 Feature outputs

- `high_gradient_mask`
  - 高梯度区域指示
- `pressure_gradient_indicator`
  - 基于压力梯度的分析特征
- `high_gradient_region_summary`
  - 高梯度区域比例、峰值和统计总结

## 5. 模型原理

### 5.0 输出组织方式

当前没有推翻 `DeepONet` 或 `Geo-FNO` 主干，而是新增了 decoder head 抽象：

- `FieldDecoderHead`
- `SurfaceDecoderHead`
- `ScalarDecoderHead`
- `FeatureDecoderHead`

这层抽象的目的是：

- 统一不同模型的输出组织方式
- 减少“每个模型都自己手拼 fields/scalars/features”的耦合
- 为后续新增模型或输出任务提供稳定接口

注意：

- 当前 `SurfaceDecoderHead` 仍主要表示“在 surface query points 上解码场量”的组织抽象
- 它不是一条完全独立的表面专用网络主干

### 5.1 DeepONet

当前 `DeepONet` 主线是典型 branch-trunk 结构：

- branch 负责编码几何和工况
- trunk 负责编码空间位置
- 二者融合后输出局部场响应

它学习的是一个算子映射：

`geometry + flow conditions + query point -> flow response`

这种方式适合：

- 非规则点云查询
- 不同样本拥有不同空间采样点
- surface / volume / slice 共用一条主线

### 5.2 Geo-FNO

当前 `Geo-FNO` 是针对非规则 2D 点集的几何感知频域模型。

它不是标准规则网格 FNO，而是：

- 在非规则点上做低频谱基展开
- 做全局频域信息混合
- 再结合局部 MLP 和条件编码进行更新

当前实现的意义：

- 保留频域全局建模优势
- 避免强行把 AirfRANS 点云插值成固定网格
- 能与当前训练、评测、推理、surface/slice loss 直接兼容

### 5.3 当前 geometry conditioning 状态

当前仓库已经明确区分两件事：

1. **当前稳定主线**
   - `geometry -> BranchInputAdapter -> fixed branch-compatible vector -> DeepONet/Geo-FNO`
2. **未来原生几何主干预留**
   - `geometry -> native geometry backbone -> geometry latent / tokens -> future conditioning path`

当前真正启用的仍然是第一条。也就是说：

- 当前训练和推理的主契约仍是 fixed-dimension `branch_inputs`
- `encoded_geometry` 目前仍是“raw geometry -> fixed branch-compatible vector”的兼容路径
- 本仓库**还没有**完成原生可变长几何主干升级
- 本仓库**还没有**切换到 PointNet / Transformer / Mesh GNN 之类的 geometry token 主线

阶段 7 已经新增了 future native geometry backbone 的接口预留与 metadata，但它的定位非常明确：

- 默认关闭
- 不进入当前主训练主线
- 主要用于：
  - 清晰描述未来接口
  - 让 checkpoint / inference artifact 自描述 geometry backbone contract
  - 为后续真正的可变长 geometry conditioning 升级做准备

换句话说，当前系统的最准确表述是：

**已经完成了从“只有 fixed branch vector 兼容逻辑”到“fixed branch path + future native geometry backbone interface reservation”的升级，但主干训练仍然走当前稳定的 fixed branch-compatible 路线。**

### 5.4 为什么当前第四通道是 `nut` 不是 `rho`

AirfRANS 主线是不可压缩 RANS，不是可压缩 Euler。  
因此第 4 通道当前更合理地定义为 `nut`：

- `rho`
  - 密度，更偏可压缩流主变量
- `nut`
  - 湍流运动粘性，更符合 AirfRANS 可真实监督的字段

## 6. 当前 physics loss 设计

当前 physics loss 已经支持两条路径：

- 第 4 通道为 `rho`
  - 使用原可压缩 Euler residual
- 第 4 通道为 `nut`
  - 使用不可压缩 RANS 风格 proxy residual

`nut` 路径下当前包含：

- 连续性残差
  - `du/dx + dv/dy`
- 动量残差
  - 对 `u / v / p` 做不可压缩 RANS 风格约束
- `nut` 输运 proxy
  - 对湍流粘性做平滑和输运近似约束

注意：

- 当前 physics loss 是研究型代理约束，不是求解器离散格式的严格复现
- 对 AirfRANS 而言，它更适合做弱物理正则，而不是“真实方程解”

## 7. 仓库结构

```text
configs/
  data/
  experiments/
src/cfd_operator/
  config/
  data/
  evaluators/
  geometry/
  inference/
  losses/
  models/
  physics/
  postprocess/
  tasks/
  trainers/
  visualization/
scripts/
examples/
tests/
outputs/
```

关键文件：

- [configs/experiments/airfrans_original_heavy.yaml](/Users/jason/Documents/CFD/configs/experiments/airfrans_original_heavy.yaml)
- [configs/experiments/airfrans_geofno_heavy.yaml](/Users/jason/Documents/CFD/configs/experiments/airfrans_geofno_heavy.yaml)
- [scripts/train.py](/Users/jason/Documents/CFD/scripts/train.py)
- [scripts/evaluate.py](/Users/jason/Documents/CFD/scripts/evaluate.py)
- [scripts/infer.py](/Users/jason/Documents/CFD/scripts/infer.py)
- [src/cfd_operator/models/deeponet.py](/Users/jason/Documents/CFD/src/cfd_operator/models/deeponet.py)
- [src/cfd_operator/models/fno.py](/Users/jason/Documents/CFD/src/cfd_operator/models/fno.py)
- [src/cfd_operator/models/heads.py](/Users/jason/Documents/CFD/src/cfd_operator/models/heads.py)
- [src/cfd_operator/schema.py](/Users/jason/Documents/CFD/src/cfd_operator/schema.py)
- [src/cfd_operator/data/adapters.py](/Users/jason/Documents/CFD/src/cfd_operator/data/adapters.py)
- [src/cfd_operator/tasks/capabilities.py](/Users/jason/Documents/CFD/src/cfd_operator/tasks/capabilities.py)
- [src/cfd_operator/postprocess/analysis.py](/Users/jason/Documents/CFD/src/cfd_operator/postprocess/analysis.py)

## 8. 安装

推荐 Python `3.10+`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

如果要使用 AirfRANS 数据转换：

```bash
pip install ".[data]"
```

## 9. 数据准备

### 9.0 当前数据流

当前统一的数据流可以概括为：

`raw dataset / npz / tabular file -> adapter -> unified CFDSurrogateSample -> legacy payload bridge -> trainer/evaluator`

这样做的意义是：

- 保留 AirfRANS 当前主训练链路
- 不要求用户重导已有 `.npz`
- 让未来接入新数据源时主要修改 adapter，而不是重写 trainer/evaluator

### 9.1 生成 toy 数据

```bash
python scripts/prepare_dataset.py --config configs/default.yaml
```

### 9.2 转换 AirfRANS

```bash
python scripts/prepare_dataset.py \
  --config configs/default.yaml \
  --override data.dataset_type=airfrans_original \
  --override data.dataset_path=outputs/data/airfrans_original_heavy.npz
```

### 9.3 当前主对比数据切分

当前统一对比主要使用：

- [outputs/data/airfrans_original_heavy_resplit.npz](/Users/jason/Documents/CFD/outputs/data/airfrans_original_heavy_resplit.npz)
- [outputs/data/airfrans_original_heavy_resplit.json](/Users/jason/Documents/CFD/outputs/data/airfrans_original_heavy_resplit.json)

切分口径：

- 总样本：`1000`
- train：`700`
- val：`150`
- test：`150`

### 9.4 Generic file dataset 示例

如果你想接入非 AirfRANS 的 2D CFD 样本，当前推荐先走 `dataset_type=file` 路径。

示例文件与说明：

- [examples/generic_file_dataset_example.csv](/Users/jason/Documents/CFD/examples/generic_file_dataset_example.csv)
- [examples/generic_file_dataset_example.md](/Users/jason/Documents/CFD/examples/generic_file_dataset_example.md)

当前支持两类方式：

- 直接提供预计算 `branch_*` 列
- 提供 `geometry_points` / `geometry_x, geometry_y`，由 dataloader 根据 `branch_feature_mode` 自动派生 `branch_inputs`

这部分已经能用于 generic 2D geometry 的训练数据适配，但仍然保持与现有固定维度 `branch_inputs` 主线兼容。

重要说明：

- 当前 generic 2D geometry 接入已经能走训练/评测/推理的 smoke 级主路径
- 但它仍然主要通过 `BranchInputAdapter` 落到 fixed-dimension `branch_inputs`
- 仓库虽然已经有 future native geometry backbone 的接口预留，但这**不代表**当前 generic geometry 已经默认使用原生可变长几何主干
- 如果后续真的要上 PointNet / Transformer / Mesh GNN，仍需要单独完成：
  - variable-length geometry batching
  - geometry latent / token conditioning 契约
  - trainer/inference 的原生 geometry backbone 主路径

## 10. 训练

### 10.1 DeepONet 训练

```bash
python scripts/train.py \
  --config configs/experiments/airfrans_original_heavy.yaml
```

### 10.2 Geo-FNO 训练

```bash
python scripts/train.py \
  --config configs/experiments/airfrans_geofno_heavy.yaml
```

### 10.3 常用 override 示例

```bash
python scripts/train.py \
  --config configs/experiments/airfrans_original_heavy.yaml \
  --override experiment.name=my_run \
  --override train.epochs=12 \
  --override loss.use_physics=false \
  --override loss.physics_weight=0.0
```

训练输出默认位于：

```text
outputs/<experiment.name>/
  checkpoints/
  logs/
  reports/
  eval/
```

## 11. 评测

```bash
python scripts/evaluate.py \
  --config configs/experiments/airfrans_original_heavy.yaml \
  --checkpoint outputs/<run>/checkpoints/best.pt
```

Geo-FNO 评测：

```bash
python scripts/evaluate.py \
  --config configs/experiments/airfrans_geofno_heavy.yaml \
  --checkpoint outputs/<geofno_run>/checkpoints/best.pt
```

评测输出包括：

- `metrics.json`
- `report.md`
- `report.json`
- loss curve
- `cl/cd` scatter
- sample 级 `surface_cp.png`
- `surface_pressure.png`
- `slice_*.png`
- `high_gradient_regions.png`

## 12. 推理与 analysis bundle

当前推理与导出不再隐式要求“数据一定来自 AirfRANS”，而是基于：

- geometry semantics
- output semantics
- dataset capability

来决定：

- 哪些输出可以预测
- 哪些指标可以评测
- 哪些文件可以导出
- 哪些项应优雅跳过

示例输入：

- [examples/inference_input.json](/Users/jason/Documents/CFD/examples/inference_input.json)

推理命令：

```bash
python scripts/infer.py \
  --checkpoint outputs/<run>/checkpoints/best.pt \
  --input examples/inference_input.json \
  --output outputs/<run>/inference.json \
  --export-dir outputs/<run>/analysis_bundle
```

analysis bundle 目前会导出：

- `predictions.json`
- `scalar_summary.json`
- `task_semantics.json`
- `dataset_capability.json`
- `surface_values.csv`
- `slice_values.csv`
- `feature_summary.json`
- `surface_cp.png`
- `surface_pressure.png`
- `slice_u.png`
- `slice_v.png`
- `slice_p.png`
- `slice_nut.png`
- `high_gradient_regions.png`

更多说明见：

- [examples/analysis_outputs.md](/Users/jason/Documents/CFD/examples/analysis_outputs.md)
- [examples/analysis_outputs_airfrans.md](/Users/jason/Documents/CFD/examples/analysis_outputs_airfrans.md)

补充说明：

- inference artifact 现在还会携带 `branch_contract` 和 `geometry_backbone_contract`
- 其中 `geometry_backbone_contract` 当前主要用于说明：
  - 当前 checkpoint 是否仍是 `fixed_branch_vector`
  - 是否只是预留了 future native geometry backbone interface
- 这项 metadata 的目的不是宣称已经完成原生几何主干升级，而是避免后续误读 checkpoint 的 geometry conditioning 方式

## 13. Colab Notebook

仓库已提供独立 notebook：

- [CFD.ipynb](/Users/jason/Documents/CFD/CFD.ipynb)

它包含：

- 环境准备
- Drive/Colab 路径配置
- 数据重切分
- 训练
- 评测
- 推理
- 图像展示

说明：

- notebook 不修改本地 Python 代码
- 会根据 `torch.cuda.is_available()` 自动选择 GPU 或 CPU
- 在 Colab 中需要先把整个仓库上传到 Google Drive 或 `/content`

## 14. 当前结果

### 14.1 DeepONet full-physics 主结果

主结果目录：

- [outputs/airfrans_r1_full_physics](/Users/jason/Documents/CFD/outputs/airfrans_r1_full_physics)

报告：

- [r_1.md](/Users/jason/Documents/CFD/r_1.md)
- [results.md](/Users/jason/Documents/CFD/results.md)

### 14.2 Geo-FNO vs DeepONet 对比

对比目录：

- Geo-FNO：
  - [outputs/airfrans_geofno_r1_nophysics12](/Users/jason/Documents/CFD/outputs/airfrans_geofno_r1_nophysics12)
- DeepONet：
  - [outputs/airfrans_deeponet_r1_nophysics12](/Users/jason/Documents/CFD/outputs/airfrans_deeponet_r1_nophysics12)

统一口径下的测试集对比：

| Metric | Geo-FNO | DeepONet no-physics |
| --- | ---: | ---: |
| `field_rmse` | `0.2742` | `0.3454` |
| `cp_surface_rmse` | `0.7797` | `1.0499` |
| `slice_rmse` | `0.1664` | `0.2125` |
| `cl_mae` | `0.0952` | `0.1299` |
| `cd_mae` | `8.93e-4` | `1.42e-3` |
| `high_gradient_iou` | `0.7539` | `0.6975` |
| `pressure_gradient_f1` | `0.5594` | `0.3465` |

当前结论：

- Geo-FNO 在场、surface、slice、feature 指标上更强
- DeepONet full-physics 版本在部分全局标量上仍更稳

## 15. 如何使用这些输出

当前最实际的用法是：

- 作为二维翼型方案的快速分析器
- 做批量设计筛选前置过滤
- 看 `cl/cd` 趋势
- 看 `Cp` 和表面压力分布
- 看 slice 剖面与尾迹恢复
- 用高梯度区域做流动变化诊断

推荐工作流：

1. 批量生成候选翼型和工况
2. 用 surrogate 先跑一遍
3. 保留较优候选
4. 对候选方案再跑真实 CFD 精算

## 16. API

启动：

```bash
python scripts/serve.py \
  --checkpoint outputs/<run>/checkpoints/best.pt \
  --host 127.0.0.1 \
  --port 8000
```

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/inference_input.json
```

## 17. 测试

```bash
pytest -q
```

当前测试覆盖：

- unified schema
- adapters
- capabilities
- decoder heads
- geometry
- data
- losses
- physics
- postprocess
- scalar outputs
- surface outputs
- slice outputs
- feature outputs
- inference smoke
- evaluator smoke
- DeepONet smoke train
- Geo-FNO smoke train

## 18. 当前局限

- 当前仍是 2D 翼型主线，不是 3D 系统
- unified schema 已经落地，但 trainer/evaluator 当前仍通过 legacy payload bridge 工作，不是唯一内部表示
- 当前通用化的重点是“数据接入层与任务接口层解耦”，不是 branch encoder 全面重写
- `heat_flux_surface` 和 `wall_shear_surface` 仍是近似代理
- shock 相关量仍是高梯度近似，不是真实监督
- physics loss 仍是 proxy 级约束，不是求解器离散格式
- 标准 FNO 尚未在规则网格重采样路径下正式接入
- Geo-FNO 在 CPU 上启用 physics loss 时代价较高，更适合 GPU

## 19. 下一步改进方向

- 继续减少 unified schema 与 legacy payload 的双轨成本，逐步让更多内部流程直接读 schema
- 为新 2D CFD 数据源补 adapter，而不是继续把逻辑写进 trainer/evaluator
- 强化 `Geo-FNO` 的 scalar head，进一步压低 `cl/cd` 误差
- 在 GPU 上重跑带 physics loss 的 Geo-FNO 对比
- 继续统一 `pressure` 与 `cp_like` 的量纲处理
- 改进 feature 伪标签，避免类别不平衡导致退化
- 增强 surface 分支，进一步降低 `cp_surface_rmse`
- 研究规则网格化后的标准 FNO baseline
- 为后续 3D 推广准备几何编码、plane slice 和显存友好的采样方案
