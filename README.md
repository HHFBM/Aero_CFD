# CFD Operator Surrogate

面向飞行器 CFD 计算加速的神经算子代理模型工程。当前版本保持 `DeepONet` 风格 operator surrogate 主线不变，同时支持两类路径：`AirfRANS` 的 2D 不可压缩 RANS 数据主路径，以及 toy/compressible 演示数据路径。工程包含监督训练、physics-informed 混合训练、评测可视化、推理 CLI 和 FastAPI 服务。

## 适用场景

- 二维参数化翼型快速气动筛选
- CFD 前期设计空间探索
- surrogate baseline / operator learning 研究
- 真实 CFD 数据工程接入前的训练部署骨架

## 当前支持能力

- 几何：NACA 4-digit 参数化翼型
- 数据：toy synthetic dataset、AirfRANS 转换接入、NPZ 文件数据集读取，预留 CSV/Parquet/Pickle 接口
- 模型：DeepONet 主模型，FNO / GeoFNO placeholder
- 输出：
  - 查询点处流场变量 `u, v, p` 与第 4 通道 `aux`
    - AirfRANS 适配时第 4 通道为 `nut`
    - toy/compressible 演示配置下第 4 通道仍可为 `rho`
  - 表面 `Cp` / `pressure_surface`
  - 表面速度 / `nut_surface`（当第 4 通道真实为 `nut` 时）
  - 表面热流代理 `heat_flux_surface`
  - 表面壁面剪切代理 `wall_shear_surface`
  - 标量气动系数 `Cl, Cd`
  - line slice field outputs
  - high-gradient indicators / summaries
- 训练：
  - 纯监督训练
  - physics-informed 混合训练
  - boundary consistency 约束训练
  - checkpoint / best model / early stopping / history 记录
- 评测：
  - field MSE / RMSE / relative error
  - `Cp` 误差
  - `Cl/Cd` MAE / relative error
  - `IID / unseen geometry / unseen condition` 多测试切片
  - 图像与 markdown/json 报告
- 推理：
  - 单样本 CLI
  - FastAPI 服务 `GET /health` `POST /predict` `POST /predict_batch`

## 关键说明

- 当前是 **二维参数化翼型代理模型**，不是工业级三维 CFD 求解器替代品。
- toy dataset 仅用于演示工程闭环，不代表真实工程精度，也不能用于结论性气动设计。
- AirfRANS 当前以“离线转换到统一 NPZ schema”的方式接入，训练主线不直接依赖 AirfRANS 原始 API。
- physics-informed 模块采用 **二维稳态可压缩 Euler 简化残差**：
  - continuity residual
  - x-momentum residual
  - y-momentum residual
  - optional energy residual
- boundary consistency 模块当前包含：
  - 壁面无穿透约束
  - 远场状态一致性约束
- 残差由自动微分计算，目的是提供物理结构约束，不对应真实求解器的离散守恒格式。
- 当前数据生成与物理项默认假设远场静压 `p_inf=1`、比热比 `gamma=1.4`，且未显式建模粘性边界层和激波捕捉。
- `heat_flux_surface` 与 `wall_shear_surface` 当前是 **近似后处理 proxy**，用于分析型输出打通，不代表工业级壁面热流/壁面剪切高精度预测。
- `shock_indicator` 与 `shock_location_summary` 当前也是 **简化近似版本**，主要用于研究验证和报告分析。

## AirfRANS 分析输出增强阶段

当前仓库已经补齐一版“分析型输出增强阶段（AirfRANS 适配版）”，但严格区分了三类输出：

- 真实监督/真实打通：
  - pointwise `u, v, p, nut`
  - scalar `cl, cd`
  - `pressure_surface`
  - `cp_surface`
  - `slice` 上的 `u, v, p, nut`
- derived outputs：
  - `cp_surface`
    - 若数据源只提供表面压力，则由 `Cp = (p - p_ref) / q_ref` 推导
  - `high_gradient_mask`
  - `pressure_gradient_indicator`
  - `high_gradient_region_summary`
  - `wall_shear_surface`
    - 当前仅为近似后处理，不是 AirfRANS 真实 wall-shear supervision
- placeholder / TODO：
  - `heat_flux_surface`
  - `shock_indicator`
  - `shock_location`
  - `rho`
  - `cm`
  - `cdp / cdv / clp / clv`

### 物理意义说明

- `pressure_surface`
  - 翼型表面静压分布，用于观察吸力峰、压差载荷和沿表面压力恢复。
- `cp_surface`
  - 无量纲压力系数，定义为 `Cp = (p - p_ref) / q_ref`。
  - 这里 `p_ref` 与 `q_ref` 来自数据或推理输入中的 freestream reference。
- `line slice`
  - 在 `x = const`、`y = const` 或任意 2D 线段上抽取 `u/v/p/nut`，用于查看尾迹、恢复段和局部流动结构。
- `high_gradient_indicator`
  - 基于压力梯度高值区域的分析指标，用于快速定位强变化区域。
  - 在 AirfRANS 场景中它是分析代理，不应表述为真实 shock ground truth。

### 为什么当前不把 `heat flux / shock / rho` 作为主监督目标

- AirfRANS 是 2D、不可压缩、亚声速 RANS 数据集，不提供真实的壁面热流监督。
- AirfRANS 当前主字段围绕 velocity / pressure / turbulent viscosity / distance function，而不是 variable density。
- shock 相关标签在该数据条件下不应被包装成正式 ground truth；当前只允许作为高梯度近似分析接口。

## 技术架构

```text
geometry params + flow conditions + query points
    -> data module / normalizers
    -> operator model (DeepONet)
    -> field head: [u, v, p, aux]
         - AirfRANS: aux = nut
         - toy compressible demo: aux = rho
    -> scalar head: [Cl, Cd]
    -> surface outputs via surface-point evaluation + postprocess
         - Cp
         - pressure
         - optional velocity / nut
         - approximate heat flux / wall shear
    -> slice outputs via slice sampling
    -> feature outputs
         - high-gradient indicator
         - high-gradient summary
    -> composite loss
         - field loss
         - surface Cp loss
         - optional surface pressure / slice / feature loss
         - scalar loss
         - physics residual loss
         - boundary consistency loss
    -> trainer / evaluator / inference / API
```

## 项目结构

```text
configs/
  data/
  model/
  train/
  eval/
  serve/
  default.yaml
src/cfd_operator/
  config/
  data/
  geometry/
  physics/
  models/
  losses/
  trainers/
  evaluators/
  inference/
  api/
  utils/
  visualization/
scripts/
tests/
examples/
outputs/
```

## 安装

推荐 Python `3.11+`。当前仓库代码保持了更保守的语法兼容性，但建议训练与部署使用较新的 Python / PyTorch 环境。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

如果要接入 AirfRANS，再额外安装可选数据依赖：

```bash
pip install ".[data]"
```

如果本机 `matplotlib` 无法写默认缓存目录，可临时设置：

```bash
export MPLCONFIGDIR=/tmp/mpl
export XDG_CACHE_HOME=/tmp
```

## 配置系统

项目使用纯 YAML 配置，并支持命令行覆盖：

```bash
python scripts/train.py --config configs/default.yaml --override train.epochs=5 --override loss.use_physics=false
```

主配置分区：

- `experiment`
- `data`
- `model`
- `train`
- `loss`
- `eval`
- `serve`

## 数据准备

### 1. 生成 toy dataset

```bash
python scripts/prepare_dataset.py --config configs/default.yaml
```

也可以覆盖样本规模：

```bash
python scripts/prepare_dataset.py \
  --config configs/default.yaml \
  --override data.dataset_path=outputs/data/toy_small.npz \
  --override data.num_geometries=8 \
  --override data.conditions_per_geometry=8 \
  --override data.num_query_points=128 \
  --override data.num_surface_points=80
```

当前 toy dataset 会显式生成：

- seen geometry / seen condition 的 `train / val / test`
- `test_unseen_geometry`
- `test_unseen_condition`

### 2. 当前样本字段

每个样本至少包含：

- `airfoil_id`
- `geometry_params`
- `flow_conditions`
- `branch_inputs`
- `query_points`
- `field_targets`
- `farfield_mask`
- `farfield_targets`
- `surface_points`
- `surface_normals`
- `surface_cp`
- `surface_pressure`
- `surface_heat_flux`
- `surface_wall_shear`
- `slice_points`
- `slice_fields`
- `shock_indicator`
- `high_gradient_mask`
- `shock_location`
- `scalar_targets`
- `metadata`
  - `fidelity_level`
  - `source`
  - `convergence_flag`

### 3. 文件型数据集

当前主路径是 `.npz`。默认 toy dataset 会保存为定长数组格式，便于直接训练。CSV/Parquet/Pickle 的接口已预留，但真实生产环境建议统一整理成 NPZ 或者自定义 reader。

### 4. AirfRANS 数据接入

当前已支持把官方 AirfRANS 数据集下载并转换成项目自己的统一 `.npz` 格式。

转换命令：

```bash
python scripts/prepare_dataset.py \
  --config configs/default.yaml \
  --override data.dataset_type=airfrans \
  --override data.dataset_path=outputs/data/airfrans_full.npz \
  --override data.airfrans_root=outputs/data/airfrans_raw \
  --override data.airfrans_task=full
```

也可以限制样本数做快速验证：

```bash
python scripts/prepare_dataset.py \
  --config configs/default.yaml \
  --override data.dataset_type=airfrans \
  --override data.dataset_path=outputs/data/airfrans_tiny.npz \
  --override data.airfrans_root=outputs/data/airfrans_raw \
  --override data.airfrans_task=full \
  --override data.airfrans_max_samples=32
```

AirfRANS 转换后的字段映射为：

- `field_targets = [u, v, p, nut]`
- `surface_pressure`
  - 由 AirfRANS 表面采样直接读取
- `surface_cp`
  - 由 `surface_pressure` 和 `cp_reference = [p_ref, q_ref]` 推导
- `scalar_targets = [cl, cd]`
  - 由官方 `Simulation.force_coefficient(reference=True)` 计算
- `slice_fields`
  - 由 2D 场采样/插值得到
- `high_gradient_*`
  - 由压力梯度近似生成，用于分析而非官方 benchmark

注意：

- AirfRANS 是 **不可压缩 RANS** 数据，不是你当前项目目标里的可压 Euler 高保真数据。
- AirfRANS 当前真实主监督集中在 `u / v / p / nut / cl / cd / pressure_surface`。
- `cp_surface` 是 derived output，不是额外原生标签。
- `heat_flux_surface / wall_shear_surface / shock_* / rho` 不是 AirfRANS 真实监督主任务。
- 如果用 AirfRANS 训练，建议把 physics loss 视为弱约束，并结合任务需要重新调 `physics_weight`。

## 训练

### 1. 纯监督训练

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --override loss.use_physics=false
```

### 2. physics-informed 混合训练

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --override loss.use_physics=true \
  --override loss.physics_weight=0.1 \
  --override loss.boundary_weight=0.1
```

用 AirfRANS 训练时，先把 `dataset_type` 改成 `file`，读取转换后的 NPZ：

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --override data.dataset_type=file \
  --override data.dataset_path=outputs/data/airfrans_full.npz
```

训练输出默认保存到：

```text
outputs/<experiment.name>/
  checkpoints/
  logs/
  reports/
  figures/
```

主要产物：

- `checkpoints/best.pt`
- `reports/history.csv`
- `reports/latest_metrics.json`
- `logs/train.log`

### 分析型输出相关训练开关

当前可通过 loss 配置单独开启或关闭：

- `loss.use_surface_pressure_loss`
- `loss.use_slice_loss`
- `loss.use_feature_loss`
- `loss.use_heat_flux_loss`
- `loss.use_wall_shear_loss`

当前推荐：

- `cp_surface`、`pressure_surface`、`slice field`：可真实打通并参与监督
- `heat_flux_surface`、`wall_shear_surface`：默认只做后处理导出，不建议当作高精度监督目标

## 评测

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/default_run/checkpoints/best.pt
```

评测 unseen geometry：

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/default_run/checkpoints/best.pt \
  --override eval.split_name=test_unseen_geometry
```

评测 unseen condition：

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/default_run/checkpoints/best.pt \
  --override eval.split_name=test_unseen_condition
```

关闭绘图时：

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/default_run/checkpoints/best.pt \
  --override eval.save_plots=false
```

评测输出包括：

- `metrics.json`
- `report.md`
- parity/scatter/field/Cp 图
- surface pressure / surface Cp 图
- `slice_<field>.png`
- `high_gradient_regions.png`
- sample 级 `predictions.json`、`scalar_summary.json`、`surface_values.csv`、`slice_values.csv`、`feature_summary.json`

## 当前模型能输出什么

### 1. Pointwise Field Outputs

- `u`
- `v`
- `p`
- 第 4 通道 `aux`
  - AirfRANS 训练配置下为 `nut`
  - toy/compressible 演示配置下可为 `rho`

### 2. Scalar Outputs

- `cl`
- `cd`
- 预留接口：`cdp / cdv / clp / clv / cm`
  - 当前默认不作为真实训练主任务

### 3. Surface Outputs

- `pressure_surface`
- `cp_surface`
  - 由 `pressure_surface` 和 freestream reference 派生
- optional `velocity_surface`
- optional `nut_surface`
- `heat_flux_surface`
  - placeholder / approximate postprocess
- `wall_shear_surface`
  - derived / approximate postprocess

### 4. Slice Outputs

- 任意 `x=const`
- 任意 `y=const`
- 任意二维线段
- slice 上可导出 `u / v / p / aux`
  - AirfRANS 主路径下 `aux = nut`

### 5. Region / Feature Outputs

- `high_gradient_mask`
- `pressure_gradient_indicator`
- `high_gradient_region_summary`
- `shock_indicator`
  - placeholder / high-gradient approximation
- `shock_location_summary`
  - placeholder / high-gradient approximation

## 这些输出和飞行器分析的关系

- `cp_surface`
  - 用于看表面载荷分布、吸力峰和沿表面压差变化。
- `pressure_surface`
  - 是 `Cp` 之外更直接的压力结果，可用于后续载荷分析。
- `slice field`
  - 有助于看中心线/法向切片上的压力恢复、尾迹和局部高梯度区。
- `high_gradient_mask` / `pressure_gradient_indicator`
  - 用于快速标记高梯度流动特征区；在 AirfRANS 场景中不应包装成真实 shock ground truth。
- `cl/cd`
  - 对应整体气动性能指标。

## 当前哪些输出是真实监督，哪些是近似

真实打通或可直接监督：

- `u, v, p, nut`（AirfRANS 主路径）
- `cl, cd`
- `pressure_surface`
- `slice_fields`
  - 由点场结果沿指定切线导出，可参与 slice loss

derived outputs：

- `cp_surface`
- `high_gradient_mask`
- `pressure_gradient_indicator`
- `high_gradient_region_summary`

近似后处理 / 占位：

- `heat_flux_surface`
- `wall_shear_surface`
- `shock_indicator`
- `shock_location_summary`
- `rho`
- `cdp / cdv / clp / clv / cm`

这些近似输出当前主要用于工程原型和研究验证，不应视为高雷诺数、强激波、复杂粘性场景下的工业级结果。

## 推理

示例输入见 [examples/inference_input.json](/Users/jason/Documents/CFD/examples/inference_input.json)。

```bash
python scripts/infer.py \
  --checkpoint outputs/default_run/checkpoints/best.pt \
  --input examples/inference_input.json \
  --output outputs/default_run/inference.json
```

支持输出格式：

- `.json`
- `.csv`
- `.npz`

输入需要包含：

- `geometry_params`
- `mach`
- `aoa`
- `query_points`
- 可选 `surface_points`
- 可选 `slice_definitions`
- 可选 `reynolds`

### 完整分析型推理

```bash
python scripts/infer.py \
  --checkpoint outputs/default_run/checkpoints/best.pt \
  --input examples/inference_input.json \
  --output outputs/default_run/inference.json \
  --export-dir outputs/default_run/analysis_bundle
```

导出目录中会包含：

- `predictions.json`
- `surface_values.csv`
- `slice_values.csv`
- `feature_summary.json`
- `surface_cp.png`
- `surface_pressure.png`
- `slice_pressure.png`
- `high_gradient_regions.png`
- `scalar_summary.png`

更详细说明见 [examples/analysis_outputs.md](/Users/jason/Documents/CFD/examples/analysis_outputs.md)。

## API 服务

启动方式：

```bash
python scripts/serve.py \
  --checkpoint outputs/default_run/checkpoints/best.pt \
  --host 127.0.0.1 \
  --port 8000
```

也可以直接用 uvicorn：

```bash
export CFD_OPERATOR_CHECKPOINT=outputs/default_run/checkpoints/best.pt
export CFD_OPERATOR_DEVICE=cpu
uvicorn cfd_operator.api.app:create_app_from_env --factory --host 127.0.0.1 --port 8000
```

示例请求：

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/inference_input.json
```

## 测试

```bash
pytest -q
```

包含：

- 几何生成测试
- 数据加载测试
- 模型 forward shape 测试
- physics residual shape 测试
- loss 计算测试
- smoke train test
- analysis postprocess / surface / slice / feature / infer bundle 测试

## 如何替换为真实 CFD 数据

建议按以下顺序接入：

1. 准备统一样本 schema。
2. 为每个样本提供几何参数或几何编码。
3. 将工况字段至少映射为 `mach`、`aoa`，必要时扩展 `reynolds`。
4. 统一查询点表示。
   - 非结构网格：采样点 / 点云
   - 结构网格：规则网格，可后续切换 FNO
5. 将求解输出映射到：
   - `field_targets[..., 4] = [u, v, p, aux]`
     - AirfRANS 路径：`aux = nut`
     - toy/compressible 路径：`aux = rho`
   - `surface_cp`
   - `scalar_targets = [cl, cd]`
6. 标注 `fidelity_level`、`source`、`convergence_flag`。
7. 实现真实 reader，优先扩展 `src/cfd_operator/data/file_dataset.py`。
8. 根据真实变量尺度重新检查 normalizer 和 loss 权重。
9. 如果网格与几何耦合更强，优先扩展 GeoFNO / geometry encoder。
10. 对 physics loss 重新校验单位、一致性和边界条件定义。

## 当前局限

- 仅支持二维参数化翼型。
- 真实几何输入目前只完成了 NACA 4-digit 主路径。
- physics-informed 残差是连续方程近似，不是离散 CFD scheme。
- 没有显式粘性项、湍流模型、壁函数和 shock-aware 建模。
- `heat_flux_surface` 和 `wall_shear_surface` 当前只是近似 proxy，不代表可直接用于热防护或摩擦阻力定量设计。
- `shock_indicator` 和 `shock_location_summary` 当前是基于梯度的近似分析输出，不是专门的 shock tracking 算法。
- toy 数据是规则生成的 pseudo-CFD 场，不代表真实数值求解器行为。
- boundary consistency 当前仍是简化约束，不等同于完整边界条件离散实现。
- AirfRANS 接入当前主要面向数据替换和工程验证，不代表已完成最优的不可压缩 RANS 专用建模。
- multifidelity 只完成接口预留，未实现完整联合训练策略。
- FNO / GeoFNO 仅有占位类，未给出可训练实现。
- API 当前是同步推理，不含批任务队列、模型热更新、鉴权和观测性。

## 下一步扩展方向

- 接入真实 CFD 数据 reader 和数据版本管理
- geometry encoder 从参数扩展到点云 / CST / CAD 派生表示
- 完整 FNO / GeoFNO 实现
- multifidelity trainer 与 residual transfer
- shock/transonic-aware loss 设计
- 更严格的边界条件 loss 与 wall/farfield sampling
- experiment tracking、MLOps pipeline、容器化部署
