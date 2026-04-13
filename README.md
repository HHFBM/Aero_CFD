# CFD Operator Surrogate

面向飞行器 CFD 计算加速的神经算子代理模型工程。当前版本聚焦二维参数化翼型、稳态可压缩外流场景，以 `DeepONet` 风格 operator surrogate 为主线，支持监督训练、physics-informed 混合训练、评测可视化、推理 CLI 和 FastAPI 服务。

## 适用场景

- 二维参数化翼型快速气动筛选
- CFD 前期设计空间探索
- surrogate baseline / operator learning 研究
- 真实 CFD 数据工程接入前的训练部署骨架

## 当前支持能力

- 几何：NACA 4-digit 参数化翼型
- 数据：toy synthetic dataset、NPZ 文件数据集读取，预留 CSV/Parquet/Pickle 接口
- 模型：DeepONet 主模型，FNO / GeoFNO placeholder
- 输出：
  - 查询点处流场变量 `u, v, p, rho`
  - 表面 `Cp`
  - 标量气动系数 `Cl, Cd`
- 训练：
  - 纯监督训练
  - physics-informed 混合训练
  - checkpoint / best model / early stopping / history 记录
- 评测：
  - field MSE / RMSE / relative error
  - `Cp` 误差
  - `Cl/Cd` MAE / relative error
  - 图像与 markdown/json 报告
- 推理：
  - 单样本 CLI
  - FastAPI 服务 `GET /health` `POST /predict` `POST /predict_batch`

## 关键说明

- 当前是 **二维参数化翼型代理模型**，不是工业级三维 CFD 求解器替代品。
- toy dataset 仅用于演示工程闭环，不代表真实工程精度，也不能用于结论性气动设计。
- physics-informed 模块采用 **二维稳态可压缩 Euler 简化残差**：
  - continuity residual
  - x-momentum residual
  - y-momentum residual
  - optional energy residual
- 残差由自动微分计算，目的是提供物理结构约束，不对应真实求解器的离散守恒格式。
- 当前数据生成与物理项默认假设远场静压 `p_inf=1`、比热比 `gamma=1.4`，且未显式建模粘性边界层和激波捕捉。

## 技术架构

```text
geometry params + flow conditions + query points
    -> data module / normalizers
    -> operator model (DeepONet)
    -> field head: [u, v, p, rho]
    -> scalar head: [Cl, Cd]
    -> surface Cp via predicted pressure
    -> composite loss
         - field loss
         - surface Cp loss
         - scalar loss
         - physics residual loss
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
  --override data.num_samples=64 \
  --override data.num_query_points=128 \
  --override data.num_surface_points=80
```

### 2. 当前样本字段

每个样本至少包含：

- `airfoil_id`
- `geometry_params`
- `flow_conditions`
- `branch_inputs`
- `query_points`
- `field_targets`
- `surface_points`
- `surface_cp`
- `scalar_targets`
- `metadata`
  - `fidelity_level`
  - `source`
  - `convergence_flag`

### 3. 文件型数据集

当前主路径是 `.npz`。默认 toy dataset 会保存为定长数组格式，便于直接训练。CSV/Parquet/Pickle 的接口已预留，但真实生产环境建议统一整理成 NPZ 或者自定义 reader。

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
  --override loss.physics_weight=0.1
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

## 评测

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/default_run/checkpoints/best.pt
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
- 可选 `reynolds`

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

## 如何替换为真实 CFD 数据

建议按以下顺序接入：

1. 准备统一样本 schema。
2. 为每个样本提供几何参数或几何编码。
3. 将工况字段至少映射为 `mach`、`aoa`，必要时扩展 `reynolds`。
4. 统一查询点表示。
   - 非结构网格：采样点 / 点云
   - 结构网格：规则网格，可后续切换 FNO
5. 将求解输出映射到：
   - `field_targets[..., 4] = [u, v, p, rho]`
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
- toy 数据是规则生成的 pseudo-CFD 场，不代表真实数值求解器行为。
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
