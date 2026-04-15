# AirfRANS Analysis Outputs

本阶段的分析型输出增强严格按 AirfRANS 当前可提供的数据来设计。

## 当前能输出什么

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
  - optional `velocity_surface`
  - optional `nut_surface`
- slice outputs
  - 指定 `x = const`
  - 指定 `y = const`
  - 任意 2D line segment
- feature outputs
  - `high_gradient_mask`
  - `pressure_gradient_indicator`
  - `high_gradient_region_summary`

## 哪些是 AirfRANS 真实监督

- `u / v / p / nut`
- `cl / cd`
- `pressure_surface`

## 哪些是 derived outputs

- `cp_surface`
  - 由 `pressure_surface` 和 freestream reference 推导
- `slice_fields`
  - 由 2D 场结果沿指定切线采样/插值得到
- `high_gradient_mask`
- `pressure_gradient_indicator`
- `high_gradient_region_summary`
- `wall_shear_surface`
  - 近似后处理 proxy

## 哪些只是接口预留 / TODO

- `heat_flux_surface`
- `shock_indicator`
- `shock_location`
- `rho`
- `cm`
- `cdp / cdv / clp / clv`

## 物理意义

- `pressure_surface`
  - 翼型表面压力分布。
- `cp_surface`
  - 压力系数，定义为 `Cp = (p - p_ref) / q_ref`。
- `line slice`
  - 2D 流场上指定线段的剖面，可用于看尾迹、恢复区和局部高梯度带。
- `high_gradient_indicator`
  - 用于定位高压力梯度区域的分析指标，不等于真实 shock ground truth。

## 为什么不把 `heat flux / shock / rho` 当主监督

- AirfRANS 是 2D、不可压缩、亚声速 RANS 数据集。
- 数据主字段是 velocity / pressure / turbulent viscosity，不是 variable density。
- 数据集不提供真实 wall heat flux 和 shock 标签。

## 导出文件

启用 analysis bundle 后，默认导出：

- `predictions.json`
- `scalar_summary.json`
- `surface_values.csv`
- `slice_values.csv`
- `feature_summary.json`
- `scalar_summary.png`
- `surface_pressure.png`
- `surface_cp.png`
- `slice_<field>.png`
- `high_gradient_regions.png`

## 推理示例

```bash
PYTHONPATH=src python scripts/infer.py \
  --checkpoint outputs/default_run/checkpoints/best.pt \
  --input examples/inference_input.json \
  --output outputs/default_run/inference.json \
  --export-dir outputs/default_run/analysis_bundle
```
