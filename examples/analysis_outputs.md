# Analysis Outputs Guide

AirfRANS 适配版的最新说明见 [analysis_outputs_airfrans.md](./analysis_outputs_airfrans.md)。

当前工程除了输出查询点场值和 `Cl/Cd`，还支持一组“分析型输出”。

## Surface Outputs

- `cp_surface`
  - 物理意义：表面压力系数，反映翼型载荷分布、吸力峰和激波相关变化。
  - 当前状态：真实打通。由表面点压力和 `cp_reference` 计算。
- `pressure_surface`
  - 物理意义：翼型表面压力分布。
  - 当前状态：真实打通。通过模型在表面点的场预测得到。
- `heat_flux_surface`
  - 物理意义：壁面热流密度，和热防护、压缩加热有关。
  - 当前状态：近似后处理占位。当前用表面压力导出的温度代理沿表面梯度近似，不代表高精度壁面热流。
- `wall_shear_surface`
  - 物理意义：壁面剪切应力，和摩擦阻力、边界层状态有关。
  - 当前状态：近似后处理占位。当前用表面切向速度变化构造 proxy，不代表高精度粘性壁面剪切。

## Slice Outputs

- `slice_fields`
  - 物理意义：在指定切线上观察 `u, v, p, rho` 的变化。
  - 用途：查看中心线压力恢复、尾迹、局部高梯度带。
  - 当前状态：真实打通。推理时可以指定 `x=const`、`y=const` 或任意线段。

## Feature Outputs

- `high_gradient_mask`
  - 物理意义：高梯度区域的近似指示，常用于定位压缩波、高载荷变化带、尾迹结构。
  - 当前状态：真实可用，但主要是后处理型近似指标。
- `shock_indicator`
  - 物理意义：更严格阈值下的高梯度/激波候选区域指示。
  - 当前状态：当前默认通过场梯度后处理得到；如果训练时启用了 `feature_head`，也可由模型直接输出。
- `shock_location_summary`
  - 物理意义：对候选激波区域的质心、范围和峰值梯度做摘要。
  - 当前状态：简化近似版本，用于分析报告，不应视为高保真 shock tracking。

## 导出文件

启用 analysis bundle 后，默认会导出：

- `predictions.json`
- `surface_values.csv`
- `slice_values.csv`
- `feature_summary.json`
- `surface_cp.png`
- `surface_pressure.png`
- `slice_pressure.png`
- `high_gradient_regions.png`
- `scalar_summary.png`

## 推理示例

```bash
python scripts/infer.py \
  --checkpoint outputs/default_run/checkpoints/best.pt \
  --input examples/inference_input.json \
  --output outputs/default_run/inference.json \
  --export-dir outputs/default_run/analysis_bundle
```
