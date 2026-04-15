# Results

## 1. Run Summary

本次结果对应一次修复 `surface Cp` 量纲 bug 后的 AirfRANS 适配版分析输出增强重训练：

- 运行目录：
  - [outputs/airfrans_analysis_retrain_20260415_cpfix](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix)
- 训练数据：
  - [outputs/data/airfrans_original_heavy.npz](/Users/jason/Documents/CFD/outputs/data/airfrans_original_heavy.npz)
- 主配置：
  - [configs/experiments/airfrans_original_heavy.yaml](/Users/jason/Documents/CFD/configs/experiments/airfrans_original_heavy.yaml)
- 最终 checkpoint：
  - [best.pt](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/checkpoints/best.pt)

本轮没有推翻原有训练主线，只在现有 DeepONet 主结构上启用了分析阶段需要的附加头和损失：

- `model.feature_output_dim=2`
- `loss.use_slice_loss=true`
- `loss.slice_weight=0.05`
- `loss.use_feature_loss=true`
- `loss.feature_weight=0.02`

说明：

- 初次尝试过显式 `surface_pressure_loss`，但由于 raw pressure 尺度过大，会严重主导总损失，因此本次正式结果中关闭了这项 loss。
- `heat_flux_surface`、`wall_shear_surface`、`shock_*` 仍未作为真实监督主任务参与训练。

## 2. Training Command

```bash
.venv/bin/python scripts/train.py \
  --config configs/experiments/airfrans_original_heavy.yaml \
  --override experiment.name=airfrans_analysis_retrain_20260415_cpfix \
  --override model.feature_output_dim=2 \
  --override loss.use_slice_loss=true \
  --override loss.slice_weight=0.05 \
  --override loss.use_feature_loss=true \
  --override loss.feature_weight=0.02 \
  --override train.epochs=30 \
  --override train.early_stopping_patience=10
```

训练记录：

- history：
  - [history.csv](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/reports/history.csv)
- 日志：
  - [train.log](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/logs/train.log)

最佳验证轮次是 `epoch 23`，对应：

- `val_loss_total = 0.8471`
- `val_loss_field = 0.4344`
- `val_loss_surface = 0.6870`
- `val_loss_slice = 0.2744`
- `val_loss_feature = 0.1296`
- `val_loss_scalar = 0.1028`

## 3. Evaluation Commands

```bash
.venv/bin/python scripts/evaluate.py \
  --config configs/experiments/airfrans_original_heavy.yaml \
  --checkpoint outputs/airfrans_analysis_retrain_20260415_cpfix/checkpoints/best.pt \
  --override experiment.name=airfrans_analysis_retrain_20260415_cpfix \
  --override model.feature_output_dim=2 \
  --override eval.split_name=test \
  --override eval.save_plots=true \
  --override eval.num_visualization_samples=3 \
  --override eval.export_analysis=true
```

同时额外评测了：

- `test_unseen_geometry`
- `test_unseen_condition`

## 4. Main Metrics

### 4.1 Test Split

指标文件：

- [metrics.json](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/eval/test/metrics.json)
- [report.md](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/eval/test/report.md)

核心结果：

- `field_rmse = 0.2441`
- `field_relative_error = 0.2611`
- `slice_rmse = 0.2025`
- `slice_relative_error = 0.3131`
- `cl_mae = 0.04557`
- `cl_relative_error = 0.09565`
- `cd_mae = 7.98e-4`
- `cd_relative_error = 0.15423`
- `cp_surface_rmse = 0.6374`
- `pressure_surface_rmse = 1308.84`
- `nut_rmse = 0.001630`
- `pressure_gradient_indicator_accuracy = 0.9733`
- `pressure_gradient_indicator_f1 = 0.4222`
- `high_gradient_accuracy = 0.9664`
- `high_gradient_iou = 0.7156`

### 4.2 Unseen Geometry

指标文件：

- [metrics.json](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_v2/eval/test_unseen_geometry/metrics.json)

核心结果：

- `field_rmse = 0.1813`
- `slice_rmse = 0.1553`
- `cl_mae = 0.03718`
- `cd_mae = 7.74e-4`
- `cp_surface_rmse = 2.2841`
- `high_gradient_iou = 0.7177`

### 4.3 Unseen Condition

指标文件：

- [metrics.json](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_v2/eval/test_unseen_condition/metrics.json)

核心结果：

- `field_rmse = 0.09681`
- `slice_rmse = 0.18168`
- `cl_mae = 0.00467`
- `cd_mae = 4.85e-5`
- `cp_surface_rmse = 1.1818`
- `high_gradient_iou = 0.8738`

## 5. Analysis Bundle Export

推理命令：

```bash
.venv/bin/python scripts/infer.py \
  --checkpoint outputs/airfrans_analysis_retrain_20260415_cpfix/checkpoints/best.pt \
  --input examples/inference_input.json \
  --output outputs/airfrans_analysis_retrain_20260415_cpfix/inference.json \
  --export-dir outputs/airfrans_analysis_retrain_20260415_cpfix/analysis_bundle
```

输出目录：

- [analysis_bundle](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/analysis_bundle)

已导出的结果包括：

- [predictions.json](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/analysis_bundle/predictions.json)
- [scalar_summary.json](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/analysis_bundle/scalar_summary.json)
- [surface_values.csv](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/analysis_bundle/surface_values.csv)
- [slice_values.csv](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/analysis_bundle/slice_values.csv)
- [feature_summary.json](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/analysis_bundle/feature_summary.json)
- `surface_cp.png`
- `surface_pressure.png`
- `slice_u.png`
- `slice_v.png`
- `slice_p.png`
- `slice_nut.png`
- `high_gradient_regions.png`
- `predicted_pressure_field.png`

示例推理输出中的标量预测为：

- `cl = 2.2964`
- `cd = 0.0210`

注意：

- 这个 inference 输入来自 [examples/inference_input.json](/Users/jason/Documents/CFD/examples/inference_input.json)，是一个通过 NACA 几何参数生成的示例，不是 AirfRANS 测试集原样本。
- 因此这里更适合作为“analysis bundle 导出能力展示”，而不是正式 benchmark 数值。

## 6. Visual Artifacts

测试集图表：

- [loss_curve.png](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/eval/test/loss_curve.png)
- [cl_scatter.png](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/eval/test/cl_scatter.png)
- [cd_scatter.png](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/eval/test/cd_scatter.png)
- [sample_00 surface_cp.png](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/eval/test/sample_00/surface_cp.png)
- [sample_00 surface_pressure.png](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/eval/test/sample_00/surface_pressure.png)
- [sample_00 slice_p.png](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/eval/test/sample_00/slice_p.png)
- [sample_00 high_gradient_regions.png](/Users/jason/Documents/CFD/outputs/airfrans_analysis_retrain_20260415_cpfix/eval/test/sample_00/high_gradient_regions.png)

## 7. Interpretation

本轮重训练说明当前工程已经能在 AirfRANS 数据条件下稳定输出并评测以下真实主线能力：

- pointwise `u / v / p / nut`
- scalar `cl / cd`
- surface `pressure_surface`
- derived `cp_surface`
- line slice fields
- high-gradient analysis outputs

同时需要明确保留边界：

- 修复 `cp` 量纲 bug 后，`cp_surface_rmse` 从上一版的 `2.0786` 降到了 `0.6374`，说明此前 surface `Cp` 预测贴近 0 的问题来自错误的二次转换，而不是模型本身完全不会学习。
- `pressure_surface` 在 `airfrans_original` 路径下是由内部 `cp_like` 压力表示重构得到，因此其绝对数值仍然依赖 reference pressure 定义。
- `cp_surface`、slice 和高梯度分析已经打通，但高梯度类指标仍属于 derived analysis，不是官方 AirfRANS benchmark。
- `heat_flux_surface`、`wall_shear_surface`、`shock_indicator`、`shock_location` 仍只应视为近似后处理或接口预留。

## 8. Geo-FNO Comparison

本轮新增接入了一个可训练的 `Geo-FNO` 基线，用于和当前 `DeepONet` 做同数据切分下的直接对比。

说明：

- 当前 AirfRANS heavy 数据的 `query_points` 是样本级非规则点云，不是统一规则网格。
- 因此标准 `FNO` 不能直接公平套用；本次接入的是适配非规则点的 `Geo-FNO` 风格模型。
- 为了先获得可比基线，本轮 `Geo-FNO` 和对应的 `DeepONet` 对照都采用了：
  - 相同重切分数据：`700 / 150 / 150`
  - `12 epoch`
  - `slice loss + feature loss`
  - `no physics loss`

相关运行目录：

- Geo-FNO：
  - [outputs/airfrans_geofno_r1_nophysics12](/Users/jason/Documents/CFD/outputs/airfrans_geofno_r1_nophysics12)
- DeepONet 对照：
  - [outputs/airfrans_deeponet_r1_nophysics12](/Users/jason/Documents/CFD/outputs/airfrans_deeponet_r1_nophysics12)

Geo-FNO 配置：

- [configs/experiments/airfrans_geofno_heavy.yaml](/Users/jason/Documents/CFD/configs/experiments/airfrans_geofno_heavy.yaml)

### 8.1 Test Metrics Comparison

| Metric | Geo-FNO | DeepONet (no physics, 12 ep) | DeepONet (full physics, 30 ep) |
| --- | ---: | ---: | ---: |
| field_rmse | 0.2742 | 0.3454 | 0.4331 |
| cp_surface_rmse | 0.7797 | 1.0499 | 0.9540 |
| slice_rmse | 0.1664 | 0.2125 | 0.2435 |
| cl_mae | 0.09517 | 0.12993 | 0.05695 |
| cd_mae | 0.000893 | 0.001415 | 0.000750 |
| high_gradient_iou | 0.7539 | 0.6975 | 0.3349 |
| pressure_gradient_f1 | 0.5594 | 0.3465 | 0.0000 |

指标文件：

- Geo-FNO：
  - [metrics.json](/Users/jason/Documents/CFD/outputs/airfrans_geofno_r1_nophysics12/eval/test/metrics.json)
- DeepONet no-physics：
  - [metrics.json](/Users/jason/Documents/CFD/outputs/airfrans_deeponet_r1_nophysics12/eval/test/metrics.json)
- DeepONet full-physics：
  - [metrics.json](/Users/jason/Documents/CFD/outputs/airfrans_r1_full_physics/eval/test/metrics.json)

### 8.2 Current Reading

这次首轮对比说明：

- 在非规则点云 AirfRANS 数据上，`Geo-FNO` 作为 architecture baseline 是成立的，不是空占位。
- 在统一的 `no-physics / 12 epoch` 口径下，`Geo-FNO` 在场误差、表面 `Cp`、slice 和 feature 指标上都优于 `DeepONet`。
- `DeepONet` 在当前 full-physics run 上的 `cl/cd` 标量依然更强，尤其 `cl_mae` 更低。
- 这说明两种模型的优势方向并不完全相同：
  - `Geo-FNO` 更擅长场与分析输出
  - `DeepONet` 当前在全局气动标量上仍然更稳

### 8.3 Next Step

下一步最值得做的是：

- 给 `Geo-FNO` 增加更稳的 scalar head 训练策略，重点压低 `cl/cd` 误差。
- 在 GPU 环境下重新开启 `physics loss`，观察其对场与表面输出是否进一步增益。
- 如果后续要继续做更严格对比，建议新增一轮完全同口径实验：
  - 相同 epoch
  - 相同 physics 开关
  - 相同 model width
  - 相同 early stopping 策略
