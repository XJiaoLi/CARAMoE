# Observation-Guided Mixture of Experts for Spatiotemporal Air Quality Forecasting with Incomplete Data

PyTorch implementation of **CARAMOE** on AirQualityBench, BLAir and ChinaAir.

[中文说明](README_zh.md)

## Requirements

Use Python 3.10 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data

Place `data_release/` beside `run.py`.

AirQualityBench and BLAir support horizons of 6, 12, 18 and 24 hours. ChinaAir supports 1, 3, 7 and 14 days. For BLAir, choose `--city beijing` or `--city london`.

## Training

Train the model before running predictions:

```bash
python run.py train --dataset aqb --horizon 12 --device cuda:0 --output-dir outputs/train/aqb_h12
python run.py train --dataset blair --city beijing --horizon 12 --device cuda:0 --output-dir outputs/train/blair_beijing_h12
python run.py train --dataset chinaair --horizon 7 --device cuda:0 --output-dir outputs/train/chinaair_h7
```

Use `--device cpu` for CPU execution. Choose a new output directory for each run and keep the generated weights and configuration files together.

## Prediction

After training completes, pass its output directory through `--checkpoint`. The program loads the best weight from that directory, runs predictions on the test set and computes prediction errors:

```bash
python run.py predict --dataset aqb --horizon 12 --checkpoint outputs/train/aqb_h12 --device cuda:0 --output-dir outputs/predict/aqb_h12
python run.py predict --dataset blair --city beijing --horizon 12 --checkpoint outputs/train/blair_beijing_h12 --device cuda:0 --output-dir outputs/predict/blair_beijing_h12
python run.py predict --dataset chinaair --horizon 7 --checkpoint outputs/train/chinaair_h7 --device cuda:0 --output-dir outputs/predict/chinaair_h7
```

You can also pass a specific `.pt` or `.pth` file. If the directory contains multiple experiments, select one experiment directory or weight file.

Errors are saved in `evaluation_metrics.json`. 
