# Observation-Guided Mixture of Experts for Spatiotemporal Air Quality Forecasting with Incomplete Data

本项目提供 **CARAMOE** 的 PyTorch 实现，支持 AirQualityBench、BLAir 和 ChinaAir 三个数据集。

## 环境

使用 Python 3.10，安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 数据

将 `data_release/` 放在 `run.py` 同级。

AirQualityBench 和 BLAir 支持预测未来 6、12、18、24 小时；ChinaAir 支持预测未来 1、3、7、14 天。BLAir 通过 `--city beijing` 或 `--city london` 选择城市。

## 训练

先运行训练，得到模型权重：

```bash
python run.py train --dataset aqb --horizon 12 --device cuda:0 --output-dir outputs/train/aqb_h12
python run.py train --dataset blair --city beijing --horizon 12 --device cuda:0 --output-dir outputs/train/blair_beijing_h12
python run.py train --dataset chinaair --horizon 7 --device cuda:0 --output-dir outputs/train/chinaair_h7
```

使用 CPU 时改为 `--device cpu`。每次训练选择新的输出目录，并保留其中的权重和配套配置文件。

## 预测

训练完成后，通过 `--checkpoint` 指定训练输出目录。程序从中读取最佳权重，在测试集上预测并计算误差：

```bash
python run.py predict --dataset aqb --horizon 12 --checkpoint outputs/train/aqb_h12 --device cuda:0 --output-dir outputs/predict/aqb_h12
python run.py predict --dataset blair --city beijing --horizon 12 --checkpoint outputs/train/blair_beijing_h12 --device cuda:0 --output-dir outputs/predict/blair_beijing_h12
python run.py predict --dataset chinaair --horizon 7 --checkpoint outputs/train/chinaair_h7 --device cuda:0 --output-dir outputs/predict/chinaair_h7
```

也可以直接指定某个 `.pt` 或 `.pth` 文件。如果目录中有多组训练结果，请指定其中一组实验目录或具体权重文件。

预测误差保存至 `evaluation_metrics.json`。
