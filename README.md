# A graphical approach for filter pruning by exploring the similarity relation between feature maps ([Link](https://doi.org/10.1016/j.patrec.2022.12.028))

## Installation

This code requires Tensorflow 2.5.0 and Python 3.8.0, please create the environment and install dependencies by
```bash
conda env create --file environment.yml
```

## Usage

You can train a model from scratch on the CIFAR-10 dataset by
```bash
usage: train.py [-h] [--arch ARCH] [--save-dir SAVE_DIR] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE] [--optimizer OPTIMIZER]
                [--learning-rate LEARNING_RATE] [--decay-step DECAY_STEP]
                [--gpu GPU]

CIFAR-10 Training

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH           model architecture
  --save-dir SAVE_DIR   model saving directory
  --epochs EPOCHS       number of training epochs
  --batch-size BATCH_SIZE
                        batch size
  --optimizer OPTIMIZER
                        optimizer
  --learning-rate LEARNING_RATE
                        initial learning rate
  --decay-step DECAY_STEP
                        learning rate decay
  --gpu GPU             GPU id to use

```

You can prune and fine-tune a model on the CIFAR-10 dataset by
```bash
usage: prune.py [-h] [--arch ARCH] [--pretrain-dir PRETRAIN_DIR]
                [--save-dir SAVE_DIR] [--threshold THRESHOLD]
                [--samples SAMPLES] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE]
                [--decay-step DECAY_STEP] [--gpu GPU]

CIFAR-10 Pruning and Fine-tuning

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH           model architecture
  --pretrain-dir PRETRAIN_DIR
                        pre-trained model saving directory
  --save-dir SAVE_DIR   pruned model saving directory
  --threshold THRESHOLD
                        similarity threshold
  --samples SAMPLES     number of input samples to calculate average
                        similarity
  --epochs EPOCHS       number of fine-tuning epochs
  --batch-size BATCH_SIZE
                        batch size
  --learning-rate LEARNING_RATE
                        initial learning rate
  --decay-step DECAY_STEP
                        learning rate decay
  --gpu GPU             GPU id to use

```

## Examples

### Train/prune the specified model each time

If optional parameters are not set, default values are used.

#### VGG-16

| threshold | Acc.   | FLOPs          | Params.      |
|-----------|--------|----------------|--------------|
| 0.95      | 92.59% | 125.25M(60.0%) | 1.38M(90.8%) |
| 0.9       | 92.24% | 124.04M(60.4%) | 1.13M(92.5%) |
| 0.85      | 92.08% | 114.78M(63.4%) | 0.94M(93.7%) |
| 0.8       | 91.86% | 107.22M(65.8%) | 0.85M(94.4%) |
| 0.75      | 90.92% | 82.65M(73.6%)  | 0.70M(95.3%) |
| 0.7       | 59.72% | 47.86M(84.7%)  | 0.44M(97.1%) |

```bash
python train.py --arch vgg16
python prune.py --arch vgg16 --threshold [0.7 0.75 0.8 0.85 0.9 0.95] --samples 50
```

#### ResNet-20

| threshold | Acc.   | FLOPs         | Params.      |
|-----------|--------|---------------|--------------|
| 0.8       | 91.79% | 32.91M(19.4%) | 0.22M(19.4%) |
| 0.75      | 91.73% | 27.46M(32.7%) | 0.20M(26.3%) |
| 0.7       | 90.90% | 20.00M(51.0%) | 0.14M(47.0%) |

```bash
python train.py --arch resnet20 --batch-size 32
python prune.py --arch resnet20 --batch-size 32 --threshold [0.7 0.75 0.8]
```

#### ResNet-32

| threshold | Acc.   | FLOPs         | Params.      |
|-----------|--------|---------------|--------------|
| 0.8       | 92.25% | 44.92M(35.0%) | 0.30M(34.7%) |
| 0.75      | 91.96% | 38.43M(44.4%) | 0.27M(42.3%) |
| 0.7       | 91.27% | 24.51M(64.5%) | 0.20M(57.4%) |

```bash
python train.py --arch resnet32 --batch-size 32
python prune.py --arch resnet32 --batch-size 32 --threshold [0.7 0.75 0.8]
```

#### ResNet-44

| threshold | Acc.   | FLOPs         | Params.      |
|-----------|--------|---------------|--------------|
| 0.8       | 92.72% | 79.45M(18.5%) | 0.51M(22.2%) |
| 0.75      | 92.59% | 70.18M(28.0%) | 0.48M(26.9%) |
| 0.7       | 92.18% | 50.80M(47.9%) | 0.38M(42.7%) |

```bash
python train.py --arch resnet44
python prune.py --arch resnet44 --threshold [0.7 0.75 0.8]
```

#### ResNet-56

| threshold | Acc.   | FLOPs          | Params.      |
|-----------|--------|----------------|--------------|
| 0.8       | 93.22% | 101.75M(19.1%) | 0.64M(24.7%) |
| 0.75      | 92.88% | 80.77M(35.8%)  | 0.57M(33.7%) |
| 0.7       | 92.42% | 62.95M(49.9%)  | 0.48M(44.0%) |

```bash
python train.py --arch resnet56
python prune.py --arch resnet56 --threshold [0.7 0.75 0.8]
```

#### ResNet-110

| threshold | Acc.   | FLOPs          | Params.      |
|-----------|--------|----------------|--------------|
| 0.8       | 92.99% | 134.88M(46.7%) | 0.74M(57.1%) |
| 0.75      | 92.97% | 107.64M(57.5%) | 0.64M(63.1%) |
| 0.7       | 92.72% | 74.81M(70.4%)  | 0.51M(70.4%) |

```bash
python train.py --arch resnet110
python prune.py --arch resnet110 --threshold [0.7 0.75 0.8]
```

#### DenseNet-40

| threshold | Acc.   | FLOPs          | Params.      |
|-----------|--------|----------------|--------------|
| 0.7       | 93.00% | 160.49M(39.4%) | 0.79M(21.0%) |
| 0.65      | 92.29% | 92.19M(65.2%)  | 0.46M(54.4%) |
| 0.6       | 90.50% | 36.75M(86.1%)  | 0.16M(84.1%) |

```bash
python train.py --arch densenet40 --batch-size 64 --optimizer sgd --learning-rate 1e-1 --decay-step 50,150
python prune.py --arch densenet40 --threshold [0.6 0.65 0.7]
```

### Train/prune all models at once

This command will first train all models from scratch and then prune all trained models by a series of preset thresholds.
```bash
source run.sh | tee run.log
```

## Citation

If you find our pruning method useful in your research, please consider citing:
```
@article{li2023graphical,
  title={A graphical approach for filter pruning by exploring the similarity relation between feature maps},
  author={Li, Jiang and Shao, Haijian and Zhai, Shengjie and Jiang, Yingtao and Deng, Xing},
  journal={Pattern Recognition Letters},
  volume={166},
  pages={69--75},
  year={2023},
  publisher={Elsevier}
}
```
