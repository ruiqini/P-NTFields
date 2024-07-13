# P-NTFields

>**Progressive Learning for Physics-informed Neural Motion Planning**
\
>[Ruiqi Ni](https://ruiqini.github.io/),
[Ahmed H Qureshi](https://qureshiahmed.github.io/)


<img src="fig/fig.png" width="778.1" height="235.7">

_[Paper](https://www.roboticsproceedings.org/rss19/p063.html) |
[GitHub](https://github.com/ruiqini/P-NTFields) |
[arXiv](https://arxiv.org/abs/2306.00616) |
Published in RSS 2023._

## Introduction

This repository is the official implementation of "Progressive Learning for Physics-informed Neural Motion Planning". 

## Installation

Clone the repository into your local machine:

```
git clone https://github.com/ruiqini/P-NTFields --recursive
```

Install requirements:

```setup
conda env create -f NTFields_env.yml
conda activate NTFields
```

Download datasets and pretrained models, exact and put `datasets/` `Experiments/` to the repository directory:

[Datasets and pretrained model](https://drive.google.com/file/d/1JTIoCYbTZnaMPbmpuM54tzzQG4_hR4Zy/view?usp=sharing)

>The repository directory should look like this:
```
P-NTFields/
├── datasets/
│   ├── arm/    # 6-DOF robot arm, cabinet environment
│   ├── c3d/    # C3D environment
│   ├── gibson/ # Gibson environment
│   └── test/   # box and bunny environment
├── Experiments
│   ├── UR5/   # pretrained model for 6-DOF arm
│   └── Gib_multi/    # pretrained model for Gibson
•   •   •
•   •   •
```

## Pre-processing

To prepare the Gibson data, run:

```
python dataprocessing/preprocess.py --config configs/gibson.txt
```

To prepare the arm data, run:

```
python dataprocessing/preprocess.py --config configs/arm.txt
```

## Testing

To visualize our path in a Gibson environment, run:

```eval
python test/gib_plan.py 
```

To visualize our path in the 6-DOF arm environment, run:

```eval
python test/arm_plan.py 
```

## Training

To train our model in multiple Gibson environment, run:

```train
python train/train_gib_multi.py
```

To train our model in the 6-DOF arm environment, run:

```train
python train/train_arm.py 
```

## Videos

|      Example 1     |       
| :----------------: | 
| ![](fig/real1.gif) |

|      Example 2     |       
| :----------------: | 
| ![](fig/real2.gif) |

|      Example 3     |       
| :----------------: | 
| ![](fig/real3.gif) |

## Citation

Please cite our paper if you find it useful in your research:

```
@article{ni2023progressive,
  title={Progressive Learning for Physics-informed Neural Motion Planning},
  author={Ni, Ruiqi and Qureshi, Ahmed H},
  journal={arXiv preprint arXiv:2306.00616},
  year={2023}
}
```

## Acknowledgement
This implementation takes [EikoNet](https://github.com/Ulvetanna/EikoNet) and [NDF](https://github.com/jchibane/ndf/) as references. We thank the authors for their excellent work.


## License

P-NTFields is released under the MIT License. See the LICENSE file for more details.

