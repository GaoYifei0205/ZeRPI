# ZeRPI

Data and codes for the paper "[ZeRPI: A graph neural network model for zero-shot prediction of RNA-protein interactions](https://www.sciencedirect.com/science/article/abs/pii/S1046202325000167)".

## Requirements

We implement our model on `Python 3.10`. These packages are mainly used:

```
torch                2.2.1+cu118
tensorboard          2.16.2
dgl                  2.1.0+cu118
numpy                1.26.3
scikit-learn         1.4.1.post1
```

## Datasets

You need to download [CD-HIT](https://github.com/weizhongli/cdhit) and [LinearPartition](https://github.com/LinearFold/LinearPartition) to process data.

## Experiments

Run `main.py`, `main_single.py`, and `main_leave_one.py` for training. Run `zero_shot_leave_one`, `zero_shot_10_10`, and `zero_shot_10_9` for inference. For example,

```
python main_single.py --lr 3e-4 --protein_id 18 --use_binary_protein 1 --save 1 --epochs 60 --device 6
```

## Citation

```
@article{GAO202545,
title = {ZeRPI: A graph neural network model for zero-shot prediction of RNA-protein interactions},
journal = {Methods},
volume = {235},
pages = {45-52},
year = {2025},
issn = {1046-2023},
doi = {https://doi.org/10.1016/j.ymeth.2025.01.014},
url = {https://www.sciencedirect.com/science/article/pii/S1046202325000167},
author = {Yifei Gao and Runhan Shi and Gufeng Yu and Yuyang Huang and Yang Yang},
}
```
