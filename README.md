# RedMotion: Motion Prediction via Redundancy Reduction [![arXiv](https://img.shields.io/badge/arXiv-2306.10840-b31b1b.svg)](https://arxiv.org/abs/2306.10840)
TL;DR: Transformer model for motion prediction that incorporates two types of redundancy reduction.

## Overview
![Model architecture](red-motion-model.png "Model architecture")

RedMotion model. Our model consists of two encoders. The ego trajectory encoder generates an embedding for the past trajectory of the ego agent. The road environment encoder generates a set of road environment descriptors as context embedding. Both embeddings are fused via cross-attention to yield trajectory proposals per agent.

## Getting started 
Coming soon...

## Prepare waymo open motion prediction dataset
Register and download the dataset from [here](https://waymo.com/open).
Clone [this repo](https://github.com/kbrodt/waymo-motion-prediction-2021) and use the prerender script as described in the readme.

### Acknowledgements
The local attention ([Beltagy et al., 2020](https://arxiv.org/abs/2004.05150)) and cross-attention ([Chen et al., 2021](https://arxiv.org/abs/2103.14899)) implementations are from lucidrain's [vit_pytorch](https://github.com/lucidrains/vit-pytorch) library.
The baseline DualMotionViT model builds upon the work by [Konev et al., 2022](https://arxiv.org/abs/2206.02163).
