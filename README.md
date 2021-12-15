# SwinTransformerV2-TensorFlow

A TensorFlow implementation of SwinTransformerV2 by Microsoft Research Asia, based on their official implementation of SwinTransformerV1 and their paper on V2. 

Paper on Version 2 (18/11/2021): [[`arXiv`]](https://arxiv.org/pdf/2103.14030.pdf)

Paper on Version 1 (17/08/2021): [[`arXiv`]](https://arxiv.org/pdf/2103.14030.pdf)

## Features:
* TensorFlow 2 implementation of version 1 and 2 of the SwinTransformer, a state-of-the-art backbone for many contemporaty tasks in computer vision. A brief overview of the architectural
changes made in version 2:

![Changes in Version 2](https://github.com/phantng/SwinTransformerV2-TensorFlow/blob/master/changes_in_v2.png?raw=true)

* A pre-norm configuration replaces the previous post-norm configuration, meant to improve training stability in larger models. 
* A scaled cosine attention replaces the dot product attention in V1, with a learnable scaler. 
* A continuous log-spaced relative position bias is used instead of the previous parametric table approach. This is implemented here as a small MLP network and a log transform
on the relative coordinates bias. 

## Requirements:
* numpy==1.21.4
* tensorflow==2.7.0
* tensorflow_addons==0.15.0

## Getting started
Currently writing up. 

## License

This project is licensed under the [MIT license](https://github.com/phantng/SwinTransformerV2-TensorFlow/blob/main/LICENSE).

## Citation
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```
