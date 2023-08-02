# Prodigy: An Expeditiously Adaptive Parameter-Free Learner
[![Downloads](https://static.pepy.tech/badge/prodigyopt)](https://pepy.tech/project/prodigyopt) [![Downloads](https://static.pepy.tech/badge/prodigyopt/month)](https://pepy.tech/project/prodigyopt)

This is the official repository used to run the experiments in the paper that proposed the Prodigy optimizer. Currently, the code is only in PyTorch.

**Prodigy: An Expeditiously Adaptive Parameter-Free Learner**  
*K. Mishchenko, A. Defazio*  
Paper: https://arxiv.org/pdf/2306.06101.pdf

## Installation and use
To install the package simply run,
```pip install prodigyopt```
Let `net` be the neural network you want to train. Then, you can use the method as follows:
```
from prodigyopt import Prodigy
# you can choose weight decay value based on your problem, 0 by default
opt = Prodigy(net.parameters(), lr=1., weight_decay=weight_decay)
```
Note that by default, Prodigy uses weight decay as in AdamW. 
If you want it to use standard $\ell_2$ regularization instead, use option `decouple=False`.

We also recommend using cosine annealing with the method:
```
# n_epoch is the total number of epochs to train the network
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epoch)
```
Extra care should be taken if you use linear warm-up at the beginning. 
The method might see slow progress and overestimate the learning rate.
To avoid issues with warm-up, use option `safeguard_warmup=True`.  
Based on the interaction with some of the users, we recommend setting `safeguard_warmup=True`,
 `use_bias_correction=True`, and `weight_decay=0.01` when training diffusion models.

## How to cite
If you find our work useful, please consider citing our paper.
```
@article{mishchenko2023prodigy,
    title={Prodigy: An Expeditiously Adaptive Parameter-Free Learner},
    author={Mishchenko, Konstantin and Defazio, Aaron},
    journal={arXiv preprint arXiv:2306.06101},
    year={2023},
    url={https://arxiv.org/pdf/2306.06101.pdf}
}
```
