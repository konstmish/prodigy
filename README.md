# Prodigy: An Expeditiously Adaptive Parameter-Free Learner
[![Downloads](https://static.pepy.tech/badge/prodigyopt)](https://pepy.tech/project/prodigyopt) [![Downloads](https://static.pepy.tech/badge/prodigyopt/month)](https://pepy.tech/project/prodigyopt)

This is the official repository used to run the experiments in the paper that proposed the Prodigy optimizer. The optimizer is implemented in PyTorch.

**Prodigy: An Expeditiously Adaptive Parameter-Free Learner**  
*K. Mishchenko, A. Defazio*  
Paper: https://arxiv.org/pdf/2306.06101.pdf

## Installation
To install the package, simply run
```pip install prodigyopt```
## How to use
Let `net` be the neural network you want to train. Then, you can use the method as follows:
```
from prodigyopt import Prodigy
# you can choose weight decay value based on your problem, 0 by default
opt = Prodigy(net.parameters(), lr=1., weight_decay=weight_decay)
```
Note that by default, Prodigy uses weight decay as in AdamW. 
If you want it to use standard $\ell_2$ regularization (as in Adam), use option `decouple=False`. 
We recommend using `lr=1.` (default) for all networks. If you want to force the method to estimate a smaller or larger learning rate, 
it is better to change the value of `d_coef` (1.0 by default). Values of `d_coef` above 1, such as 2 or 10, 
will force a larger estimate of the learning rate; set it to 0.5 or even 0.1 if you want a smaller learning rate.

We also recommend using cosine annealing with the method:
```
# n_epoch is the total number of epochs to train the network
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
```
We do not recommend using restarts in cosine annealing, so we suggest setting `T_max=total_steps`, where 
`total_steps` should be the number of times `scheduler.step()` is called. 

Extra care should be taken if you use linear warm-up at the beginning: 
The method will see slow progress due to the initially small base learning rate, 
so it might overestimate `d`.
To avoid issues with warm-up, use option `safeguard_warmup=True`.  
Based on the interaction with some of the users, we recommend setting `safeguard_warmup=True`,
 `use_bias_correction=True`, and `weight_decay=0.01` when training diffusion models.  

See [this Google Colab](https://colab.research.google.com/drive/1TrhEfI3stJ-yNp7_ZxUAtfWjj-Qe_Hym?usp=sharing) 
for a toy example of how one can use Prodigy to train ResNet-18 on Cifar10 (test accuracy 80% after 20 epochs).

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
