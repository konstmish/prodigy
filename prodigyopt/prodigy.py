import math
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.optim
import logging
import os
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class Prodigy(torch.optim.Optimizer):
    r"""
    Implements Adam with Prodigy step-sizes.
    Leave LR set to 1 unless you encounter instability.
   
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the Prodigy learning rate.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        beta3 (float):
            coefficients for computing the Prodidy stepsize using running averages.
            If set to None, uses the value of square root of beta2 (default: None).
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        decouple (boolean):
            Use AdamW style decoupled weight decay
        use_bias_correction (boolean):
            Turn on Adam's bias correction. Off by default.
        safeguard_warmup (boolean):
            Remove lr from the denominator of D estimate to avoid issues during warm-up stage. Off by default.
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        d_coef (float):
            Coefficient in the expression for the estimate of d (default 1.0).
            Values such as 0.5 and 2.0 typically work as well. 
            Changing this parameter is the preferred way to tune the method.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate.
            Default is inf, for unrestricted. Values like 1.02 give a kind of learning
            rate warmup effect.
        fsdp_in_use (bool):
            If you're using sharded parameters, this should be set to True. The optimizer
            will attempt to auto-detect this, but if you're using an implementation other
            than PyTorch's builtin version, the auto-detection won't work.
        slice_p (int): Reduce memory usage by calculating LR adaptation statistics on only every 
            pth entry of each tensor. For values greater than 1 this an an approximation to standard 
            Prodigy. Values ~11 are reasonable (default 1).
    """
    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.999), beta3=None,
                 eps=1e-8, weight_decay=0, decouple=True, 
                 use_bias_correction=False, safeguard_warmup=False,
                 d0=1e-6, d_coef=1.0, growth_rate=float('inf'),
                 fsdp_in_use=False,
                 slice_p=1):
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if decouple and weight_decay > 0:
            print(f"Using decoupled weight decay")

       
        defaults = dict(lr=lr, betas=betas, beta3=beta3,
                        eps=eps, weight_decay=weight_decay,
                        d=d0, d0=d0, d_max=d0,
                        d_numerator=0.0, d_coef=d_coef,
                        k=0, growth_rate=growth_rate,
                        use_bias_correction=use_bias_correction,
                        decouple=decouple, safeguard_warmup=safeguard_warmup,
                        fsdp_in_use=fsdp_in_use,
                        slice_p=slice_p)
        self.d0 = d0
        super().__init__(params, defaults)
        self.init_step()

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True


    def init_step(self):
        self.d_denom = 0.0

        #write values that are taken from the first param group but used for all param groups in
        #step_parameter() to self, in order to match the original code behaviour
        #TODO this can probably be simplified; depending on the variable, the values can be assumed
        #identical for all groups, and taken directly from group[] by step_parameter()

        group = self.param_groups[0]
        use_bias_correction = group['use_bias_correction']
        self.beta1, self.beta2 = group['betas']
        self.beta3 = group['beta3']
        if self.beta3 is None:
            self.beta3 = math.sqrt(self.beta2)
        k = group['k']

        self.d = group['d']
        lr = max(group['lr'] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - self.beta2**(k+1))**0.5) / (1 - self.beta1**(k+1))
        else:
            bias_correction = 1

        self.dlr = self.d*lr*bias_correction
       
        self.decouple = group['decouple']
        self.fsdp_in_use = group['fsdp_in_use']

        self.d_numerator = group['d_numerator']
        self.d_numerator *= self.beta3

        for group in self.param_groups:
            group_lr = group['lr']
            if group_lr not in [lr, 0.0]:
                raise RuntimeError(f"Setting different lr values in different parameter groups is only supported for values of 0")

        
    def step_parameter(self, p, group, i):
        if p.grad is None:
            return
        decay = group['weight_decay']
        k = group['k']
        eps = group['eps']
        group_lr = group['lr']
        d0 = group['d0']
        safeguard_warmup = group['safeguard_warmup']
        slice_p = group['slice_p']
        if hasattr(p, "_fsdp_flattened"):
            self.fsdp_in_use = True

        grad = p.grad.data

        # Apply weight decay (coupled variant)
        if decay != 0 and not self.decouple:
            grad.add_(p.data, alpha=decay)

        state = self.state[p]

        # State initialization
        if 'step' not in state:
            state['step'] = 0
            state['s'] = torch.zeros_like(p.data.flatten()[::slice_p]).detach()
            if p.count_nonzero() > 0:
                state['p0'] = p.flatten()[::slice_p].detach().clone()
            else:
                state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)

            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data).detach()
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data).detach()

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
       
        s = state['s']
        p0 = state['p0']

        if group_lr > 0.0:
            # we use d / d0 instead of just d to avoid getting values that are too small
            sliced_grad = grad.flatten()[::slice_p]
            self.d_numerator += (self.d / d0) * self.dlr * torch.dot(sliced_grad, p0.data - p.data.flatten()[::slice_p]).item()

            # Adam EMA updates
            exp_avg.mul_(self.beta1).add_(grad, alpha=self.d * (1-self.beta1))
            exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=self.d * self.d * (1-self.beta2))

            if safeguard_warmup:
                s.mul_(self.beta3).add_(sliced_grad, alpha=((self.d / d0) * self.d))
            else:
                s.mul_(self.beta3).add_(sliced_grad, alpha=((self.d / d0) * self.dlr))
            self.d_denom += s.abs().sum().item()


        state['step'] += 1

        denom = exp_avg_sq.sqrt().add_(self.d * eps)

        # Apply weight decay (decoupled variant)
        if decay != 0 and self.decouple:
            p.data.add_(p.data, alpha=-decay * self.dlr)

        ### Take step
        p.data.addcdiv_(exp_avg, denom, value=-self.dlr)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        #if step_parameter() was called for all parameter during a fused backpass,
        #all gradients are None and this loop won't do anything:
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                self.step_parameter(p, group, i)

        self.calculate_d()
        self.init_step() #first init_step is called in __init__
        return loss
        
    def calculate_d(self):
        group = self.param_groups[0]
        d_max = group['d_max']
        d_coef = group['d_coef']
        growth_rate = group['growth_rate']
        lr = max(group['lr'] for group in self.param_groups)

        d_hat = self.d #FIXME no effect, is overwritten; but matches original code

        # if we have not done any progres, return
        # if we have any gradients available, will have d_denom > 0 (unless \|g\|=0)
        if self.d_denom != 0:
            if lr > 0.0: #FIXME lr is max of group lr - must always be > 0; but matches original code
                if self.fsdp_in_use:
                    dist_tensor = torch.zeros(2).cuda()
                    dist_tensor[0] = self.d_numerator
                    dist_tensor[1] = self.d_denom
                    dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                    global_d_numerator = dist_tensor[0]
                    global_d_denom = dist_tensor[1]
                else:
                    global_d_numerator = self.d_numerator
                    global_d_denom = self.d_denom

                d_hat = d_coef * global_d_numerator / global_d_denom
                if self.d == group['d0']:
                    self.d = max(self.d, d_hat)
                d_max = max(d_max, d_hat)
                self.d = min(d_max, self.d * growth_rate)

            for group in self.param_groups:
                group['d_numerator'] = global_d_numerator #FIXME would fail unassigned if lr was <= 0; but matches original code
                group['d_denom'] = global_d_denom
                group['d'] = self.d
                group['d_max'] = d_max
                group['d_hat'] = d_hat
                group['k'] = group['k'] + 1

