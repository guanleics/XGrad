import math
import torch
from torch.optim.optimizer import Optimizer

class SGD(Optimizer):
    r"""Implements SGD algorithm with weight prediction.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, dampening=0,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Invalid alpha parameter at index 0: {}".format(momentum))
        if not 0.0 <= dampening < 1.0:
            raise ValueError("Invalid dampening parameter at index 0: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)

    def get_params(self, module, clone):
        if clone:
            state_dict = module.state_dict()
            for key in state_dict:
                state_dict[key] = state_dict[key].clone()
        else:
            buffered_state_dicts = deque(maxlen=1)
            state_dict = module.state_dict()
            for key in state_dict:
                if "running_" in key:
                    continue
                if "mask" in key:
                    buffered_state_dicts[key] = state_dict[key].clone()
                else:
                    buffered_state_dicts[key].copy_(state_dict[key])
            state_dict = buffered_state_dicts
        return state_dict

    def set_params(self, state_dict, module):
        cur_state_dict = module.state_dict()
        for key in state_dict:
            state_dict[key] = cur_state_dict[key]
            #if "running_" in key or "mask" in key:
            #    state_dict[key] = cur_state_dict[key]
        module.load_state_dict(state_dict, False)

        # Load the mask.
        for key in state_dict:
            if "mask" in key:
                attribute_names = key.split(".")
                attribute = module
                for attribute_name in attribute_names:
                    attribute = getattr(attribute, attribute_name)
                # NOTE: Do we need to clone here?
                attribute = state_dict[key]

    def load_predicted_weights(self, s):
        # param_groups = list(self.parameters())
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                buf = param_state['momentum_buffer']
                step_size = group['lr']
                p.data.add_(-step_size * s, buf)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                momentum_buffer = state['momentum_buffer']
                momentum = group['momentum']
                dampening = group['dampening']
                momentum_buffer.mul_(momentum).add_(1-dampening, grad)
                p.data.add_(-group['lr'], momentum_buffer)

        return loss
