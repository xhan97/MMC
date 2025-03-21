import math
import numpy as np


def adjust_learning_rate(args, optimizer, epoch):
    """Adjust learning rate during training."""
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate**3)
        lr = (
            eta_min
            + (lr - eta_min) * (1 + math.cos(math.pi * epoch * 3 / args.epochs)) / 2
        )
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr *= args.lr_decay_rate**steps

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
