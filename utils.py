#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from torch import optim
import os
import random


def make_optimizer(args, model, lr):
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {
            'momentum': args.momentum,
            'dampening': args.dampening,
            'nesterov': args.nesterov
        }
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon,
            'amsgrad': args.amsgrad
        }
    elif args.optimizer == 'ADAMAX':
        optimizer_function = optim.Adamax
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {
            'eps': args.epsilon,
            'momentum': args.momentum
        }
    else:
        raise Exception()

    kwargs['lr'] = lr
    kwargs['weight_decay'] = args.gamma

    return optimizer_function(trainable, **kwargs)


def create_labels(data_path):
    l = []
    for file_ in os.listdir(data_path):
        f = os.path.join(data_path, file_)
        l.append([f, f.split('_')[0].split('\\')[1]])

    return l


def split_labels(l, split_ratio=0.8):
    random.shuffle(l)

    return l[:int(len(l) * split_ratio)], l[int(len(l) * split_ratio):]


def print_hyperparams(args):
    print('-------------------------')
    print(f'For model: {args.model_name}')
    print(f'---> lr: {args.lr}')
    print(f'---> opt: {args.optimizer}')
    print(f'---> mom: {args.momentum}')
    print(f'---> beta1: {args.beta1}')
    print(f'---> beta2: {args.beta2}')
    print(f'---> epsilon: {args.epsilon}')
    print(f'---> gamma: {args.gamma}')