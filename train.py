import os
import torch
import data
import loss
import utils
import time
from option import args
from model import ThinAge, TinyAge
from test import test
import random

models = {'ThinAge': ThinAge, 'TinyAge': TinyAge}


def get_model(pretrained=False):
    model = args.model_name
    assert model in models
    if pretrained:
        path = os.path.join('./pretrained/{}.pt'.format(model))
        assert os.path.exists(path)
        return torch.load(path)
    model = models[model]()

    return model


def main(train, val):
    model = get_model()
    device = torch.device('cuda')
    model = model.to(device)

    args.labels = train
    loader = data.Data(args).train_loader
    rank = torch.Tensor([i for i in range(101)]).cuda()
    for i in range(args.epochs):
        optimizer = utils.make_optimizer(args, model, args.lr)
        model.train()

        running_loss = 0.0
        for j, inputs in enumerate(loader):
            img, label, age = inputs
            img = img.to(device)
            label = label.to(device)
            age = age.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            ages = torch.sum(outputs*rank, dim=1)
            loss1 = loss.kl_loss(outputs, label)
            loss2 = loss.L1_loss(ages, age)
            total_loss = loss1 + loss2
            running_loss += total_loss

            total_loss.backward()
            optimizer.step()

        torch.save(model, './pretrained/{}.pt'.format(args.model_name))
        torch.save(model.state_dict(), './pretrained/{}_dict.pt'.format(args.model_name))

    print('Train:')
    print('Epoch = [{}] \t[MAE: {:.4f}]'.format(i + 1, running_loss / len(loader)))
    test(args, val)


if __name__ == '__main__':
    labels = utils.create_labels('data/UTKFace')
    train, val = utils.split_labels(labels)

    # Search field
    lr_ = [1e-4, 1e-3, 1e-2, 1e-1]
    opt_ = ['SGD', 'ADAM', 'ADAMAX', 'RMSprop']
    mom_ = [0.85, 0.9, 0.95]
    beta1_ = [0.75, 0.8, 0.85, 0.9, 0.95, 0.999]
    beta2_ = [0.75, 0.8, 0.85, 0.9, 0.95, 0.999]
    epsilon_ = [1e-8, 1e-7, 1e-6, 1e-5]
    gamma_ = [0.7, 0.8, 0.9, 0.95]

    args.epochs = 10
    for model_type in ['TinyAge']:
        args.model_name = model_type
        args.lr = lr_[0]
        args.optimizer = opt_[1]
        args.momentum = mom_[1]
        args.beta1 = beta1_[1]
        args.beta2 = beta2_[2]
        args.epsilon = epsilon_[2]
        args.gamma = gamma_[1]

        utils.print_hyperparams(args)
        main(train, val)
