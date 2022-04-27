import torch
import csv
import numpy as np
import os
from option import args
from PIL import Image
from torchvision import transforms
import utils


def preprocess(img):
    img = Image.open(img).convert('RGB')
    imgs = [img, img.transpose(Image.FLIP_LEFT_RIGHT)]
    transform_list = [
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_list)
    imgs = [transform(i) for i in imgs]
    imgs = [torch.unsqueeze(i, dim=0) for i in imgs]

    return imgs


def test(args, val):
    model = torch.load('./pretrained/{}.pt'.format(args.model_name))
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    rank = torch.Tensor([i for i in range(101)]).cuda()
    error = 0
    count = 0
    for i in val:
        name, age = i
        age = int(age)
        if args.val_img:
            img_path = os.path.join(args.val_img, name)
        else:
            img_path = name

        imgs = preprocess(img_path)
        predict_age = 0
        for img in imgs:
            img = img.to(device)
            output = model(img)
            predict_age += torch.sum(output*rank, dim=1).item()/2
        # print('label:{} \tage:{:.2f}'.format(age, predict_age))
        error += abs(predict_age-age)
        count += 1
    print('---> Test MAE: {:.4f}'.format(error/count))


if __name__ == '__main__':
    labels = utils.create_labels('data/UTKFace')
    train, val = utils.split_labels(labels)
    test(args, val)
