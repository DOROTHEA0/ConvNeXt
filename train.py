import torch
from models.convnext import *
from utils import get_params_groups, create_lr_scheduler
import argparse


def train(opt):
    model = convnext_tiny(num_classes=2)
    p = get_params_groups(model)

    optimizer = torch.optim.AdamW(p, lr=opt.lr)
    # lr_scheduler =





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="number of batch of each epoch")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    opt = parser.parse_args()
    train(opt)
