import os
import math
import datetime
import copy
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from torchvision import transforms
import torchvision.models as models
import argparse
from data_local_loader import test_data_loader, data_loader_with_split
from tqdm import tqdm



try:
    import nsml
    from nsml import DATASET_PATH, IS_ON_NSML
    TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'train')
    VAL_DATASET_PATH = None
except:
    IS_ON_NSML=False
    TRAIN_DATASET_PATH = os.path.join('train')
    VAL_DATASET_PATH = None


def validate(epoch, model):
    global best_acc

    model.eval()
    valid_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, samples in enumerate(val_loader):
            inputs = samples[1].to(device)
            targets = samples[2].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Validation: Epoch=%d, Loss=%.3f, Acc=%.3f' % (epoch, valid_loss / total,  correct / total))


    # Save checkpoint.
    acc = correct / total
    if acc > best_acc:
        # print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        best_acc = acc
        print('Best Acc : %.3f' % best_acc)


def train(epoch,model):
    # train mode
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_i, sample in enumerate(tr_loader):
        optimizer.zero_grad()

        inputs, targets = sample[1], sample[2]
        
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


    if IS_ON_NSML:
        nsml.report(
            summary=True,
            step=epoch,
            epoch_total=epoch_times,
            loss=train_loss / total,
            acc= correct / total
        )

        nsml.save(str(epoch + 1))
    print('Training   Epoch=%d, Loss=%.3f, Acc=%.3f' % (epoch, train_loss / total, correct / total))

def _infer(model, root_path):

    test_loader = test_data_loader(
        root=os.path.join(root_path, 'test_label'),
        phase='test',
        batch_size=64
    )

    res_fc = None
    res_id = None
    print(model)
    for idx, (data_id, image) in enumerate(tqdm(test_loader)):
        image = image.cuda()
        fc = model(image)
        fc = fc.detach().cpu().numpy()

        if idx == 0:
            res_fc = fc
            res_id = data_id
        else:
            res_fc = np.concatenate((res_fc, fc), axis=0)
            res_id = res_id + data_id

    res_cls = np.argmax(res_fc, axis=1)


    return [res_id, res_cls]

def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, 'model.pt'))
        print('Model saved')

    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return _infer(model, path)

    nsml.bind(save=save, load=load, infer=infer)  # 'nsml.bind' function must be called at the end.

if __name__ == '__main__':
    global device
    global val_loader
    global tr_loader
    global optimizer
    global criterion
    global vgg16_ft
    global epoch_times

    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--train_split", type=float, default=0.9)
    args.add_argument("--num_classes", type=int, default=6)
    args.add_argument("--lr", type=int, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=10)
    args.add_argument("--print_iter", type=int, default=10)
    args.add_argument("--eval_split", type=str, default='val')
    args.add_argument("--batch_size", type=int, default=64)

    # reserved for nsml
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)

    config = args.parse_args()

    train_split = config.train_split
    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    eval_split = config.eval_split
    batch_size = config.batch_size
    mode = config.mode



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    vgg16_ft = models.vgg16_bn(pretrained=True)
    for param in vgg16_ft.parameters():
        param.requires_grad = False
    vgg16_ft.features[0] = nn.Conv2d(9,64,kernel_size=(3,3),stride=1,padding=(1,1))
    num_ftrs = vgg16_ft.classifier[6].in_features
    vgg16_ft.classifier[6] = nn.Linear(num_ftrs,10)
    vgg16_ft = vgg16_ft.to(device)

    efficientnet = efficientnet = models.efficientnet_b0(pretrained=True)

    # optimizer
    optimizer = optim.SGD(vgg16_ft.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam()


    # criterion
    criterion = nn.CrossEntropyLoss()

    #lr 
    # lmbda = lambda epoch: 0.65 ** epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

    # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6,8,9], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="exp_range",gamma=0.85)


    model = vgg16_ft
    # model = efficientnet
    if IS_ON_NSML:
        bind_model(model, optimizer)

    if config.pause:
        nsml.paused(scope=locals())
        
    if config.mode =='train':
        # epoch times
        epoch_times = 30
        start_epoch = 0

        best_acc = 0

        tr_loader, val_loader, val_label = data_loader_with_split(
                root=TRAIN_DATASET_PATH,
                train_split=train_split,
                batch_size = batch_size
            )

        for epoch in range(start_epoch, start_epoch + epoch_times):
            train(epoch,model)
            scheduler.step()
            validate(epoch, model)
