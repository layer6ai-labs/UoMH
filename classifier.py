# Modified code from https://github.com/kuangliu/pytorch-cifar

import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from classification_models import *
from classification_utils import progress_bar, deterministic_shuffle,class_weights

import numpy as np
import pickle

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--run_name', default="0", type=str)
parser.add_argument('--model', default="res18", type=str)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--do_weight', default=0, type=int)
args = parser.parse_args()

num_classes = 100
max_epochs = 200
pct_val = 0.001
early_stopping_patience = 10000
last_best_epoch = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])

trainvalset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)

N = len(trainvalset.targets)
train_val_perm = deterministic_shuffle(np.arange(N))
num_train = N-round(N*pct_val)

trainset = torch.utils.data.Subset(trainvalset, train_val_perm[:num_train])
valset = torch.utils.data.Subset(trainvalset, train_val_perm[num_train:])

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=100, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.model == "res18":
    net = ResNet18(num_classes=num_classes)
elif args.model == "res34":
    net = ResNet34(num_classes=num_classes)
elif args.model == "vgg":
    net = VGG('VGG19', num_classes=num_classes)
else:
    assert False

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('runs/classification_checkpoints'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./runs/classification_checkpoints/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if args.do_weight == 1:
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([w*len(class_weights) for w in class_weights]).to(device))
elif args.do_weight == 0:
    criterion = nn.CrossEntropyLoss()
else:
    assert False
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

def update_lr(new_lr):
    print("Updating lr:", new_lr)
    for g in optimizer.param_groups:
        g['lr'] = new_lr

class_accs = {split: {i: {cls_id: [] for cls_id in range(num_classes)} for i in range(max_epochs)} for split in ["train", "val", "test"]}
class_scores = {split: {i: {cls_id: [] for cls_id in range(num_classes)} for i in range(max_epochs)} for split in ["train", "val", "test"]}
train_losses = []
val_losses = []

def gather_metrics(outs, targs, split, epoch):
    global class_accs, class_scores
 
    accs = (torch.argmax(outs, 1) == targs).float()

    bs = outs.shape[0]
    correct_probs = F.softmax(outs, dim=1)[torch.arange(bs),targs]

    for label,acc,correct_prob in zip(targs, accs, correct_probs):
        class_accs[split][epoch][label.item()].append(acc.detach().item())
        class_scores[split][epoch][label.item()].append(correct_prob.detach().item())

def print_epoch_acc(epoch, split):
    acc = sum( [ sum(class_accs[split][epoch][i]) for i in range(num_classes)] ) / sum( [len(class_accs[split][epoch][i]) for i in range(num_classes)] )
    acc = round(acc, 2)
    print(f"{split} accuracy at epoch {epoch}: {acc}")

# Training
def train(epoch,trainloader,itr_idx=0):
    global train_losses
    
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        train_losses.append(loss.detach())
        optimizer.step()

        gather_metrics(outputs, targets, "train", epoch)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader.dataset.indices) //  trainloader.batch_size, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, loader, split):
    global best_acc, val_losses, train_losses, class_accs, class_scores, last_best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
      
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            if split == "val":
                val_losses.append(loss.detach())
            
            gather_metrics(outputs, targets, split, epoch)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc and split == "val":
        last_best_epoch = epoch
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('runs/classification_checkpoints'):
            os.mkdir('runs/classification_checkpoints')
            
        torch.save(state, f'./runs/classification_checkpoints/ckpt_{args.run_name}.pth')
        best_acc = acc

    with open(f'./runs/classification_checkpoints/ckpt_{args.run_name}.pickle', 'wb') as handle:
        pickle.dump({
            "val_losses": val_losses, 
            "train_losses": train_losses, 
            "class_accs": class_accs, 
            "class_scores": class_scores
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)

for epoch in range(start_epoch, start_epoch+max_epochs):
    print('\nEpoch: %d' % epoch)
    
    if epoch == 60:
        update_lr(0.02)
    elif epoch == 120:
        update_lr(0.004)
    elif epoch == 160:
        update_lr(0.0008)

    
    train(epoch,trainloader)
    test(epoch, valloader, "val")
    test(epoch, testloader, "test")