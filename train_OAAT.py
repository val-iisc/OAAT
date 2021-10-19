""" The code is adapted from https://github.com/csdongxian/AWP/tree/main/trades_AWP and https://github.com/cassidylaidlaw/perceptual-advex """
from __future__ import print_function
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from autoaugment import CIFAR10Policy
import models
from perceptual_advex.distances import LPIPSDistance
from perceptual_advex.perceptual_attack_adv import get_lpips_model
from perceptual_advex.perceptual_attack_adv import get_lpips_model_100classes
from utils import Bar, Logger, AverageMeter, accuracy
from utils_awp import TradesAWP
import torchvision
import random
import pickle
from torch.utils.data import Dataset
from PIL import Image
import wandb
from torchvision import datasets, transforms
import torchvision
from autoaugment import CIFAR10Policy
from autoaugment import SVHNPolicy
from torch.utils.data.sampler import SubsetRandomSampler
from defaults import use_default


parser = argparse.ArgumentParser(description='PyTorch OAAT Adversarial Training')
parser.add_argument('--arch', type=str, default='WideResNet34', choices=['ResNet18', 'PreActResNet18','WideResNet34'])
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='resume training from which epoch')
parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100','SVHN'])
parser.add_argument('--data-path', type=str, default='../data',
                    help='where is the dataset')
parser.add_argument('--weight-decay', '--wd', default=3e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf'],
                    help='The threat model')
parser.add_argument('--epsilon', default=16/255, type=float,
                    help='perturbation')
parser.add_argument('--beta', default=2.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES inital value of beta')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='./model-cifar-WideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--resume-model', default='', type=str,
                    help='path of model to resume training')
parser.add_argument('--resume-optim', default='', type=str,
                    help='path of optimizer to resume training')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--awp-gamma', default=0.005, type=float,
                    help='whether or not to add parametric noise')
parser.add_argument('--awp-warmup', default=10, type=int,
                    help='We could apply AWP after some epochs for accelerating.')

parser.add_argument('--swa_save_epoch', default=1, type=int,
                    help='Start saving SWA models after this epoch')
parser.add_argument('--lr_schedule', default='cosine',choices=['cosine', 'step'],
                    help='schedule used for training')
parser.add_argument('--exp_name', default='OAAT',
                    help='name of the method used for training')
parser.add_argument('--beta_final', default=3, type=float,
                    help='the final value of beta at the end of training ')
parser.add_argument('--mixup_alpha', default=0.45, type=float,
                    help='the value of mixup coeeficient in KL loss ')
parser.add_argument('--mixup_epsilon', default=16/255, type=float,
                    help='the epsilon value used to generate mixup attack ')
parser.add_argument('--lpips_weight', default=1, type=float,
                    help='the value of weight of lpips term in inner maximization')
parser.add_argument('--use_CE', default=1, type=int,
                    help='uses CE loss for inner maximization when set to 1 else uses KL loss')
parser.add_argument('--auto', default=1, type=float,
                    help='0 for no autoaugment 0.5 for autoaugment with probabilty 0.5 and 1 for autoaugment with probability 1')
parser.add_argument('--tau_swa_list', type=float, nargs='*', default=[0.995,0.9996,0.9998],  help='The tau values for SWA')
parser.add_argument('--label_smoothing', default=0, type=int,
                    help='put it as 1 it want to use label smoothing for the clean loss in outer minimization')
parser.add_argument('--OAAT_warmup', default=0, type=int,
                    help='put it as 1 if want to use linear warmup of 10 epochs')
parser.add_argument('--use_defaults', type=str, default='NONE' ,choices=['NONE','CIFAR10_RN18', 'CIFAR10_WRN','CIFAR100_WRN', 'CIFAR100_PRN18','SVHN_PRN18'],
                    help='Use None is want to use the hyperparamters passed in the python training command else use the desired set of default hyperparameters')
parser.add_argument('--alternate_iter_eps', default=12/255, type=float,
                    help='the epsilon value after which alternate iters start ')

### args for wandb initialization and logging in wandb ####

parser.add_argument('--wandb-run', default="OAAT")
parser.add_argument('--wandb-notes', default="OAAT")
parser.add_argument('--wandb-project', default="OAAT")
parser.add_argument('--wandb-dir', default="./wandb_log")


args = parser.parse_args()
if args.use_defaults!='NONE':
    args = use_default(args.use_defaults)
print(args)


epsilon = args.epsilon
if args.awp_gamma <= 0.0:
    args.awp_warmup = np.infty
if args.data == 'CIFAR100':
    NUM_CLASSES = 100
elif args.data == 'CIFAR10' or args.data == 'SVHN':
    NUM_CLASSES = 10

# settings
model_dir = args.model_dir
wandb_dir = args.wandb_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(wandb_dir):
    os.makedirs(wandb_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

# setup data loader
class Aug_loader_cifar100(torchvision.datasets.CIFAR100):

    def __getitem__(self, index):
        img, _ = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        aug_1 = self.transform[0](img)
        aug_2 = self.transform[1](img)
        aug_3 = self.transform[2](img)

        return aug_1, aug_2, aug_3, self.targets[index], index

class Aug_loader_cifar10(torchvision.datasets.CIFAR10):

    def __getitem__(self, index):
        img, _ = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        aug_1 = self.transform[0](img)
        aug_2 = self.transform[1](img)
        aug_3 = self.transform[2](img)

        return aug_1, aug_2, aug_3, self.targets[index], index

class Aug_loader_svhn(torchvision.datasets.SVHN):

    def __getitem__(self, index):
        img, _ = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        aug_1 = self.transform[0](img)
        aug_2 = self.transform[1](img)
        aug_3 = self.transform[2](img)

        return aug_1, aug_2, aug_3, self.labels[index], index

if args.data == 'CIFAR10' or args.data =='CIFAR100':
    policy = CIFAR10Policy()
elif args.data == 'SVHN':
    policy = SVHNPolicy()
# setup data loader
if args.auto==1:

    transform_train_main = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),policy,
        transforms.ToTensor(),
    ])

    transform_train_main_SVHN = transforms.Compose([policy, transforms.ToTensor(),])

else:

    transform_train_main = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_train_main_SVHN = transforms.Compose([transforms.ToTensor(),])


transform_train_auto = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),policy,
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

transform_train_auto_SVHN = transforms.Compose([policy,
    transforms.ToTensor(),
])
if args.data == 'CIFAR10' or args.data == 'CIFAR100':
    if args.data == 'CIFAR100':
        trainset = Aug_loader_cifar100(root=args.data_path,train=True,transform=[transform_train_main,transform_train_auto,transform_test], download=True)
    else:
        trainset = Aug_loader_cifar10(root=args.data_path,train=True,transform=[transform_train_main,transform_train_auto,transform_test], download=True)
    valset = getattr(datasets, args.data)(
        root=args.data_path, train=True, download=True, transform=transform_test)
    testset = getattr(datasets, args.data)(
        root=args.data_path, train=False, download=True, transform=transform_test)
elif args.data == 'SVHN':
    trainset = Aug_loader_svhn(root=args.data_path,split='train',transform=[transform_train_main_SVHN,transform_train_auto_SVHN,transform_test], download=True)
    valset = getattr(datasets, args.data)(
        root=args.data_path, split='train', download=True, transform=transform_test)
    testset = getattr(datasets, args.data)(
        root=args.data_path, split='test', download=True, transform=transform_test)

if args.data == 'CIFAR10':
    train_size = 49000
    valid_size = 1000
    test_size  = 10000
    train_indices = list(range(50000))
    val_indices = []
    count = np.zeros(10)
    for index in range(len(trainset)):
        _,_,_, target,_ = trainset[index]
        if(np.all(count==100)):
            break
        if(count[target]<100):
            count[target] += 1
            val_indices.append(index)
            train_indices.remove(index)
elif args.data == 'CIFAR100':
    train_size = 47500
    valid_size = 2500
    test_size  = 10000
    train_indices = list(range(50000))
    val_indices = []
    count = np.zeros(100)
    for index in range(len(trainset)):
        _,_,_, target,_ = trainset[index]
        if(np.all(count==10)):
            break
        if(count[target]<10):
            count[target] += 1
            val_indices.append(index)
            train_indices.remove(index)
elif args.data == 'SVHN':
    train_size = 70757
    valid_size = 2500
    test_size  = 26032
    train_indices = list(range(70757+2500))
    val_indices = []
    count = np.zeros(10)
    for index in range(len(trainset)):
        _,_,_, target,_ = trainset[index]
        if(np.all(count==250)):
            break
        if(count[target]<250):
            count[target] += 1
            val_indices.append(index)
            train_indices.remove(index)


        
print("Overlap indices:",list(set(train_indices) & set(val_indices)))
print("Size of train set:",len(train_indices))
print("Size of val set:",len(val_indices))


train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,sampler=SubsetRandomSampler(train_indices), **kwargs)
val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,sampler=SubsetRandomSampler(val_indices), **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
print('{} dataloader: Done'.format(args.data)) 


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=100, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def perturb_input(model,
                  x_natural,
                  target,args,
                  step_size=2/255,
                  epsilon=8/255,
                  perturb_steps=10,
                  noi=0.001,
                  distance='l_inf'):

    criterion = nn.CrossEntropyLoss()

    batch_size = len(x_natural)
    if distance == 'l_inf':
        
        x_adv = x_natural.detach() + torch.FloatTensor(np.random.uniform(-noi,noi,x_natural.shape)).cuda().detach()
        for step_num in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                if args.use_CE == 1:
                    loss = criterion(model(x_adv), target)
                else:
                    loss = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                        F.softmax(model(x_natural), dim=1),
                                        reduction='sum')
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv



def perturb_input_lpips(model, 
                    data,
                    target, args,
                    step_size=2/255,
                    epsilon=8/255,
                    perturb_steps=10,
                    noi=0.001,lpips_weight=1,lpips_distance=0):


    batch_size = len(data)
    eps = epsilon
    bounds = np.array([[0,1],[0,1],[0,1]])
    eps_iter = step_size
    #assert not model.training, 'Model is in  training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps,eps,(B,C,H,W))).cuda()
    noise  = torch.clamp(noise,-eps,eps)
    
    y_imgs = data
    steps = perturb_steps
    for step in range(steps):
        img = data + noise
        img = Variable(img,requires_grad=True)
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass
        out  = model(img)
        if args.use_CE == 1:
            cost = torch.nn.CrossEntropyLoss()(out,target.cuda()) - lpips_weight*torch.mean(lpips_distance(y_imgs, img))
        else:
            cost = F.kl_div(F.log_softmax(model(img), dim=1),F.softmax(model(data), dim=1), reduction='batchmean')  - lpips_weight*torch.mean(lpips_distance(y_imgs, img))
        #backward pass
        cost.backward()
        #get gradient of loss wrt data
        per =  torch.sign(img.grad.data)
        #convert eps 0-1 range to per channel range 
        per[:,0,:,:] = (eps_iter * (bounds[0,1] - bounds[0,0])) * per[:,0,:,:]
        if(per.size(1)>1):
            per[:,1,:,:] = (eps_iter * (bounds[1,1] - bounds[1,0])) * per[:,1,:,:]
            per[:,2,:,:] = (eps_iter * (bounds[2,1] - bounds[2,0])) * per[:,2,:,:]
        #  ascent
        adv = img.data + per.cuda()
        #clip per channel data out of the range
        img.requires_grad =False
        img[:,0,:,:] = torch.clamp(adv[:,0,:,:],bounds[0,0],bounds[0,1])
        if(per.size(1)>1):
            img[:,1,:,:] = torch.clamp(adv[:,1,:,:],bounds[1,0],bounds[1,1])
            img[:,2,:,:] = torch.clamp(adv[:,2,:,:],bounds[2,0],bounds[2,1])
        img = img.data
        noise = img - data
        noise  = torch.clamp(noise,-eps,eps)
    img = torch.clamp(data + noise, 0.0, 1.0)
    return img


def train(model, train_loader, optimizer, epoch, awp_adversary, start_wa, tau_list, exp_avgs, vareps, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    losses_clean = AverageMeter()
    top1_clean = AverageMeter()
    end = time.time()

    if epoch <= (args.epochs//4):
        var_step_size = vareps/2.0
        var_pert_steps = 5
    else:
        var_step_size = vareps/4.0
        var_pert_steps = 10

    print('epoch: {}'.format(epoch))
    bar = Bar('Processing', max=len(train_loader))

    for batch_idx, (data, data_auto, data_test,  target, index) in enumerate(train_loader):
        if args.auto !=0:
            data = torch.cat((data[:data.size()[0]//2,:,:,:],data_auto[data.size()[0]//2:,:,:,:]),dim=0)
        else:
            data = data
        x_natural, target = data.to(device), target.to(device)

        noi = min(4/255.0, vareps)
        x_adv = perturb_input(model=model,
                              x_natural=x_natural,
                              target=target, args = args,
                              step_size=var_step_size,
                              epsilon=vareps,
                              perturb_steps=var_pert_steps,
                              noi=noi)
        model.train()
        if epoch >= args.awp_warmup:
            x_adv_awp = torch.clamp(x_natural + 2*(x_adv - x_natural), 0, 1)
            awp = awp_adversary.calc_awp(inputs_adv=x_adv_awp,
                                         inputs_clean=x_natural,
                                         targets=target,
                                         beta=args.beta)
            awp_adversary.perturb(awp)

        optimizer.zero_grad()
        logits_adv = model(x_adv)
        loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1),
                               F.softmax(model(x_natural), dim=1),
                               reduction='batchmean')
        logits = model(x_natural)
        if args.label_smoothing == 1:
            criterion_CE = LabelSmoothingLoss()
            loss_natural = criterion_CE(logits, target) 
        else:
            loss_natural = F.cross_entropy(logits, target)
        loss = loss_natural + args.beta * loss_robust
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)


        prec1, prec5 = accuracy(logits_adv, target, topk=(1, 5))
        prec1_clean, prec5_clean = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), x_natural.size(0))
        losses_clean.update(loss_natural.item(),x_natural.size(0))
        top1.update(prec1.item(), x_natural.size(0))
        top1_clean.update(prec1_clean.item(), x_natural.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        for start_ep, tau, new_state_dict in zip(start_wa, tau_list, exp_avgs):
            if epoch == start_ep:
                for key,value in model.state_dict().items():
                    new_state_dict[key] = value
            elif epoch > start_ep:
                for key,value in model.state_dict().items():
                    new_state_dict[key] = (1-tau)*value + tau*new_state_dict[key]
            else:
                pass

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total:{total:}| ETA:{eta:}| Loss:{loss:.4f}| top1:{top1:.2f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg)
        bar.next()
    bar.finish()
    return losses.avg, top1.avg, losses_clean.avg, top1_clean.avg, exp_avgs


def train_lpips_alt(model, train_loader, optimizer, epoch, awp_adversary, start_wa, tau_list, exp_avgs, vareps, alpha, lpips_weight, lpips_distance, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    losses_clean = AverageMeter()
    top1_clean = AverageMeter()
    end = time.time()
    var_step_size = vareps/4.0

    print('epoch: {}'.format(epoch))
    bar = Bar('Processing', max=len(train_loader))

    for batch_idx, (data, data_auto, data_test,  target, index) in enumerate(train_loader):
        if args.auto != 0:
            data = torch.cat((data[:data.size()[0]//2,:,:,:],data_auto[data.size()[0]//2:,:,:,:]),dim=0)
        else:
            data = data

        x_natural, target = data.to(device), target.to(device)
        if batch_idx % 2 == 0:
          x_adv = perturb_input(model=model,
                                  x_natural=x_natural,
                                  target=target, args = args,
                                  step_size=args.mixup_epsilon/4.0,
                                  epsilon=args.mixup_epsilon,
                                  perturb_steps=10,
                                  noi=4/255.0)
        else:
            x_adv = perturb_input_lpips(model=model,                                        
                                       data=x_natural,
                                       target=target, args = args,
                                       step_size=var_step_size,
                                       epsilon=vareps,
                                       perturb_steps=10,
                                       noi=4/255.0,lpips_weight=lpips_weight,lpips_distance=lpips_distance)

        model.train()
        if epoch >= args.awp_warmup:
            awp = awp_adversary.calc_awp(inputs_adv=x_adv,
                                         inputs_clean=x_natural,
                                         targets=target,
                                         beta=args.beta)
            awp_adversary.perturb(awp)

        optimizer.zero_grad()
        logits_adv = model(x_adv)
        logits_clean = model(x_natural)
        
        if batch_idx % 2 == 0:
            x_adv_vareps = torch.min(torch.max(x_adv, x_natural - vareps), x_natural + vareps)
            logits_adv_vareps = model(x_adv_vareps)
            loss_robust = F.kl_div(F.log_softmax(logits_adv_vareps, dim=1),
                               alpha*F.softmax(logits_clean, dim=1) + (1-alpha)*F.softmax(logits_adv, dim=1),
                               reduction='batchmean')
        else:
            loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1),
                       F.softmax(logits_clean, dim=1),
                       reduction='batchmean')

        if args.label_smoothing == 1:
            criterion_CE = LabelSmoothingLoss()
            loss_natural = criterion_CE(logits_clean, target) 
        else:
            loss_natural = F.cross_entropy(logits_clean, target)
        loss = loss_natural + args.beta * loss_robust
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)

        prec1, prec5 = accuracy(logits_adv, target, topk=(1, 5))
        prec1_clean, prec5_clean = accuracy(logits_clean, target, topk=(1, 5))
        losses.update(loss.item(), x_natural.size(0))
        losses_clean.update(loss_natural.item(),x_natural.size(0))
        top1.update(prec1.item(), x_natural.size(0))
        top1_clean.update(prec1_clean.item(), x_natural.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        
        for start_ep, tau, new_state_dict in zip(start_wa, tau_list, exp_avgs):
            if epoch == start_ep:
                for key,value in model.state_dict().items():
                    new_state_dict[key] = value
            elif epoch > start_ep:
                for key,value in model.state_dict().items():
                    new_state_dict[key] = (1-tau)*value + tau*new_state_dict[key]
            else:
                pass

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total:{total:}| ETA:{eta:}| Loss:{loss:.4f}| top1:{top1:.2f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg)
        bar.next()
    bar.finish()
    return losses.avg, top1.avg, losses_clean.avg, top1_clean.avg, exp_avgs

def test(model, test_loader, criterion):
    global best_acc
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(test_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total: {total:}| ETA: {eta:}| Loss:{loss:.4f}| top1: {top1:.2f}'.format(
                batch=batch_idx + 1,
                size=len(test_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg)
            bar.next()
    bar.finish()
    return losses.avg, top1.avg

def adjust_learning_rate_linear(optimizer, epoch, args):
    lr = epoch*(args.lr/10)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_cosine(optimizer, epoch, args):
    if args.OAAT_warmup == 1:
        lr = args.lr * 0.5 * (1 + np.cos((epoch - 11) / (args.epochs-10) * np.pi))
    else:
        lr = args.lr * 0.5 * (1 + np.cos((epoch - 1) / args.epochs * np.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate_step(optimizer, epoch, args):
    lr = args.lr
    if epoch >= 75*args.epochs/110:
        lr = args.lr * 0.1
    if epoch >= 90*args.epochs/110:
        lr = args.lr * 0.01
    if epoch >= 100*args.epochs/110:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    model = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    proxy = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
    awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma)

    criterion = nn.CrossEntropyLoss()
    
    wandb.init(name=args.wandb_run,notes = args.wandb_notes,project = args.wandb_project,dir = args.wandb_dir,config=args)

    logger = Logger(os.path.join(model_dir, 'log.txt'), title=args.arch)
    logger.set_names(['Learning Rate',
                      'Adv Train Loss', 'Nat Train Loss', 'Nat Val Loss',
                      'Adv Train Acc.', 'Nat Train Acc.', 'Nat Val Acc.'])

    if args.resume_model:
        model.load_state_dict(torch.load(args.resume_model, map_location=device))
    if args.resume_optim:
        optimizer.load_state_dict(torch.load(args.resume_optim, map_location=device))

    start_wa = [1, int(3*args.epochs/4)+1, int(3*args.epochs/4)+1]
    tau_list = args.tau_swa_list
    exp_avgs = []
    for i in range(len(tau_list)):
        model_tau = getattr(models, args.arch)(num_classes=NUM_CLASSES)
        start_ept=start_wa[i]
        tau = tau_list[i]
        model_tau.cuda()
        model_tau = torch.nn.DataParallel(model_tau)
        if start_ept >= args.start_epoch:
            model_tau = model
        else:
          model_tau.load_state_dict(torch.load(str(args.model_dir)+"/"+'{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(args.exp_name, start_ept, tau, args.data, args.lpips_weight, args.mixup_alpha, args.auto, 1, args.beta_final, args.weight_decay, args.start_epoch-1)))
        exp_avgs.append(model_tau.state_dict())
    vareps  = 4/255
    alpha = 1
    beta_initial = args.beta


    for epoch in range(args.start_epoch, args.epochs + 1):

        if epoch==(int(3*args.epochs/4)+1):
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # adjust learning rate for SGD
        if args.OAAT_warmup == 1:
            if epoch<=10:
                lr = adjust_learning_rate_linear(optimizer, epoch, args)
            else:
                if args.lr_schedule == 'cosine':
                    lr = adjust_learning_rate_cosine(optimizer, epoch, args)
                elif args.lr_schedule == 'step':
                    lr = adjust_learning_rate_step(optimizer, epoch, args)

        else:
            if args.lr_schedule == 'cosine':
                lr = adjust_learning_rate_cosine(optimizer, epoch, args)
            elif args.lr_schedule == 'step':
                lr = adjust_learning_rate_step(optimizer, epoch, args)

        # adversarial training
        if epoch>(args.epochs//4):
            vareps = epoch*epsilon/args.epochs
        if vareps > args.alternate_iter_eps:
            if epoch == args.start_epoch:
                start_epoch=1
            else:
                start_epoch = args.start_epoch
            if args.data == 'CIFAR100':
                lpips_model = get_lpips_model_100classes(args.arch, load_state_dict = str(args.model_dir)+"/"+'{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(args.exp_name, 1, 0.995, args.data, args.lpips_weight, args.mixup_alpha, args.auto, start_epoch, args.beta_final, args.weight_decay, epoch-1))
            else:
                lpips_model = get_lpips_model(args.arch, load_state_dict = str(args.model_dir)+"/"+'{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(args.exp_name, 1, 0.995, args.data, args.lpips_weight, args.mixup_alpha, args.auto, start_epoch, args.beta_final, args.weight_decay, epoch-1))
            alpha = alpha - args.mixup_alpha/(args.epochs - int(3*args.epochs/4) + 1)
            lpips_weight = args.lpips_weight*(epoch - int(3*args.epochs/4))/(args.epochs - int(3*args.epochs/4))
            args.beta = beta_initial + (args.beta_final - beta_initial)* (epoch - int(3*args.epochs/4))/(args.epochs - int(3*args.epochs/4))
            lpips_model = lpips_model.cuda()
            lpips_distance = LPIPSDistance(lpips_model)
            adv_loss, adv_acc, clean_loss, clean_acc, exp_avgs = train_lpips_alt(model, train_loader, optimizer, epoch, awp_adversary, start_wa, tau_list, exp_avgs, vareps, alpha, lpips_weight, lpips_distance, args)
        else:
            adv_loss, adv_acc, clean_loss, clean_acc, exp_avgs = train(model, train_loader, optimizer, epoch, awp_adversary, start_wa, tau_list, exp_avgs, vareps, args)

        # evaluation and logging
        wandb.log({'Adv Loss (Train set) (Beta*KL(Adv||Clean)': adv_loss},step=epoch)
        wandb.log({'Adv Acc @ vareps (Train set)': adv_acc},step=epoch)
        wandb.log({'Clean Loss (Train set)': clean_loss},step=epoch)
        wandb.log({'Clean Acc (Train set)': clean_acc},step=epoch)
        print('================================================================')

        val_loss, val_acc = test(model, val_loader, criterion)
        wandb.log({'CE loss on clean samples (Val set)': val_loss},step=epoch)
        wandb.log({'Clean Acc (Val set)': val_acc},step=epoch)
        print('================================================================')

        logger.append([lr, adv_loss, clean_loss, val_loss, adv_acc, clean_acc, val_acc])


        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, '{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(args.exp_name, args.data, args.lpips_weight, args.mixup_alpha, args.auto, args.start_epoch, args.beta_final, args.weight_decay, epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, '{}_{}_{}_{}_{}_{}_{}_{}_{}.tar'.format(args.exp_name, args.data, args.lpips_weight, args.mixup_alpha, args.auto, args.start_epoch, args.beta_final, args.weight_decay, epoch)))

        if epoch > args.swa_save_epoch-1:
            for idx, start_ep, tau, new_state_dict in zip(range(len(tau_list)), start_wa, tau_list, exp_avgs):
                if start_ep <= epoch:
                    torch.save(new_state_dict,str(args.model_dir)+"/"+'{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(args.exp_name, start_ep, tau, args.data, args.lpips_weight, args.mixup_alpha, args.auto, args.start_epoch, args.beta_final, args.weight_decay, epoch))
            

if __name__ == '__main__':
    main()
