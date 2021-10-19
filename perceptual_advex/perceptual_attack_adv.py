from typing import Optional, Union
import torch
import torchvision.models as torchvision_models
from torchvision.models.utils import load_state_dict_from_url
import math
from torch import nn
from torch.nn import functional as F
from typing_extensions import Literal

from .distances_adv import normalize_flatten_features, LPIPSDistance
from .utilities import MarginLoss
from .models import AlexNetFeatureModel, CifarAlexNet, FeatureModel
from . import utilities


_cached_alexnet: Optional[AlexNetFeatureModel] = None
_cached_alexnet_cifar: Optional[AlexNetFeatureModel] = None

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################ WideResNet34 model #####################################################


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        x_layer1 = out
        out = self.block2(out)
        x_layer2 = out
        out = self.block3(out)
        x_layer3 = out
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return (x_layer1, x_layer2, x_layer3)


    def features(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        x_layer1 = out
        out = self.block2(out)
        x_layer2 = out
        out = self.block3(out)
        x_layer3 = out
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return (x_layer1, x_layer2, x_layer3)

def WideResNet34(num_classes=10):
    return WideResNet(depth=34, num_classes=num_classes)

################################################################################## ResNet18 model #########################################################        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        x_layer1 = out
        out = self.layer2(out)
        x_layer2 = out
        out = self.layer3(out)
        x_layer3 = out
        out = self.layer4(out)
        x_layer4 = out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return (x_layer1, x_layer2, x_layer3, x_layer4)

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        x_layer1 = out
        out = self.layer2(out)
        x_layer2 = out
        out = self.layer3(out)
        x_layer3 = out
        out = self.layer4(out)
        x_layer4 = out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return (x_layer1, x_layer2, x_layer3, x_layer4)

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

###################################################################  PreactResNet18 model ###############################################################


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        x_layer1 = out
        out = self.layer2(out)
        x_layer2 = out
        out = self.layer3(out)
        x_layer3 = out
        out = self.layer4(out)
        x_layer4 = out
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return (x_layer1, x_layer2, x_layer3, x_layer4)

    def features(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        x_layer1 = out
        out = self.layer2(out)
        x_layer2 = out
        out = self.layer3(out)
        x_layer3 = out
        out = self.layer4(out)
        x_layer4 = out
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return (x_layer1, x_layer2, x_layer3, x_layer4)

def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes)






def get_lpips_model(
    lpips_model_spec: Union[
        Literal['self', 'resnet'],
        FeatureModel,
    ],
    model: Optional[FeatureModel] = None,load_state_dict=0
) -> FeatureModel:
    global _cached_alexnet, _cached_alexnet_cifar

    lpips_model: FeatureModel

    if lpips_model_spec == 'self':
        if model is None:
            raise ValueError(
                'Specified "self" for LPIPS model but no model passed'
            )
        return model
    elif lpips_model_spec == 'alexnet':
        if _cached_alexnet is None:
            alexnet_model = torchvision_models.alexnet(pretrained=True)
            _cached_alexnet = AlexNetFeatureModel(alexnet_model)
        lpips_model = _cached_alexnet
        if torch.cuda.is_available():
            lpips_model.cuda()
    elif lpips_model_spec == 'alexnet_cifar':
        if _cached_alexnet_cifar is None:
            alexnet_model = CifarAlexNet()
            _cached_alexnet_cifar = AlexNetFeatureModel(alexnet_model)
        lpips_model = _cached_alexnet_cifar
        if torch.cuda.is_available():
            lpips_model.cuda()
        try:
            state = torch.load('./alexnet_cifar.pt')
        except FileNotFoundError:
            state = load_state_dict_from_url(
                'https://perceptual-advex.s3.us-east-2.amazonaws.com/'
                'alexnet_cifar.pt',
                progress=True,
            )
        lpips_model.load_state_dict(state['model'])
    elif lpips_model_spec == 'ResNet18':
        lpips_model = ResNet18(num_classes = 10)
        lpips_model = torch.nn.DataParallel(lpips_model)
        lpips_model.load_state_dict(torch.load(load_state_dict))
        lpips_model.eval()

    elif lpips_model_spec == 'WideResNet34':
        lpips_model = WideResNet34(num_classes = 10)
        lpips_model = torch.nn.DataParallel(lpips_model)
        lpips_model.load_state_dict(torch.load(load_state_dict))
        lpips_model.eval()

    elif lpips_model_spec == 'PreActResNet18':
        lpips_model = PreActResNet18(num_classes = 10)
        lpips_model = torch.nn.DataParallel(lpips_model)
        lpips_model.load_state_dict(torch.load(load_state_dict))
        lpips_model.eval()

    elif isinstance(lpips_model_spec, str):
        raise ValueError(f'Invalid LPIPS model "{lpips_model_spec}"')
    else:
        lpips_model = lpips_model_spec

    lpips_model.eval()
    return lpips_model


def get_lpips_model_100classes(
    lpips_model_spec: Union[
        Literal['self', 'resnet'],
        FeatureModel,
    ],
    model: Optional[FeatureModel] = None,load_state_dict=0
) -> FeatureModel:
    global _cached_alexnet, _cached_alexnet_cifar

    lpips_model: FeatureModel

    if lpips_model_spec == 'self':
        if model is None:
            raise ValueError(
                'Specified "self" for LPIPS model but no model passed'
            )
        return model
    elif lpips_model_spec == 'alexnet':
        if _cached_alexnet is None:
            alexnet_model = torchvision_models.alexnet(pretrained=True)
            _cached_alexnet = AlexNetFeatureModel(alexnet_model)
        lpips_model = _cached_alexnet
        if torch.cuda.is_available():
            lpips_model.cuda()
    elif lpips_model_spec == 'alexnet_cifar':
        if _cached_alexnet_cifar is None:
            alexnet_model = CifarAlexNet()
            _cached_alexnet_cifar = AlexNetFeatureModel(alexnet_model)
        lpips_model = _cached_alexnet_cifar
        if torch.cuda.is_available():
            lpips_model.cuda()
        try:
            state = torch.load('./alexnet_cifar.pt')
        except FileNotFoundError:
            state = load_state_dict_from_url(
                'https://perceptual-advex.s3.us-east-2.amazonaws.com/'
                'alexnet_cifar.pt',
                progress=True,
            )
        lpips_model.load_state_dict(state['model'])
    elif lpips_model_spec == 'ResNet18':
        lpips_model = ResNet18(num_classes = 100)
        lpips_model = torch.nn.DataParallel(lpips_model)
        lpips_model.load_state_dict(torch.load(load_state_dict))
        lpips_model.eval()

    elif lpips_model_spec == 'WideResNet34':
        lpips_model = WideResNet34(num_classes = 100)
        lpips_model = torch.nn.DataParallel(lpips_model)
        lpips_model.load_state_dict(torch.load(load_state_dict))
        lpips_model.eval()

    elif lpips_model_spec == 'PreActResNet18':
        lpips_model = PreActResNet18(num_classes = 100)
        lpips_model = torch.nn.DataParallel(lpips_model)
        lpips_model.load_state_dict(torch.load(load_state_dict))
        lpips_model.eval()

    elif isinstance(lpips_model_spec, str):
        raise ValueError(f'Invalid LPIPS model "{lpips_model_spec}"')
    else:
        lpips_model = lpips_model_spec

    lpips_model.eval()
    return lpips_model

class FastLagrangePerceptualAttack(nn.Module):
    def __init__(self, model, bound=0.5, step=None, num_iterations=20,
                 lam=10, h=1e-1, lpips_model='self', decay_step_size=True,
                 increase_lambda=True, projection='none', kappa=math.inf,
                 include_image_as_activation=False, randomize=False):
        """
        Perceptual attack using a Lagrangian relaxation of the
        LPIPS-constrainted optimization problem.

        bound is the (soft) bound on the LPIPS distance.
        step is the LPIPS step size.
        num_iterations is the number of steps to take.
        lam is the lambda value multiplied by the regularization term.
        h is the step size to use for finite-difference calculation.
        lpips_model is the model to use to calculate LPIPS or 'self' or
            'alexnet'
        """

        super().__init__()

        assert randomize is False

        self.model = model
        self.bound = bound
        if step is None:
            self.step = self.bound
        else:
            self.step = step
        self.num_iterations = num_iterations
        self.lam = lam
        self.h = h
        self.decay_step_size = decay_step_size
        self.increase_lambda = increase_lambda

        self.lpips_model = get_lpips_model(lpips_model, model)
        self.lpips_distance = LPIPSDistance(
            self.lpips_model,
            include_image_as_activation=include_image_as_activation,
        )
        self.projection = PROJECTIONS[projection](self.bound, self.lpips_model)
        self.loss = MarginLoss(kappa=kappa)

    def forward(self, inputs, labels):
        perturbations = torch.zeros_like(inputs)
        perturbations.normal_(0, 0.01)

        perturbations.requires_grad = True

        step_size = self.step
        lam = self.lam

        input_features = normalize_flatten_features(
            self.lpips_model.features(inputs)).detach()

        for attack_iter in range(self.num_iterations):
            # Decay step size, but increase lambda over time.
            if self.decay_step_size:
                step_size = \
                    self.step * 0.1 ** (attack_iter / self.num_iterations)
            if self.increase_lambda:
                lam = \
                    self.lam * 0.1 ** (1 - attack_iter / self.num_iterations)

            if perturbations.grad is not None:
                perturbations.grad.data.zero_()

            adv_inputs = inputs + perturbations

            if self.model == self.lpips_model:
                adv_features, adv_logits = \
                    self.model.features_logits(adv_inputs)
            else:
                adv_features = self.lpips_model.features(adv_inputs)
                adv_logits = self.model(adv_inputs)

            adv_loss = self.loss(adv_logits, labels)

            adv_features = normalize_flatten_features(adv_features)
            lpips_dists = (adv_features - input_features).norm(dim=1)

            loss = -adv_loss + lam * F.relu(lpips_dists - self.bound)
            loss.sum().backward()

            grad = perturbations.grad.data
            grad_normed = grad / \
                (grad.reshape(grad.size()[0], -1).norm(dim=1)
                 [:, None, None, None] + 1e-8)

            dist_grads = (
                adv_features -
                normalize_flatten_features(self.lpips_model.features(
                    inputs + perturbations - grad_normed * self.h))
            ).norm(dim=1) / 0.1

            perturbation_updates = -grad_normed * (
                step_size / (dist_grads + 1e-4)
            )[:, None, None, None]

            perturbations.data = (
                (inputs + perturbations + perturbation_updates).clamp(0, 1) -
                inputs
            ).detach()

        adv_inputs = (inputs + perturbations).detach()
        return self.projection(inputs, adv_inputs, input_features).detach()


class NoProjection(nn.Module):
    def __init__(self, bound, lpips_model):
        super().__init__()

    def forward(self, inputs, adv_inputs, input_features=None):
        return adv_inputs


class BisectionPerceptualProjection(nn.Module):
    def __init__(self, bound, lpips_model, num_steps=10):
        super().__init__()

        self.bound = bound
        self.lpips_model = lpips_model
        self.num_steps = num_steps

    def forward(self, inputs, adv_inputs, input_features=None):
        batch_size = inputs.shape[0]
        if input_features is None:
            input_features = normalize_flatten_features(
                self.lpips_model.features(inputs))

        lam_min = torch.zeros(batch_size, device=inputs.device)
        lam_max = torch.ones(batch_size, device=inputs.device)
        lam = 0.5 * torch.ones(batch_size, device=inputs.device)

        for _ in range(self.num_steps):
            projected_adv_inputs = (
                inputs * (1 - lam[:, None, None, None]) +
                adv_inputs * lam[:, None, None, None]
            )
            adv_features = self.lpips_model.features(projected_adv_inputs)
            adv_features = normalize_flatten_features(adv_features).detach()
            diff_features = adv_features - input_features
            norm_diff_features = torch.norm(diff_features, dim=1)

            lam_max[norm_diff_features > self.bound] = \
                lam[norm_diff_features > self.bound]
            lam_min[norm_diff_features <= self.bound] = \
                lam[norm_diff_features <= self.bound]
            lam = 0.5*(lam_min + lam_max)
        return projected_adv_inputs.detach()


class NewtonsPerceptualProjection(nn.Module):
    def __init__(self, bound, lpips_model, projection_overshoot=1e-1,
                 max_iterations=10):
        super().__init__()

        self.bound = bound
        self.lpips_model = lpips_model
        self.projection_overshoot = projection_overshoot
        self.max_iterations = max_iterations
        self.bisection_projection = BisectionPerceptualProjection(
            bound, lpips_model)

    def forward(self, inputs, adv_inputs, input_features=None):
        original_adv_inputs = adv_inputs
        if input_features is None:
            input_features = normalize_flatten_features(
                self.lpips_model.features(inputs))

        needs_projection = torch.ones_like(adv_inputs[:, 0, 0, 0]) \
            .bool()

        needs_projection.requires_grad = False
        iteration = 0
        while needs_projection.sum() > 0 and iteration < self.max_iterations:
            adv_inputs.requires_grad = True
            adv_features = normalize_flatten_features(
                self.lpips_model.features(adv_inputs[needs_projection]))
            adv_lpips = (input_features[needs_projection] -
                         adv_features).norm(dim=1)
            adv_lpips.sum().backward()

            projection_step_size = (adv_lpips - self.bound) \
                .clamp(min=0)
            projection_step_size[projection_step_size > 0] += \
                self.projection_overshoot

            grad_norm = adv_inputs.grad.data[needs_projection] \
                .view(needs_projection.sum(), -1).norm(dim=1)
            inverse_grad = adv_inputs.grad.data[needs_projection] / \
                grad_norm[:, None, None, None] ** 2

            adv_inputs.data[needs_projection] = (
                adv_inputs.data[needs_projection] -
                projection_step_size[:, None, None, None] *
                (1 + self.projection_overshoot) *
                inverse_grad
            ).clamp(0, 1).detach()

            needs_projection[needs_projection] = \
                projection_step_size > 0
            iteration += 1

        if needs_projection.sum() > 0:
            # If we still haven't projected all inputs after max_iterations,
            # just use the bisection method.
            adv_inputs = self.bisection_projection(
                inputs, original_adv_inputs, input_features)

        return adv_inputs.detach()


PROJECTIONS = {
    'none': NoProjection,
    'linesearch': BisectionPerceptualProjection,
    'bisection': BisectionPerceptualProjection,
    'gradient': NewtonsPerceptualProjection,
    'newtons': NewtonsPerceptualProjection,
}


class FirstOrderStepPerceptualAttack(nn.Module):
    def __init__(self, model, bound=0.5, num_iterations=5,
                 h=1e-3, kappa=1, lpips_model='self',
                 targeted=False, randomize=False,
                 include_image_as_activation=False):
        """
        Perceptual attack using conjugate gradient to solve the constrained
        optimization problem.

        bound is the (approximate) bound on the LPIPS distance.
        num_iterations is the number of CG iterations to take.
        h is the step size to use for finite-difference calculation.
        """

        super().__init__()

        assert randomize is False

        self.model = model
        self.bound = bound
        self.num_iterations = num_iterations
        self.h = h

        self.lpips_model = get_lpips_model(lpips_model, model)
        self.lpips_distance = LPIPSDistance(
            self.lpips_model,
            include_image_as_activation=include_image_as_activation,
        )
        self.loss = MarginLoss(kappa=kappa, targeted=targeted)

    def _multiply_matrix(self, v):
        """
        If (D phi) is the Jacobian of the features function for the model
        at inputs, then approximately calculates
            (D phi)T (D phi) v
        """

        self.inputs.grad.data.zero_()

        with torch.no_grad():
            v_features = self.lpips_model.features(self.inputs.detach() +
                                                   self.h * v)
            D_phi_v = (
                normalize_flatten_features(v_features) -
                self.input_features
            ) / self.h

        torch.sum(self.input_features * D_phi_v).backward(retain_graph=True)

        return self.inputs.grad.data.clone()

    def forward(self, inputs, labels):
        self.inputs = inputs

        inputs.requires_grad = True
        if self.model == self.lpips_model:
            input_features, orig_logits = self.model.features_logits(inputs)
        else:
            input_features = self.lpips_model.features(inputs)
            orig_logits = self.model(inputs)
        self.input_features = normalize_flatten_features(input_features)

        loss = self.loss(orig_logits, labels)
        loss.sum().backward(retain_graph=True)

        inputs_grad = inputs.grad.data.clone()
        if inputs_grad.abs().max() < 1e-4:
            return inputs

        # Variable names are from
        # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
        x = torch.zeros_like(inputs)
        r = inputs_grad - self._multiply_matrix(x)
        p = r

        for cg_iter in range(self.num_iterations):
            r_last = r
            p_last = p
            x_last = x
            del r, p, x

            r_T_r = (r_last ** 2).sum(dim=[1, 2, 3])
            if r_T_r.max() < 1e-1 and cg_iter > 0:
                # If the residual is small enough, just stop the algorithm.
                x = x_last
                break

            A_p_last = self._multiply_matrix(p_last)

            # print('|r|^2 =', ' '.join(f'{z:.2f}' for z in r_T_r))
            alpha = (
                r_T_r /
                (p_last * A_p_last).sum(dim=[1, 2, 3])
            )[:, None, None, None]
            x = x_last + alpha * p_last

            # These calculations aren't necessary on the last iteration.
            if cg_iter < self.num_iterations - 1:
                r = r_last - alpha * A_p_last

                beta = (
                    (r ** 2).sum(dim=[1, 2, 3]) /
                    r_T_r
                )[:, None, None, None]
                p = r + beta * p_last

        x_features = self.lpips_model.features(self.inputs.detach() +
                                               self.h * x)
        D_phi_x = (
            normalize_flatten_features(x_features) -
            self.input_features
        ) / self.h

        lam = (self.bound / D_phi_x.norm(dim=1))[:, None, None, None]

        inputs_grad_norm = inputs_grad.reshape(
            inputs_grad.size()[0], -1).norm(dim=1)
        # If the grad is basically 0, don't perturb that input. It's likely
        # already misclassified, and trying to perturb it further leads to
        # numerical instability.
        lam[inputs_grad_norm < 1e-4] = 0
        x[inputs_grad_norm < 1e-4] = 0

        # print('LPIPS', self.lpips_distance(
        #    inputs,
        #    inputs + lam * x,
        # ))

        return (inputs + lam * x).clamp(0, 1).detach()


class PerceptualPGDAttack(nn.Module):
    def __init__(self, model, bound=0.5, step=None, num_iterations=5,
                 cg_iterations=5, h=1e-3, lpips_model='self',
                 decay_step_size=False, kappa=1,
                 projection='newtons', randomize=False,
                 random_targets=False, num_classes=None,
                 include_image_as_activation=False):
        """
        Iterated version of the conjugate gradient attack.

        step_size is the step size in LPIPS distance.
        num_iterations is the number of steps to take.
        cg_iterations is the conjugate gradient iterations per step.
        h is the step size to use for finite-difference calculation.
        project is whether or not to project the perturbation into the LPIPS
            ball after each step.
        """

        super().__init__()

        assert randomize is False

        self.model = model
        self.bound = bound
        self.num_iterations = num_iterations
        self.decay_step_size = decay_step_size
        self.step = step
        self.random_targets = random_targets
        self.num_classes = num_classes

        if self.step is None:
            if self.decay_step_size:
                self.step = self.bound
            else:
                self.step = 2 * self.bound / self.num_iterations

        self.lpips_model = get_lpips_model(lpips_model, model)
        self.first_order_step = FirstOrderStepPerceptualAttack(
            model, bound=self.step, num_iterations=cg_iterations, h=h,
            kappa=kappa, lpips_model=self.lpips_model,
            include_image_as_activation=include_image_as_activation,
            targeted=self.random_targets)
        self.projection = PROJECTIONS[projection](self.bound, self.lpips_model)

    def _attack(self, inputs, labels):
        with torch.no_grad():
            input_features = normalize_flatten_features(
                self.lpips_model.features(inputs))

        start_perturbations = torch.zeros_like(inputs)
        start_perturbations.normal_(0, 0.01)
        adv_inputs = inputs + start_perturbations
        for attack_iter in range(self.num_iterations):
            if self.decay_step_size:
                step_size = self.step * \
                    0.1 ** (attack_iter / self.num_iterations)
                self.first_order_step.bound = step_size
            adv_inputs = self.first_order_step(adv_inputs, labels)
            adv_inputs = self.projection(inputs, adv_inputs, input_features)

        # print('LPIPS', self.first_order_step.lpips_distance(
        #    inputs,
        #    adv_inputs,
        # ))

        return adv_inputs

    def forward(self, inputs, labels):
        if self.random_targets:
            return utilities.run_attack_with_random_targets(
                self._attack,
                self.model,
                inputs,
                labels,
                self.num_classes,
            )
        else:
            return self._attack(inputs, labels)

class LagrangePerceptualAttack(nn.Module):
    def __init__(self, model, bound=0.5, step=None, num_iterations=20,
                 binary_steps=5, h=0.1, kappa=1, lpips_model='self',
                 projection='newtons', decay_step_size=True,
                 num_classes=None,
                 include_image_as_activation=False,
                 randomize=False, random_targets=False):
        """
        Perceptual attack using a Lagrangian relaxation of the
        LPIPS-constrainted optimization problem.
        bound is the (soft) bound on the LPIPS distance.
        step is the LPIPS step size.
        num_iterations is the number of steps to take.
        lam is the lambda value multiplied by the regularization term.
        h is the step size to use for finite-difference calculation.
        lpips_model is the model to use to calculate LPIPS or 'self' or
            'alexnet'
        """

        super().__init__()

        assert randomize is False

        self.model = model
        self.bound = bound
        self.decay_step_size = decay_step_size
        self.num_iterations = num_iterations
        if step is None:
            if self.decay_step_size:
                self.step = self.bound
            else:
                self.step = self.bound * 2 / self.num_iterations
        else:
            self.step = step
        self.binary_steps = binary_steps
        self.h = h
        self.random_targets = random_targets
        self.num_classes = num_classes

        self.lpips_model = get_lpips_model(lpips_model, model)
        self.lpips_distance = LPIPSDistance(
            self.lpips_model,
            include_image_as_activation=include_image_as_activation,
        )
        self.loss = MarginLoss(kappa=kappa, targeted=self.random_targets)
        self.projection = PROJECTIONS[projection](self.bound, self.lpips_model)

    def threat_model_contains(self, inputs, adv_inputs):
        """
        Returns a boolean tensor which indicates if each of the given
        adversarial examples given is within this attack's threat model for
        the given natural input.
        """

        return self.lpips_distance(inputs, adv_inputs) <= self.bound

    def _attack(self, inputs, labels):
        perturbations = torch.zeros_like(inputs)
        perturbations.normal_(0, 0.01)
        perturbations.requires_grad = True

        batch_size = inputs.shape[0]
        step_size = self.step

        lam = 0.01 * torch.ones(batch_size, device=inputs.device)

        input_features = normalize_flatten_features(
            self.lpips_model.features(inputs)).detach()

        live = torch.ones(batch_size, device=inputs.device, dtype=torch.bool)

        for binary_iter in range(self.binary_steps):
            for attack_iter in range(self.num_iterations):
                if self.decay_step_size:
                    step_size = self.step * \
                        (0.1 ** (attack_iter / self.num_iterations))
                else:
                    step_size = self.step

                if perturbations.grad is not None:
                    perturbations.grad.data.zero_()

                adv_inputs = (inputs + perturbations)[live]

                if self.model == self.lpips_model:
                    adv_features, adv_logits = \
                        self.model.features_logits(adv_inputs)
                else:
                    adv_features = self.lpips_model.features(adv_inputs)
                    adv_logits = self.model(adv_inputs)

                adv_labels = adv_logits.argmax(1)
                adv_loss = self.loss(adv_logits, labels[live])
                adv_features = normalize_flatten_features(adv_features)
                lpips_dists = (adv_features - input_features[live]).norm(dim=1)
                all_lpips_dists = torch.zeros(batch_size, device=inputs.device)
                all_lpips_dists[live] = lpips_dists

                loss = -adv_loss + lam[live] * F.relu(lpips_dists - self.bound)
                loss.sum().backward()

                grad = perturbations.grad.data[live]
                grad_normed = grad / \
                    (grad.reshape(grad.size()[0], -1).norm(dim=1)
                     [:, None, None, None] + 1e-8)

                dist_grads = (
                    adv_features -
                    normalize_flatten_features(self.lpips_model.features(
                        adv_inputs - grad_normed * self.h))
                ).norm(dim=1) / self.h

                updates = -grad_normed * (
                    step_size / (dist_grads + 1e-8)
                )[:, None, None, None]

                perturbations.data[live] = (
                    (inputs[live] + perturbations[live] +
                     updates).clamp(0, 1) -
                    inputs[live]
                ).detach()

                if self.random_targets:
                    live[live] = (adv_labels != labels[live]) | (lpips_dists > self.bound)
                else:
                    live[live] = (adv_labels == labels[live]) | (lpips_dists > self.bound)
                if live.sum() == 0:
                    break

            lam[all_lpips_dists >= self.bound] *= 10
            if live.sum() == 0:
                break

        adv_inputs = (inputs + perturbations).detach()
        adv_inputs = self.projection(inputs, adv_inputs, input_features)
        return adv_inputs

    def forward(self, inputs, labels):
        if self.random_targets:
            return utilities.run_attack_with_random_targets(
                self._attack,
                self.model,
                inputs,
                labels,
                self.num_classes,
            )
        else:
            return self._attack(inputs, labels)
