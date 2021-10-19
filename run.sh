#!/bin/sh


CUDA_VISIBLE_DEVICES=0 python train_OAAT.py > train_OAAT.txt


CUDA_VISIBLE_DEVICES=0 python eval.py --data CIFAR10 --arch WideResNet34 --main_model ./model-cifar-ResNet/OAAT_CIFAR10_1_0.45_1_1_3_0.0003_200.pkl > eval_OAAT.txt


