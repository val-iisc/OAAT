#!/bin/sh


CUDA_VISIBLE_DEVICES=0 python train_OAAT.py > train_OAAT.txt

CUDA_VISIBLE_DEVICES=0 python validation.py > val_OAAT.txt

CUDA_VISIBLE_DEVICES=0 python eval.py --main_model ./model-cifar-ResNet/OAAT_CIFAR10_1_0.2_0_1_3_0.0005_110.pkl > eval_OAAT.txt


