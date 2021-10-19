''' Default hyperparamters for CIFAR10_RN18, CIFAR100_PRN18, CIFAR10_WRN, CIFAR100_WRN and SVHN_PRN18 '''
import argparse
def use_default(default_arg):
    parser = argparse.ArgumentParser(description='PyTorch OAAT Adversarial Training')
    if default_arg == "CIFAR10_RN18":
        parser.add_argument('--arch', type=str, default='ResNet18', choices=['ResNet18', 'PreActResNet18','WideResNet34'])
        parser.add_argument('--epochs', type=int, default=110, metavar='N',
                    help='number of epochs to train')
        parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100','SVHN'])
        parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
        parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                    help='learning rate')
        parser.add_argument('--epsilon', default=16/255, type=float,
                    help='perturbation')
        parser.add_argument('--beta', default=1.5, type=float,
                    help='regularization, i.e., 1/lambda in TRADES inital value of beta')
        parser.add_argument('--model-dir', default='./model-cifar-ResNet',
                    help='directory of model for saving checkpoint')

        parser.add_argument('--beta_final', default=3, type=float,
                    help='the final value of beta at the end of training ')
        parser.add_argument('--mixup_alpha', default=0.2, type=float,
                    help='the value of mixup coeeficient in KL loss ')
        parser.add_argument('--mixup_epsilon', default=24/255, type=float,
                    help='the epsilon value used to generate mixup attack ')
        parser.add_argument('--lpips_weight', default=1, type=float,
                    help='the value of weight of lpips term in inner maximization')
        parser.add_argument('--use_CE', default=1, type=int,
                    help='uses CE loss for inner maximization when set to 1 else uses KL loss')
        parser.add_argument('--auto', default=0, type=float,
                    help='0 for no autoaugment 0.5 for autoaugment with probabilty 0.5 and 1 for autoaugment with probability 1')
        parser.add_argument('--tau_swa_list', type=float, nargs='*', default=[0.995,0.9996,0.9998],  help='The tau values for SWA')
        parser.add_argument('--label_smoothing', default=0, type=int,
                    help='put it as 1 it want to use label smoothing for the clean loss in outer minimization')
        parser.add_argument('--OAAT_warmup', default=0, type=int,
                    help='put it as 1 if want to use linear warmup of 10 epochs')
        parser.add_argument('--alternate_iter_eps', default=12/255, type=float,
                    help='the epsilon value after which alternate iters start ')



    elif default_arg == "CIFAR10_WRN":
        parser.add_argument('--arch', type=str, default='WideResNet34', choices=['ResNet18', 'PreActResNet18','WideResNet34'])
        parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
        parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100','SVHN'])
        parser.add_argument('--weight-decay', '--wd', default=3e-4,
                    type=float, metavar='W')
        parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                    help='learning rate')
        parser.add_argument('--epsilon', default=16/255, type=float,
                    help='perturbation')
        parser.add_argument('--beta', default=2, type=float,
                    help='regularization, i.e., 1/lambda in TRADES inital value of beta')
        parser.add_argument('--model-dir', default='./model-cifar-WideResNet',
                    help='directory of model for saving checkpoint')

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
        parser.add_argument('--alternate_iter_eps', default=12/255, type=float,
                    help='the epsilon value after which alternate iters start ')

    elif default_arg == "CIFAR100_WRN":
        parser.add_argument('--arch', type=str, default='WideResNet34', choices=['ResNet18', 'PreActResNet18','WideResNet34'])
        parser.add_argument('--epochs', type=int, default=110, metavar='N',
                    help='number of epochs to train')
        parser.add_argument('--data', type=str, default='CIFAR100', choices=['CIFAR10', 'CIFAR100','SVHN'])
        parser.add_argument('--weight-decay', '--wd', default=3e-4,
                    type=float, metavar='W')
        parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                    help='learning rate')
        parser.add_argument('--epsilon', default=16/255, type=float,
                    help='perturbation')
        parser.add_argument('--beta', default=2, type=float,
                    help='regularization, i.e., 1/lambda in TRADES inital value of beta')
        parser.add_argument('--model-dir', default='./model-cifar-WideResNet',
                    help='directory of model for saving checkpoint')

        parser.add_argument('--beta_final', default=4, type=float,
                    help='the final value of beta at the end of training ')
        parser.add_argument('--mixup_alpha', default=0.2, type=float,
                    help='the value of mixup coeeficient in KL loss ')
        parser.add_argument('--mixup_epsilon', default=24/255, type=float,
                    help='the epsilon value used to generate mixup attack ')
        parser.add_argument('--lpips_weight', default=2, type=float,
                    help='the value of weight of lpips term in inner maximization')
        parser.add_argument('--use_CE', default=1, type=int,
                    help='uses CE loss for inner maximization when set to 1 else uses KL loss')
        parser.add_argument('--auto', default=0.5, type=float,
                    help='0 for no autoaugment 0.5 for autoaugment with probabilty 0.5 and 1 for autoaugment with probability 1')
        parser.add_argument('--tau_swa_list', type=float, nargs='*', default=[0.995,0.9996,0.9998],  help='The tau values for SWA')
        parser.add_argument('--label_smoothing', default=1, type=int,
                    help='put it as 1 it want to use label smoothing for the clean loss in outer minimization')
        parser.add_argument('--OAAT_warmup', default=1, type=int,
                    help='put it as 1 if want to use linear warmup of 10 epochs')
        parser.add_argument('--alternate_iter_eps', default=12/255, type=float,
                    help='the epsilon value after which alternate iters start ')

    elif default_arg == "CIFAR100_PRN18":

        parser.add_argument('--arch', type=str, default='PreActResNet18', choices=['ResNet18', 'PreActResNet18','WideResNet34'])
        parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
        parser.add_argument('--data', type=str, default='CIFAR100', choices=['CIFAR10', 'CIFAR100','SVHN'])
        parser.add_argument('--weight-decay', '--wd', default=3e-4,
                    type=float, metavar='W')
        parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                    help='learning rate')
        parser.add_argument('--epsilon', default=16/255, type=float,
                    help='perturbation')
        parser.add_argument('--beta', default=2, type=float,
                    help='regularization, i.e., 1/lambda in TRADES inital value of beta')
        parser.add_argument('--model-dir', default='./model-cifar-PreactResNet',
                    help='directory of model for saving checkpoint')
        parser.add_argument('--beta_final', default=3, type=float,
                    help='the final value of beta at the end of training ')
        parser.add_argument('--mixup_alpha', default=0.25, type=float,
                    help='the value of mixup coeeficient in KL loss ')
        parser.add_argument('--mixup_epsilon', default=24/255, type=float,
                    help='the epsilon value used to generate mixup attack ')
        parser.add_argument('--lpips_weight', default=3, type=float,
                    help='the value of weight of lpips term in inner maximization')
        parser.add_argument('--use_CE', default=1, type=int,
                    help='uses CE loss for inner maximization when set to 1 else uses KL loss')
        parser.add_argument('--auto', default=0.5, type=float,
                    help='0 for no autoaugment 0.5 for autoaugment with probabilty 0.5 and 1 for autoaugment with probability 1')
        parser.add_argument('--tau_swa_list', type=float, nargs='*', default=[0.995,0.9996,0.9998],  help='The tau values for SWA')
        parser.add_argument('--label_smoothing', default=1, type=int,
                    help='put it as 1 it want to use label smoothing for the clean loss in outer minimization')
        parser.add_argument('--OAAT_warmup', default=1, type=int,
                    help='put it as 1 if want to use linear warmup of 10 epochs')
        parser.add_argument('--alternate_iter_eps', default=12/255, type=float,
                    help='the epsilon value after which alternate iters start ')

    elif default_arg == 'SVHN_PRN18':
        parser.add_argument('--arch', type=str, default='PreActResNet18', choices=['ResNet18', 'PreActResNet18','WideResNet34'])
        parser.add_argument('--epochs', type=int, default=110, metavar='N',
                    help='number of epochs to train')
        parser.add_argument('--data', type=str, default='SVHN', choices=['CIFAR10', 'CIFAR100','SVHN'])
        parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
        parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate')
        parser.add_argument('--epsilon', default=12/255, type=float,
                    help='perturbation')
        parser.add_argument('--beta', default=4, type=float,
                    help='regularization, i.e., 1/lambda in TRADES inital value of beta')
        parser.add_argument('--model-dir', default='./model-svhn-PreactResNet',
                    help='directory of model for saving checkpoint')

        parser.add_argument('--beta_final', default=4, type=float,
                    help='the final value of beta at the end of training ')
        parser.add_argument('--mixup_alpha', default=0.25, type=float,
                    help='the value of mixup coeeficient in KL loss ')
        parser.add_argument('--mixup_epsilon', default=12/255, type=float,
                    help='the epsilon value used to generate mixup attack ')
        parser.add_argument('--lpips_weight', default=3, type=float,
                    help='the value of weight of lpips term in inner maximization')
        parser.add_argument('--use_CE', default=0, type=int,
                    help='uses CE loss for inner maximization when set to 1 else uses KL loss')
        parser.add_argument('--auto', default=1, type=float,
                    help='0 for no autoaugment 0.5 for autoaugment with probabilty 0.5 and 1 for autoaugment with probability 1')
        parser.add_argument('--tau_swa_list', type=float, nargs='*', default=[0.995,0.9996,0.9998],  help='The tau values for SWA')
        parser.add_argument('--label_smoothing', default=0, type=int,
                    help='put it as 1 it want to use label smoothing for the clean loss in outer minimization')
        parser.add_argument('--OAAT_warmup', default=0, type=int,
                    help='put it as 1 if want to use linear warmup of 10 epochs')
        parser.add_argument('--alternate_iter_eps', default=9/255, type=float,
                    help='the epsilon value after which alternate iters start ')
    else:
        print("Use_default not Found")
        exit

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',help='input batch size for testing (default: 128)')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',help='resume training from which epoch')
    parser.add_argument('--data-path', type=str, default='../data',help='where is the dataset')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf'],help='The threat model')
    parser.add_argument('--resume-model', default='', type=str,help='path of model to resume training')
    parser.add_argument('--resume-optim', default='', type=str,help='path of optimizer to resume training')
    parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',help='save frequency')
    parser.add_argument('--awp-gamma', default=0.005, type=float,help='whether or not to add parametric noise')
    parser.add_argument('--awp-warmup', default=10, type=int,help='We could apply AWP after some epochs for accelerating.')
    parser.add_argument('--swa_save_epoch', default=1, type=int,help='Start saving SWA models after this epoch')
    parser.add_argument('--lr_schedule', default='cosine',choices=['cosine', 'step'],help='schedule used for training')
    parser.add_argument('--exp_name', default='OAAT',help='name of the method used for training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--use_defaults', type=str, default='CIFAR10_RN18' ,choices=['NONE','CIFAR10_RN18', 'CIFAR10_WRN','CIFAR100_WRN', 'CIFAR100_PRN18','SVHN_PRN18'],help='Use None is want to use the hyperparamters passed in the python training command else use the desired set of default hyperparameters')
    parser.add_argument('--wandb-run', default="OAAT")
    parser.add_argument('--wandb-notes', default="OAAT")
    parser.add_argument('--wandb-project', default="OAAT")
    parser.add_argument('--wandb-dir', default="./wandb_log")

    args = parser.parse_args()
    return args
