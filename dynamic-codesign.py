import argparse
from utee import misc, quant, selector
import torch
import torch.backends.cudnn as cudnn
from utee.misc import *
from utee.rnn_model import *
cudnn.benchmark =True
from collections import OrderedDict
from pruning.methods import weight_prune, weight_prune_second, normalized_params, weight_prune_approx_layer_wise, weight_prune_approx_global
from pruning.methods import weight_prune_approx_layer_wise_k_set, weight_approx_incrementally
from pruning.methods import weight_approx_incrementally_two_group_quantization, weight_approx_incrementally_two_group_random, weight_approx_incrementally_two_group_magnitude
from pruning.methods import weight_approx_incrementally_one_group_quantization, weight_approx_incrementally_one_group_random, weight_approx_incrementally_one_group_magnitude
from pruning.methods import weight_approx_incrementally_two_group_quantization_unpart, weight_approx_incrementally_two_group_magnitude_unpart
from pruning.utils import to_var, test, prune_rate
import torch.nn as nn
from parameters import *
from transitions import *
import torch.optim as optim

#python dynamic-codesign.py --type cifar10 --config 0 --group 0 --incremental 0
parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
parser.add_argument('--type', default='cifar10', help='|'.join(selector.known_models))
parser.add_argument('--quant_method', default='linear', help='linear|minmax|log|tanh')

parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 64)')
parser.add_argument('--second', type=int, default=0, help='if the second step is conducted')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_step_one', dest='resume_step_one', action='store_true', help='use first step model')


parser.add_argument('--incremental', default=0, type=int,
                    help='incremental approximation type rand|quant|mag')
parser.add_argument('--group', type=int, default=0, help='number of groups 0|1')
parser.add_argument('--iterations', type=int, default=8, help='number of incremental learning')
parser.add_argument('--config', type=int, default=0, help='threshold conservative 0 | modest 1 | aggressive 2')


parser.add_argument('--gpu', default='0', help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=110, help='random seed (default: 1)')
parser.add_argument('--model_root', default='~/.torch/models/', help='folder to save the model')
parser.add_argument('--data_root', default='/tmp/public_dataset/pytorch/', help='folder to save the model')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--curr_iter', type=int, default=0, help='input batch size for training (default: 64)')
parser.add_argument('--gamma', type=float, default=0.001, help='updating the probability')
parser.add_argument('--crate', type=float, default=1.6, help='threshold scope')

parser.add_argument('--k', type=int, default=10, help='approximation level')
parser.add_argument('--input_size', type=int, default=224, help='input size of image')
parser.add_argument('--n_sample', type=int, default=20, help='number of samples to infer the scaling factor')
parser.add_argument('--param_bits', type=int, default=8, help='bit-width for parameters')
parser.add_argument('--bn_bits', type=int, default=32, help='bit-width for running mean and std')
parser.add_argument('--fwd_bits', type=int, default=8, help='bit-width for layer output')
parser.add_argument('--overflow_rate', type=float, default=0.0, help='overflow rate')
args = parser.parse_args()








best_prec1 = 0
args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
args.ngpu = len(args.gpu)
misc.ensure_dir(args.logdir)
args.model_root = misc.expand_user(args.model_root)
args.data_root = misc.expand_user(args.data_root)
args.input_size = 299 if 'inception' in args.type else args.input_size
assert args.quant_method in ['linear', 'minmax', 'log', 'tanh']
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

assert torch.cuda.is_available(), 'no cuda'
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# load model and dataset fetcher
model_raw, ds_fetcher, is_imagenet = selector.select(args.type, model_root=args.model_root)


args.ngpu = args.ngpu if is_imagenet else 1

GAMMA = 0.999
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
#N_ACTIONS = 35
CHANNEL_CONF = [3, 128, 128, 256, 256, 512, 512]#, 1024]
rnn_ins = LSTMController(EMBEDDING_DIM, HIDDEN_DIM, CHANNEL_CONF ,N_ACTIONS, args.batch_size).cuda()
target_rnn = LSTMController(EMBEDDING_DIM, HIDDEN_DIM, CHANNEL_CONF ,N_ACTIONS, args.batch_size).cuda()
target_rnn.load_state_dict(rnn_ins.state_dict())
target_rnn.eval()

rnn_optimizer = optim.RMSprop(rnn_ins.parameters(), lr=0.001, eps=0.01)#, alpha=0.95, eps=0.01)


'''
# prune and approximate the weights
pre_masks = []
for p in model_raw.parameters():
    if len(p.data.size()) != 1:
        pre_masks.append(torch.ones(p.size()).float())

#masks, masks_amul= weight_prune_approx_global(model_raw, pre_masks, args.gamma, args.crate, args.k, cur_iter = args.curr_iter)
args.k = [1]
masks, masks_amul, masks_act, threshold = weight_prune_approx_layer_wise_k_set(model_raw, pre_masks, args.gamma, args.crate, args.k, cur_iter = args.curr_iter)
#args.k = 3
#masks, masks_amul, masks_act, threshold= weight_prune_approx_layer_wise(model_raw, pre_masks, args.gamma, args.crate, args.k, cur_iter = args.curr_iter)
model_raw.set_masks(masks, masks_amul, masks_act)
normalized_params(model_raw, masks)
prune_rate(model_raw)
'''

capacity=10000
memory = [ReplayMemory(capacity),ReplayMemory(capacity),ReplayMemory(capacity),ReplayMemory(capacity),ReplayMemory(capacity),ReplayMemory(capacity)]
# eval model
val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
for param in model_raw.features.parameters():
    param.requires_grad = False
'''
for epoch in range(args.start_epoch, args.epochs):
    import time
    time.sleep(3)
    print('>>>>epoch: '+str(epoch)+'\n')

    acc1, acc5 = misc.eval_model(model_raw, rnn_ins, target_rnn, rnn_optimizer, GAMMA, memory, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)

    res_str = "type={}, quant_method={}, param_bits={}, bn_bits={}, fwd_bits={}, overflow_rate={}, acc1={:.4f}, acc5={:.4f}".format(
        args.type, args.quant_method, args.param_bits, args.bn_bits, args.fwd_bits, args.overflow_rate, acc1, acc5)
    print(res_str)
'''
train_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=True, input_size=args.input_size)[0]
# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model_raw.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)




for epoch in range(args.start_epoch, args.epochs):
    #if args.distributed:
    #    train_sampler.set_epoch(epoch)
    #adjust_learning_rate(optimizer, epoch)
    # train for one epoch
    import time
    time.sleep(3)
    print('>>>>epoch: '+str(epoch)+'\n')
    train_mode(train_ds, model_raw, rnn_ins, target_rnn, rnn_optimizer, GAMMA, memory, criterion, optimizer, epoch, args)

    if 1:
        acc1, acc5 = misc.eval_model(model_raw, rnn_ins, target_rnn, rnn_optimizer, GAMMA, memory, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)

        res_str = "type={}, quant_method={}, param_bits={}, bn_bits={}, fwd_bits={}, overflow_rate={}, acc1={:.4f}, acc5={:.4f}".format(
            args.type, args.quant_method, args.param_bits, args.bn_bits, args.fwd_bits, args.overflow_rate, acc1, acc5)
        print(res_str)
    # evaluate on validation set

    '''
    prec1, prec5 = misc.eval_model(model_raw, rnn_ins, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)

    print(' * Prec@1 {top1:.3f} Prec@5 {top5:.3f}'
          .format(top1=prec1*100, top5=prec5*100))
    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.type,
        'state_dict': model_raw.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),

    }, is_best, filename = 'vanilla/' + args.type + '_' + str(args.epochs) + '_codesign_checkpoint.pth.tar')
    #error source?
    store_txt = 'vanilla/' + args.type + '_' + str(args.epochs) + '_codesign.txt'
    with open(store_txt, 'a') as f:
        f.write('{:.4f} {:.4f}'.format(prec1 * 100, prec5 * 100) + '\n')
    #normalized_params(model_raw, masks)
    #prune_rate(model_raw)
    '''





