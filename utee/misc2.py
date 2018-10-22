import cv2
import os
import shutil
import pickle as pkl
import time
import numpy as np
import hashlib
import torch
from IPython import embed
from parameters import *
from pruning.methods import weight_prune_approx_global, weight_prune_approx_layer_wise, weight_prune_approx_layer_wise_k_set
from transitions import *
from pruning.focalloss import *
import time

class Logger(object):
    def __init__(self):
        self._logger = None

    def init(self, logdir, name='log'):
        if self._logger is None:
            import logging
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log_file = os.path.join(logdir, name)
            if os.path.exists(log_file):
                os.remove(log_file)
            self._logger = logging.getLogger()
            self._logger.setLevel('INFO')
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            self._logger.addHandler(fh)
            self._logger.addHandler(ch)

    def info(self, str_info):
        self.init('/tmp', 'tmp.log')
        self._logger.info(str_info)
logger = Logger()

print = logger.info
def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)

def load_pickle(path):
    begin_st = time.time()
    with open(path, 'rb') as f:
        print("Loading pickle object from {}".format(path))
        v = pkl.load(f)
    print("=> Done ({:.4f} s)".format(time.time() - begin_st))
    return v

def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        print("Dumping pickle object to {}".format(path))
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

def auto_select_gpu(mem_bound=500, utility_bound=0, gpus=(0, 1, 2, 3, 4, 5, 6, 7), num_gpu=1, selected_gpus=None):
    import sys
    import os
    import subprocess
    import re
    import time
    import numpy as np
    if 'CUDA_VISIBLE_DEVCIES' in os.environ:
        sys.exit(0)
    if selected_gpus is None:
        mem_trace = []
        utility_trace = []
        for i in range(5): # sample 5 times
            info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
            mem = [int(s[:-5]) for s in re.compile('\d+MiB\s/').findall(info)]
            utility = [int(re.compile('\d+').findall(s)[0]) for s in re.compile('\d+%\s+Default').findall(info)]
            mem_trace.append(mem)
            utility_trace.append(utility)
            time.sleep(0.1)
        mem = np.mean(mem_trace, axis=0)
        utility = np.mean(utility_trace, axis=0)
        assert(len(mem) == len(utility))
        nGPU = len(utility)
        ideal_gpus = [i for i in range(nGPU) if mem[i] <= mem_bound and utility[i] <= utility_bound and i in gpus]

        if len(ideal_gpus) < num_gpu:
            print("No sufficient resource, available: {}, require {} gpu".format(ideal_gpus, num_gpu))
            sys.exit(0)
        else:
            selected_gpus = list(map(str, ideal_gpus[:num_gpu]))
    else:
        selected_gpus = selected_gpus.split(',')

    print("Setting GPU: {}".format(selected_gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected_gpus)
    return selected_gpus

def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))

def model_snapshot(model, new_file, old_file=None, verbose=False):
    from collections import OrderedDict
    import torch
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if old_file and os.path.exists(expand_user(old_file)):
        if verbose:
            print("Removing old model {}".format(expand_user(old_file)))
        os.remove(expand_user(old_file))
    if verbose:
        print("Saving model to {}".format(expand_user(new_file)))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if v.is_cuda:
            v = v.cpu()
        state_dict[k] = v
    torch.save(state_dict, expand_user(new_file))


def load_lmdb(lmdb_file, n_records=None):
    import lmdb
    import numpy as np
    lmdb_file = expand_user(lmdb_file)
    if os.path.exists(lmdb_file):
        data = []
        env = lmdb.open(lmdb_file, readonly=True, max_readers=512)
        with env.begin() as txn:
            cursor = txn.cursor()
            begin_st = time.time()
            print("Loading lmdb file {} into memory".format(lmdb_file))
            for key, value in cursor:
                _, target, _ = key.decode('ascii').split(':')
                target = int(target)
                img = cv2.imdecode(np.fromstring(value, np.uint8), cv2.IMREAD_COLOR)
                data.append((img, target))
                if n_records is not None and len(data) >= n_records:
                    break
        env.close()
        print("=> Done ({:.4f} s)".format(time.time() - begin_st))
        return data
    else:
        print("Not found lmdb file".format(lmdb_file))

def str2img(str_b):
    return cv2.imdecode(np.fromstring(str_b, np.uint8), cv2.IMREAD_COLOR)

def img2str(img):
    return cv2.imencode('.jpg', img)[1].tostring()

def md5(s):
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()


def optimize_model(memory, BATCH_SIZE, rnn_ins, target_rnn, GAMMA, optimizer):
    import random
    import torch.nn.functional as F


    stage = random.randint(0, 4)
    #if random.randint(0,1)==1:
    #    stage = 5
    transition = memory[stage].sample()
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = transition[0]

    #('state', 'hidden', 'id', 'action', 'next_state', 'next_hidden', 'next_id', 'reward')
    current_state = batch.state
    current_hidden = batch.hidden
    current_id = batch.id
    current_action = batch.action
    next_state = batch.next_state
    next_hidden = batch.next_hidden
    next_id = batch.next_id
    reward = batch.reward

    if stage != 5: #for normal state
        rnn_ins.set_hidden(current_hidden)
        state_action_values = rnn_ins(current_state, current_id).gather(1, current_action.unsqueeze(-1))

        target_rnn.set_hidden(next_hidden)
        next_state_values = target_rnn(next_state, next_id).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = ((next_state_values * GAMMA) + reward).detach()
    else: #for terminal state
        rnn_ins.set_hidden(current_hidden)
        state_action_values = rnn_ins(current_state, current_id).gather(1, current_action.unsqueeze(-1))
        expected_state_action_values = reward.detach()

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1),reduce = False)
    print(loss)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    #for param in rnn_ins.parameters():
        #param.grad.data.clamp_(-1, 1)
    optimizer.step()


def optimize_sequence_model(memory, rnn_ins, target_rnn, GAMMA, optimizer):
    import random
    import torch.nn.functional as F

    BATCH_SIZE = 128
    if memory[0].memory[0].size()[0]<10000:
        return

    if 0:
        idxes = random.sample(range(0, memory[0].memory[0].size()[0]), BATCH_SIZE)
    else:
        ratio = 0.5
        positive_sample_size = int(BATCH_SIZE*ratio)
        negtive_sample_size = BATCH_SIZE - positive_sample_size
        error_length = torch.sum(memory[5].memory[7] <= -1).item()

        sorted_index = memory[5].memory[7].sort()[1]
        positive_sample_idx = sorted_index[random.sample( range(0, error_length), positive_sample_size)]
        negtive_sample_idx = sorted_index[random.sample( range(error_length, len(sorted_index)), negtive_sample_size)]
        idxes=torch.cat((positive_sample_idx, negtive_sample_idx)).tolist()

    current_hidden = rnn_ins.init_hidden(BATCH_SIZE)
    seq_length = 6
    pred_values_scores = torch.zeros([seq_length, BATCH_SIZE, 1]).cuda()
    target_values_scores = torch.zeros([seq_length, BATCH_SIZE, 1]).cuda()

    for stage in range(6):
        #transition = memory[stage].memory[batch_idx]
        batch = memory[stage].memory

        #retrivel
        current_state = batch[0][idxes]

        #current_hidden = (batch[5][0][idxes], batch[5][1][idxes])
        #current_hidden = (batch[1][0][:, idxes, :], batch[1][0][:, idxes, :])

        current_id = batch[2]
        current_action = batch[3][idxes]
        next_state = batch[4][idxes] if batch[4] is not None else None
        next_hidden = (batch[5][0][:, idxes, :], batch[5][1][:, idxes, :]) if batch[5] is not None else None
        next_id = batch[6] if batch[6] is not None else None
        reward = batch[7][idxes]


        if stage != 5: #for normal state
            rnn_ins.set_hidden(current_hidden)
            state_action_values = rnn_ins(current_state, current_id).gather(1, current_action.unsqueeze(-1))
            current_hidden = rnn_ins.hidden
            #current_hidden = (batch[5][0][:, idxes, :], batch[5][0][:, idxes, :])

            target_rnn.set_hidden(current_hidden) #if we use previous results
            #next_state_values = target_rnn(next_state, next_id).max(1)[0].detach()
            max_id = rnn_ins(next_state, next_id).max(1)[1]
            rnn_ins.set_hidden(current_hidden)
            next_state_values = torch.gather(target_rnn(next_state, next_id), dim=1, index=max_id.unsqueeze(1)).detach()   #.max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = ((next_state_values.squeeze(1) * GAMMA) + reward).detach()
        else: #for terminal state
            rnn_ins.set_hidden(current_hidden)
            state_action_values = rnn_ins(current_state, current_id).gather(1, current_action.unsqueeze(-1))
            expected_state_action_values = reward.detach()



        pred_values_scores[stage] = state_action_values
        target_values_scores[stage] = expected_state_action_values.unsqueeze(1)

        '''
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduce=False)
        optimizer.zero_grad()
        loss.backward(torch.ones(state_action_values.size()).cuda(), retain_graph=True)
        for param in rnn_ins.parameters():
            # print(param.size())
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
        pass
        '''
    #loss_fn = FocalLoss()
    #loss = loss_fn()


    #loss_fn = torch.nn.MSELoss(reduce=False)
    #loss = loss_fn(pred_values_scores, target_values_scores)
    #rndidx = random.randint(0,5)
    seed_array = torch.ones(pred_values_scores.size()).cuda()*1/2
    rndidx = torch.bernoulli(seed_array)
    #loss = F.smooth_l1_loss(pred_values_scores, target_values_scores, reduce=False) #, reduce=False
    loss = F.mse_loss(pred_values_scores, target_values_scores, reduce=False)
    #loss = F.smooth_l1_loss(pred_values_scores, target_values_scores)
    # Optimize the model

    #mask = target_values_scores<-1
    #rndidx[mask] = 1
    optimizer.zero_grad()
    loss.backward(rndidx,retain_graph=True) #torch.ones(pred_values_scores.size()).cuda()
    #loss.backward(torch.ones(pred_values_scores.size()).cuda(), retain_graph=True)
    #loss.backward(retain_graph=True)

    #print(loss[5].abs().mean())
    #loss.backward( retain_graph=True)

    for param in rnn_ins.parameters():
        #print(param.size())
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)

    optimizer.step()
    pass






def eval_model(model, rnn_ins, target_rnn, rnn_optimizer, GAMMA, memory, ds, n_sample=None, ngpu=1, is_imagenet=False, is_evaluation=True):
    import tqdm
    tqdm.tqdm.monitor_interval = 0
    import torch
    from torch import nn
    from torch.autograd import Variable

    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        def forward(self, input):
            input.data.div_(255.)
            input.data[:, 0, :, :].sub_(self.mean[0]).div_(self.std[0])
            input.data[:, 1, :, :].sub_(self.mean[1]).div_(self.std[1])
            input.data[:, 2, :, :].sub_(self.mean[2]).div_(self.std[2])
            return self.model(input)

    correct1, correct5 = 0, 0
    accu_rewards = 0
    n_passed = 0
    if is_imagenet:
        model = ModelWrapper(model)
    model = model.eval()
    model = torch.nn.DataParallel(model, device_ids=range(ngpu)).cuda()

    n_sample = len(ds) if n_sample is None else n_sample
    #with torch.no_grad():
    for idx, (data, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
    #for idx, (data, target) in enumerate(ds):
        n_passed += len(data)
        data = Variable(torch.FloatTensor(data), volatile=True).cuda()
        indx_target = torch.LongTensor(target)
        with torch.no_grad():
            output, batch_rewards = model(data, rnn_ins, memory, target, is_evaluation)
        bs = output.size(0)
        idx_pred = output.data.sort(1, descending=True)[1]

        idx_gt1 = indx_target.expand(1, bs).transpose_(0, 1)
        idx_gt5 = idx_gt1.expand(bs, 5)

        correct1 += idx_pred[:, :1].cpu().eq(idx_gt1).sum().item()
        correct5 += idx_pred[:, :5].cpu().eq(idx_gt5).sum().item()
        accu_rewards += batch_rewards
        #optimize_model(memory, bs, rnn_ins, target_rnn, GAMMA, rnn_optimizer)
        if not is_evaluation:
            optimize_sequence_model(memory, rnn_ins, target_rnn, GAMMA, rnn_optimizer)
            if idx % 10 == 0:
                target_rnn.load_state_dict(rnn_ins.state_dict())

        #del data, target, output
        if idx >= n_sample - 1:
            break

        torch.cuda.empty_cache()
    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed
    average_rewards = accu_rewards / n_passed
    res_str = "----------------> acc1={:.4f}, acc5={:.4f}, Rewards={}".format(acc1, acc5, average_rewards)
    print(res_str)

    '''
    for idx in range(1000):
        optimize_sequence_model(memory, rnn_ins, target_rnn, GAMMA, rnn_optimizer)
        if idx % 10 == 0:
            target_rnn.load_state_dict(rnn_ins.state_dict())
        torch.cuda.empty_cache()
    '''

    return acc1, acc5

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train_mode(train_loader, model, rnn_ins, target_rnn, rnn_optimizer, GAMMA, memory, criterion, optimizer, epoch, args):
    global update_cc
    import tqdm
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    #model.train()
    #data = Variable(torch.FloatTensor(data), volatile=True).cuda()
    #indx_target = torch.LongTensor(target)

    end = time.time()
    accu_rewards = 0
    n_sample = len(train_loader) #if n_sample is None else n_sample
    #for idx, (data, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
    model = model.eval()
    for i, (input, target) in enumerate(train_loader):
        args.curr_iter += 1
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda() #non_blocking=True
        input=input.float()
        target = torch.LongTensor(target).cuda() #CUDA_LAUNCH_BLOCKING

        with torch.no_grad():
            output, batch_rewards = model(input, rnn_ins, memory, target)

        accu_rewards += batch_rewards
        #optimize_model(memory, bs, rnn_ins, target_rnn, GAMMA, rnn_optimizer)
        for w in range(10):
            optimize_sequence_model(memory, rnn_ins, target_rnn, GAMMA, rnn_optimizer)
            update_cc += 1
        if update_cc % 800 == 0:
            target_rnn.load_state_dict(rnn_ins.state_dict())

        # compute output
        #output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        '''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #torch.sum(masks_amul[7] == 5).item() / masks_amul[7].numel()
        '''
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    average_rewards = accu_rewards / 60000
    res_str = "----------------> acc1={top1.avg:.4f}, acc5={top5.avg:.4f}, Rewards={average_rewards}".format(top1=top1, top5=top5, average_rewards=average_rewards)
    print(res_str)


def load_state_dict(model, model_urls, model_root):
    from torch.utils import model_zoo
    from torch import nn
    import re
    from collections import OrderedDict
    own_state_old = model.state_dict()
    own_state = OrderedDict() # remove all 'group' string
    for k, v in own_state_old.items():
        k = re.sub('group\d+\.', '', k)
        own_state[k] = v

    state_dict = model_zoo.load_url(model_urls, model_root)

    for name, param in state_dict.items():
        if name not in own_state:
            print(own_state.keys())
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        raise KeyError('missing keys in state_dict: "{}"'.format(missing))



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')