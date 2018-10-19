import random
from collections import namedtuple
import torch
Transition = namedtuple('Transition',
                        ('state','hidden','id', 'action', 'next_state','next_hidden','next_id','reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        if 0:
            self.memory = [torch.zeros([128, 128]).cuda(), (torch.zeros([1, 128, 64]).cuda(), torch.zeros([1, 128, 64]).cuda()), \
                       int(0), torch.zeros([128]).cuda(), torch.zeros([128, 128]).cuda(), (torch.zeros([1, 128, 64]).cuda(), \
                       torch.zeros([1, 128, 64]).cuda()), int(0), torch.zeros([128]).cuda()]

        self.memory = []
        self.position = 0
        self.filled = 0

    def push(self, *args):
        """Saves a transition."""
        current_bs = args[0].size()[0]
        if self.position == 0 and self.filled == 0:
            self.memory = [args[0], args[1], args[2], args[3], \
                           args[4], args[5], args[6], args[7]]
            self.position = (self.position + current_bs) % self.capacity
        else:
            if self.memory[0].size()[0] < self.capacity:
                self.memory[0] = torch.cat((self.memory[0], torch.zeros(args[0].size()).cuda()), dim=0)
                self.memory[1] = (torch.cat((self.memory[1][0], torch.zeros(args[1][0].size()).cuda()), dim=1), \
                                  torch.cat((self.memory[1][1], torch.zeros(args[1][1].size()).cuda()), dim=1))
                # self.memory[2] = torch.cat((self.memory[2], torch.zeros([128, 128]).cuda()), dim=0)
                self.memory[3] = torch.cat((self.memory[3], torch.zeros(args[3].size()).long().cuda()), dim=0)
                if args[4] is not None:
                    self.memory[4] = torch.cat((self.memory[4], torch.zeros(args[4].size()).cuda()), dim=0)
                    self.memory[5] = (torch.cat((self.memory[5][0], torch.zeros(args[5][0].size()).cuda()), dim=1), \
                                      torch.cat((self.memory[5][1], torch.zeros(args[5][1].size()).cuda()), dim=1))
                # self.memory[6] = torch.cat((self.memory[6], torch.zeros([128, 128]).cuda()), dim=0)
                self.memory[7] = torch.cat((self.memory[7], torch.zeros(args[7].size()).cuda()), dim=0)

            self.memory[0][self.position : self.position + current_bs] = args[0]
            self.memory[1][0][:, self.position : self.position + current_bs] = args[1][0]
            self.memory[1][1][:, self.position:self.position + current_bs] = args[1][1]
            self.memory[2] = args[2]
            self.memory[3][self.position : self.position + current_bs] = args[3]
            if args[4] is not None:
                self.memory[4][self.position : self.position + current_bs] = args[4]  # if args[4] is not None else None
            if args[5] is not None:
                self.memory[5][0][:, self.position : self.position + current_bs] = args[5][0]  # if args[4]!=None else None
                self.memory[5][1][:, self.position : self.position + current_bs] = args[5][1]  # if args[4]!=None else None
            self.memory[6] = args[6] if args[6] is not None else None
            self.memory[7][self.position : self.position + current_bs] = args[7]
            if self.position + current_bs < self.capacity:
                self.position = (self.position + current_bs) % self.capacity
            else:
                self.position = 0
                self.filled = 1

        '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        '''


    def sample(self):
        return random.sample(self.memory, 1)

    def __len__(self):
        return len(self.memory)