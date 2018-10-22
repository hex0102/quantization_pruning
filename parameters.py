import torch
IL = 3
FL = 8

PER = 1
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 80000
steps_done = 0
N_ACTIONS = 5
update_cc = 0
mappings = torch.zeros(N_ACTIONS,4).cuda()
cc=0

for i in range(1,7,5):
    for j in range(i,7,5):
        for m in range(j,7,5):
            for n in range(m,7,5):
                mappings[cc,:] = torch.cuda.FloatTensor([i,j,m,n])
                #mappings[cc, :] = torch.cuda.FloatTensor([4, 4, 4, 4])
                #mappings.append(torch.cuda.FloatTensor([i,j,m,n]))#[i,j,m,n]
                cc=cc+1



pass