import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pruning.utils import prune_rate, arg_nonzero_min
from parameters import *
from approximate import *

def weight_prune(model, pruning_perc, k):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''    
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    # generate mask
    masks = []
    masks_amul = []
    masks_act = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = p.data.abs() > threshold
            masks.append(pruned_inds.float())
            masks_amul.append(torch.ones(p.data.size()).cuda()*k)
            masks_act.append(1*k)
    return masks, masks_amul, masks_act

def weight_prune_approx_global(model, pre_masks, gamma, crate, k, cur_iter = 0):
    '''
    Prune and approximate:  globallly larger-magnitude weights are approximated, smaller-magitude weights are prunes
    current iteration:
    return: masks # 1 is approximated; 0 is pruned
    '''
    all_weights = torch.ones(1).cuda()
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights = torch.cat((all_weights,p.clone().data.reshape(-1).abs()))
            #all_weights += list(p.cpu().data.abs().numpy().flatten())
    #all_weights = np.array(all_weights)
    mu_vec = torch.mean(all_weights)
    std_vec = torch.std(all_weights)
    #threshold = np.percentile(np.array(all_weights), pruning_perc)

    idx = 0
    masks_amul = []
    masks = []
    power = -1
    probThreshold = (1 + gamma * cur_iter) ** power
    threshold_vec = mu_vec + crate * std_vec
    print("threshold {:.4f}".format(threshold_vec.item()))
    for p in model.parameters():
        if len(p.data.size()) != 1:
            random_number = torch.rand(p.size())
            random_number = (random_number < probThreshold).cuda().float()

            alpha_vec = 0.9 * threshold_vec
            beta_vec = 1.1 * threshold_vec
            abs_kernel = p.abs()
            new_T = pre_masks[idx].cuda() - ((abs_kernel < alpha_vec).float())*random_number
            new_T = new_T + (abs_kernel > beta_vec).float()*random_number
            new_T = torch.clamp(new_T, min=0, max=1)
            #print()
            print("last and current diff {:.4f}".format(torch.sum(pre_masks[idx].cuda() == new_T).item()/new_T.numel()))
            masks.append(new_T)
            masks_amul.append(torch.ones(p.size()).cuda()*k)
            idx += 1
    return masks, masks_amul


def weight_prune_approx_layer_wise(model, pre_masks, gamma, crate, k, cur_iter = 0, threshold = None):
    '''
    Prune and approximate:  larger-magnitude weights are approximated, smaller-magitude weights are prunes
    current iteration:
    return: masks # 1 is approximated; 0 is pruned
    '''
    idx = 0
    masks_amul = []
    masks = []
    power = -1
    probThreshold = (1 + gamma * cur_iter) ** power
    all_weights = []
    masks_act = []
    if cur_iter == 0:
        threshold = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            random_number = torch.rand(p.size())
            random_number = (random_number < probThreshold).cuda().float()
            if cur_iter == 0:
                mu_vec = torch.mean(p.abs())
                std_vec = torch.std(p.abs())
                threshold_vec = mu_vec + crate[idx] * std_vec
                if crate[idx] == 0:
                    threshold_vec = 0
                threshold.append(threshold_vec)
            #print("threshold {:.4f}".format(threshold_vec.item()))
            alpha_vec = 0.9 * threshold[idx]
            beta_vec = 1.1 * threshold[idx]
            abs_kernel = p.abs()
            new_T = pre_masks[idx].cuda() - ((abs_kernel < alpha_vec).float())*random_number
            new_T = new_T + (abs_kernel >= beta_vec).float()*random_number
            new_T = torch.clamp(new_T, min=0, max=1)
            #np.percentile(np.array(abs_kernel.cpu().data.numpy().flatten()), 80)
            #print("last and current equal {:.4f}".format(torch.sum(pre_masks[idx].cuda() == new_T).item() / new_T.numel()))
            #all_weights = list(p.cpu().data.abs().numpy().flatten())
            #threshold = np.percentile(np.array(all_weights), 80.0)
            #pruned_inds = p.data.abs() > threshold
            #masks.append(pruned_inds.float())
            masks_act.append(1 * 3)
            masks.append(new_T)
            masks_amul.append(torch.ones(p.size()).cuda()*k)
            idx += 1
    return masks, masks_amul, masks_act, threshold


def weight_prune_approx_layer_wise_k_set(model, pre_masks, gamma, crate, k_set, cur_iter = 0, threshold = None):
    '''
    Prune and approximate:  larger-magnitude weights are approximated, smaller-magitude weights are prunes
    current iteration:
    return: masks # 1 is approximated; 0 is pruned
    '''
    idx = 0
    masks_amul = []
    masks = []
    power = -1
    probThreshold = (1 + gamma * cur_iter) ** power
    all_weights = []
    masks_act = []
    if cur_iter == 0:
        threshold = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            random_number = torch.rand(p.size())
            random_number = (random_number < probThreshold).cuda().float()
            if cur_iter == 0:
                mu_vec = torch.mean(p.abs())
                std_vec = torch.std(p.abs())
                threshold_vec = mu_vec + crate[idx] * std_vec
                if crate[idx] == 0:
                    threshold_vec = 0
                threshold.append(threshold_vec)
            #print("threshold {:.4f}".format(threshold_vec.item()))
            alpha_vec = 0.9 * threshold[idx]
            beta_vec = 1.1 * threshold[idx]
            abs_kernel = p.abs()
            new_T = pre_masks[idx].cuda() - ((abs_kernel < alpha_vec).float())*random_number
            new_T = new_T + (abs_kernel >= beta_vec).float()*random_number
            new_T = torch.clamp(new_T, min=0, max=1)


            #masks[mask_amul != (IL + FL)] = 1.0

            masks_act.append(1 * 5)
            masks.append(new_T)

            mask_amul = torch.ones(p.data.size()).cuda() * 5
            #
            #for k in k_set:  # k_set=[3,2,1]
            #    approximated_p = amul(p.data, torch.ones(p.data.size()).cuda() * k, IL, FL)
            #    deviations = torch.abs((approximated_p - p.data) / p.data)
            #    tolerable_mask_in_pruning_zone = (1 - new_T).byte() * (deviations < 0.25)
            #    # allowed_masks.append(tolerable_mask_in_pruning_zone)
            #    mask_amul[tolerable_mask_in_pruning_zone] = k

            masks_amul.append(mask_amul)

            idx += 1
    return masks, masks_amul, masks_act, threshold


def weight_prune_second(model, pre_masks, pruning_perc, k):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''
    idx = 0
    all_weights = []
    processed_datas = [] #orignial masked data set to 0
    for p in model.parameters():
        if len(p.data.size()) != 1:
            temp_data = p.clone().cpu().data.abs()
            temp_data[1-pre_masks[idx].byte()]=0
            processed_datas.append(temp_data)
            all_weights += list(temp_data.numpy().flatten())
            idx += 1

    threshold = np.percentile(np.array(all_weights), pruning_perc)

    # generate mask
    masks = []
    masks_amul = []
    masks_act = []
    idx = 0
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = processed_datas[idx] > threshold
            masks.append(pruned_inds.float())
            masks_amul.append(torch.ones(p.data.size()).cuda()*k)
            masks_act.append(1*k)
            idx += 1
    return masks, masks_amul, masks_act

def weight_approx_incrementally_two_group_quantization(model, pre_masks, masks_amul, curr_iter, iterations):  #quantization-based ranking
    idx = 0
    for p in model.parameters():
        if len(p.data.size()) != 1:
            if idx != 0 and idx != 7:
                approximated_p = amul(p.data, torch.ones(p.data.size()).cuda()*1 ,IL, FL)
                deviations = torch.abs((approximated_p - p.data))
                temp_data = deviations.cpu().data
                temp_data[pre_masks[idx] == 0] = 0
                temp_data[masks_amul[idx] != 3] = 0
                #temp_data[masks_amul[idx]!=3]
                left_perc_total = torch.sum(pre_masks[idx].byte()).item()/pre_masks[idx].numel()
                amul_l_perc0 = (1 - left_perc_total) + 0 + (left_perc_total * 8 / 10) / iterations * (curr_iter)
                amul_l_perc1 = (1-left_perc_total) + 0 + (left_perc_total * 8/10)/iterations * (curr_iter + 1)
                amul_h_perc0 = (1 - left_perc_total) + left_perc_total * 8 / 10 + (left_perc_total * 2 / 10) / iterations * (curr_iter)
                amul_h_perc1 = (1 - left_perc_total) + left_perc_total*8/10 + (left_perc_total * 2/10)/iterations * (curr_iter+1)
                threshold0 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_l_perc0 * 100)
                threshold1 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_l_perc1*100)
                threshold2 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_h_perc0*100)
                threshold3 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_h_perc1*100)
                masks_amul[idx][(temp_data < threshold1) * (temp_data > threshold0 ) ] = 1
                masks_amul[idx][(temp_data < threshold3) * (temp_data > threshold2)] = 2
            #approximate_perc = (1-left_perc) + (left_perc/iterations)*(curr_iter+1)
            idx += 1
    return  masks_amul

def weight_approx_incrementally_two_group_quantization_unpart(model, pre_masks, masks_amul, curr_iter, iterations):  #quantization-based ranking
    idx = 0
    for p in model.parameters():
        if len(p.data.size()) != 1:
            if idx != 0 and idx != 7:
                approximated_p = amul(p.data, torch.ones(p.data.size()).cuda()*1 ,IL, FL)
                deviations = torch.abs((approximated_p - p.data))
                temp_data = deviations.cpu().data
                temp_data[pre_masks[idx] == 0] = 0
                temp_data[masks_amul[idx] != 3] = 0
                #temp_data[masks_amul[idx]!=3]
                left_perc_total = torch.sum(pre_masks[idx].byte()).item()/pre_masks[idx].numel()
                amul_l_perc0 = (1 - left_perc_total) + 0 + (left_perc_total) / iterations * (curr_iter)
                amul_l_perc1 = (1 - left_perc_total) + 0 + (left_perc_total) / iterations * (curr_iter) + (left_perc_total) / iterations * 8/10
                amul_h_perc0 = (1 - left_perc_total) + 0 + (left_perc_total) / iterations * (curr_iter) + (left_perc_total) / iterations * 8/10
                amul_h_perc1 = (1-left_perc_total) + 0 + (left_perc_total)/iterations * (curr_iter + 1)

                threshold0 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_l_perc0 * 100)
                threshold1 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_l_perc1*100)
                threshold2 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_h_perc0*100)
                threshold3 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_h_perc1*100)
                masks_amul[idx][(temp_data < threshold1) * (temp_data >= threshold0 ) ] = 1
                masks_amul[idx][(temp_data < threshold3) * (temp_data >= threshold2)] = 2
            #approximate_perc = (1-left_perc) + (left_perc/iterations)*(curr_iter+1)
            idx += 1
    return  masks_amul


def weight_approx_incrementally_one_group_quantization(model, pre_masks, masks_amul, curr_iter, iterations):  #quantization-based ranking
    idx = 0
    for p in model.parameters():
        if len(p.data.size()) != 1:
            if idx != 0 and idx != 7:
                approximated_p = amul(p.data, torch.ones(p.data.size()).cuda()*1 ,IL, FL)
                deviations = torch.abs((approximated_p - p.data))
                temp_data = deviations.cpu().data
                temp_data[pre_masks[idx] == 0] = 0
                temp_data[masks_amul[idx] != 3] = 0
                left_perc_total = torch.sum(pre_masks[idx].byte()).item()/pre_masks[idx].numel()
                amul_l_perc0 = (1 - left_perc_total) + 0 + (left_perc_total * 1) / iterations * (curr_iter)
                amul_l_perc1 = (1-left_perc_total) + 0 + (left_perc_total * 1)/iterations * (curr_iter + 1)
                threshold0 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_l_perc0 * 100)
                threshold1 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_l_perc1*100)
                masks_amul[idx][(temp_data < threshold1) * (temp_data > threshold0 ) ] = 1
            idx += 1
    return  masks_amul


def weight_approx_incrementally_two_group_magnitude(model, pre_masks, masks_amul, curr_iter, iterations): #magnitudes-based ranking
    idx = 0
    for p in model.parameters():
        if len(p.data.size()) != 1:
            if idx != 0 and idx != 7:
                temp_data = p.clone().cpu().data.abs()
                temp_data[pre_masks[idx] == 0] = 0
                temp_data[masks_amul[idx] != 3] = 0
                #temp_data[masks_amul[idx]!=3]
                left_perc_total = torch.sum(pre_masks[idx].byte()).item()/pre_masks[idx].numel()
                amul_l_perc0 = (1 - left_perc_total) + 0 + (left_perc_total * 8 / 10) / iterations * (curr_iter)
                amul_l_perc1 = (1-left_perc_total) + 0 + (left_perc_total * 8/10)/iterations * (curr_iter + 1)
                amul_h_perc0 = (1 - left_perc_total) + left_perc_total * 8 / 10 + (left_perc_total * 2 / 10) / iterations * (curr_iter)
                amul_h_perc1 = (1 - left_perc_total) + left_perc_total*8/10 + (left_perc_total * 2/10)/iterations * (curr_iter+1)
                threshold0 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_l_perc0 * 100)
                threshold1 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_l_perc1*100)
                threshold2 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_h_perc0*100)
                threshold3 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_h_perc1*100)
                masks_amul[idx][(temp_data < threshold1) * (temp_data > threshold0 ) ] = 1
                masks_amul[idx][(temp_data < threshold3) * (temp_data > threshold2)] = 2
            #approximate_perc = (1-left_perc) + (left_perc/iterations)*(curr_iter+1)
            idx += 1
    return  masks_amul


def weight_approx_incrementally_two_group_magnitude_unpart(model, pre_masks, masks_amul, curr_iter, iterations): #magnitudes-based ranking
    idx = 0
    for p in model.parameters():
        if len(p.data.size()) != 1:
            if idx != 0 and idx != 7:
                temp_data = p.clone().cpu().data.abs()
                temp_data[pre_masks[idx] == 0] = 0
                temp_data[masks_amul[idx] != 3] = 0
                #temp_data[masks_amul[idx]!=3]
                left_perc_total = torch.sum(pre_masks[idx].byte()).item()/pre_masks[idx].numel()
                amul_l_perc0 = (1 - left_perc_total) + 0 + (left_perc_total) / iterations * (curr_iter)
                amul_l_perc1 = (1 - left_perc_total) + 0 + (left_perc_total) / iterations * (curr_iter) + (left_perc_total) / iterations * 8/10
                amul_h_perc0 = (1 - left_perc_total) + 0 + (left_perc_total) / iterations * (curr_iter) + (left_perc_total) / iterations * 8/10
                amul_h_perc1 = (1-left_perc_total) + 0 + (left_perc_total)/iterations * (curr_iter + 1)
                #amul_h_perc0 = (1 - left_perc_total) + left_perc_total * 9 / 10 + (left_perc_total * 1 / 10) / iterations * (curr_iter)
                #amul_h_perc1 = (1 - left_perc_total) + left_perc_total*9/10 + (left_perc_total * 1/10)/iterations * (curr_iter+1)
                threshold0 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_l_perc0 * 100)
                threshold1 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_l_perc1*100)
                threshold2 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_h_perc0*100)
                threshold3 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_h_perc1*100)
                masks_amul[idx][(temp_data < threshold1) * (temp_data >= threshold0 ) ] = 1
                masks_amul[idx][(temp_data < threshold3) * (temp_data >= threshold2)] = 2
            #approximate_perc = (1-left_perc) + (left_perc/iterations)*(curr_iter+1)
            idx += 1
    return  masks_amul

def weight_approx_incrementally_one_group_magnitude(model, pre_masks, masks_amul, curr_iter, iterations): #magnitudes-based ranking
    idx = 0
    for p in model.parameters():
        if len(p.data.size()) != 1:
            if idx != 0 and idx != 7:
                temp_data = p.clone().cpu().data.abs()
                temp_data[pre_masks[idx] == 0] = 0
                temp_data[masks_amul[idx] != 3] = 0
                #temp_data[masks_amul[idx]!=3]
                left_perc_total = torch.sum(pre_masks[idx].byte()).item()/pre_masks[idx].numel()
                amul_l_perc0 = (1 - left_perc_total) + 0 + (left_perc_total)/iterations * (curr_iter)
                amul_l_perc1 = (1-left_perc_total) + 0 + (left_perc_total)/iterations * (curr_iter + 1)
                threshold0 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_l_perc0 * 100)
                threshold1 = np.percentile(np.array(list(temp_data.numpy().flatten())), amul_l_perc1 * 100)
                masks_amul[idx][(temp_data < threshold1) * (temp_data > threshold0 ) ] = 1
            idx += 1
    return  masks_amul


def weight_approx_incrementally_two_group_random(model, pre_masks, masks_amul, curr_iter, iterations): #magnitudes-based ranking
    idx = 0
    for p in model.parameters():
        if len(p.data.size()) != 1:
            if idx != 0 and idx != 7:
                temp_data = p.clone().cpu().data.abs()
                temp_data[pre_masks[idx] == 0] = 0
                temp_data[masks_amul[idx] != 3] = 0
                equal_distribution = torch.randint(low=1, high=iterations-curr_iter+1, size=temp_data.size())
                equal_distribution = equal_distribution.int() * (temp_data != 0).int()
                group_distribution = torch.randint(low=0, high=10, size=temp_data.size())
                group2 = (group_distribution == 0) * (equal_distribution == 1)
                group1 = (group_distribution != 0) * (equal_distribution == 1)
                masks_amul[idx][group1] = 1
                masks_amul[idx][group2] = 2
            idx += 1
    return  masks_amul

def weight_approx_incrementally_one_group_random(model, pre_masks, masks_amul, curr_iter, iterations): #magnitudes-based ranking
    idx = 0
    for p in model.parameters():
        if len(p.data.size()) != 1:
            if idx != 0 and idx != 7:
                temp_data = p.clone().cpu().data.abs()
                temp_data[pre_masks[idx] == 0] = 0
                temp_data[masks_amul[idx] != 3] = 0
                equal_distribution = torch.randint(low=1, high=iterations-curr_iter+1, size=temp_data.size())
                equal_distribution = equal_distribution.int() * (temp_data != 0).int()
                group_distribution = torch.randint(low=0, high=10, size=temp_data.size())
                group2 = (group_distribution == 0) * (equal_distribution == 1)
                group1 = (group_distribution != 0) * (equal_distribution == 1)
                masks_amul[idx][group1] = 1
                masks_amul[idx][group2] = 1
            idx += 1
    return  masks_amul




def weight_approx_incrementally(model, pre_masks, masks_amul, curr_iter, iterations):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''
    idx = 0 #notice the first layer. which is not pruned but will be approximated?
    for p in model.parameters():
        if len(p.data.size()) != 1:
            temp_data = p.clone().cpu().data.abs()
            temp_data[1-pre_masks[idx].byte()]= 0
            left_perc = torch.sum(pre_masks[idx].byte()).item()/pre_masks[idx].numel()
            approximate_perc = (1-left_perc) + (left_perc/iterations)*(curr_iter+1)
            threshold = np.percentile(np.array(list(temp_data.numpy().flatten())), approximate_perc*100)
            masks_amul[idx][temp_data < threshold] = 1
            idx += 1
    return masks_amul




def normalized_params(model, masks):
    idx = 0
    for p in model.parameters():
        if len(p.data.size()) != 1:
            p.data = p.data * masks[idx].cuda()
            idx += 1

def layer_wise_weight_prune(model, pruning_perc, iterations, k_set):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights = list(p.cpu().data.abs().numpy().flatten())
        current_pruning_perc = pruning_perc / iterations



    all_weights = []
    if len(p.data.size()) != 1:
        all_weights = list(p.cpu().data.abs().numpy().flatten())


    current_pruning_perc = pruning_perc/iterations
    actual_pruning_perc = 0
    for i in range(iterations):
        threshold = np.percentile(np.array(all_weights), current_pruning_perc)
        pruned_inds = p.data.abs() > threshold
        masks = pruned_inds.float()
        #allowed_masks = []
        mask_amul = torch.ones(p.data.size()).cuda() * (IL + FL)
        for k in k_set: # k_set=[3,2,1]
            approximated_p = amul(p.data, torch.ones(p.data.size()).cuda()*k ,IL, FL)
            deviations = torch.abs((approximated_p - p.data)/p.data)
            tolerable_mask_in_pruning_zone = (1 - pruned_inds) * (deviations < 0.25)
            #allowed_masks.append(tolerable_mask_in_pruning_zone)
            mask_amul[tolerable_mask_in_pruning_zone] = k
        masks[mask_amul != (IL+FL)] = 1.0

        actual_pruning_perc = actual_pruning_perc + 1 - (torch.sum(p.data)/p.data.numel())[0]
        current_pruning_perc = (pruning_perc - actual_pruning_perc) / (iterations - 1 - i) + current_pruning_perc
        # retrain

    return masks, mask_amul



def weight_prune_advanced(p, pruning_perc, iterations, k_set):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''
    all_weights = []
    if len(p.data.size()) != 1:
        all_weights = list(p.cpu().data.abs().numpy().flatten())


    current_pruning_perc = pruning_perc/iterations
    actual_pruning_perc = 0
    for i in range(iterations):
        threshold = np.percentile(np.array(all_weights), current_pruning_perc)
        pruned_inds = p.data.abs() > threshold
        masks = pruned_inds.float()
        #allowed_masks = []
        mask_amul = torch.ones(p.data.size()).cuda() * (IL + FL)
        for k in k_set: # k_set=[3,2,1]
            approximated_p = amul(p.data, torch.ones(p.data.size()).cuda()*k ,IL, FL)
            deviations = torch.abs((approximated_p - p.data)/p.data)
            tolerable_mask_in_pruning_zone = (1 - pruned_inds) * (deviations < 0.25)
            #allowed_masks.append(tolerable_mask_in_pruning_zone)
            mask_amul[tolerable_mask_in_pruning_zone] = k
        masks[mask_amul != (IL+FL)] = 1.0

        actual_pruning_perc = actual_pruning_perc + 1 - (torch.sum(p.data)/p.data.numel())[0]
        current_pruning_perc = (pruning_perc - actual_pruning_perc) / (iterations - 1 - i) + current_pruning_perc
        # retrain

    return masks, mask_amul





def prune_one_filter(model, masks):
    '''
    Pruning one least ``important'' feature map by the scaled l2norm of 
    kernel weights
    arXiv:1611.06440
    '''
    NO_MASKS = False
    # construct masks if there is not yet
    if not masks:
        masks = []
        NO_MASKS = True

    values = []
    for p in model.parameters():

        if len(p.data.size()) == 4: # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])

    assert len(masks) == len(values), "something wrong here"

    values = np.array(values)

    # set mask corresponding to the filter to prune
    to_prune_layer_ind = np.argmin(values[:, 0])
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.

    print('Prune filter #{} in layer #{}'.format(
        to_prune_filter_ind, 
        to_prune_layer_ind))

    return masks


def filter_prune(model, pruning_perc):
    '''
    Prune filters one by one until reach pruning_perc
    (not iterative pruning)
    '''
    masks = []
    current_pruning_perc = 0.

    while current_pruning_perc < pruning_perc:
        masks = prune_one_filter(model, masks)
        model.set_masks(masks)
        current_pruning_perc = prune_rate(model, verbose=False)
        print('{:.2f} pruned'.format(current_pruning_perc))

    return masks
