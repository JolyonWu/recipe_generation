
import torch
import os
import pickle
import sys
sys.path.append('../')
sys.path.append('../model')

map_loc = None if torch.cuda.is_available() else 'cpu'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#将label转换为one-hot格式
def label2onehot(labels, pad_value):

    # input labels to one hot vector
    inp_ = torch.unsqueeze(labels, 2)
    one_hot = torch.FloatTensor(labels.size(0), labels.size(1), pad_value + 1).zero_().to(device)
    one_hot.scatter_(2, inp_, 1)
    one_hot, _ = one_hot.max(dim=1)
    # remove pad position
    one_hot = one_hot[:, :-1]
    # eos position is always 0
    one_hot[:, 0] = 0

    return one_hot

#找到label中填充<pad>的部分
def mask_from_eos(ids, eos_value, mult_before=True):
    mask = torch.ones(ids.size()).to(device).byte()
    mask_aux = torch.ones(ids.size(0)).to(device).byte()

    # find eos in ingredient prediction
    for idx in range(ids.size(1)):
        # force mask to have 1s in the first position to avoid division by 0 when predictions start with eos
        if idx == 0:
            continue
        if mult_before:
            mask[:, idx] = mask[:, idx] * mask_aux
            mask_aux = mask_aux * (ids[:, idx] != eos_value)
        else:
            mask_aux = mask_aux * (ids[:, idx] != eos_value)
            mask[:, idx] = mask[:, idx] * mask_aux
    return mask

def save_model(model, optimizer, checkpoints_dir, numepoch,suff=''):
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(
            checkpoints_dir, 'model_'+numepoch+'_' + suff + '.ckpt'))

    else:
        torch.save(model.state_dict(), os.path.join(
            checkpoints_dir, 'model_'+numepoch+'_' + suff + '.ckpt'))

    torch.save(optimizer.state_dict(), os.path.join(
        checkpoints_dir, 'optim_'+numepoch+'_' + suff + '.ckpt'))

def save_model_best(model, optimizer, checkpoints_dir, suff=''):
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(
            checkpoints_dir, 'model_'+ suff + '.ckpt'))

    else:
        torch.save(model.state_dict(), os.path.join(
            checkpoints_dir,  'model_'+ suff + '.ckpt'))

    torch.save(optimizer.state_dict(), os.path.join(
        checkpoints_dir, 'optim_' + suff + '.ckpt'))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_lr(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group['lr'] = group['lr']*decay_factor


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

