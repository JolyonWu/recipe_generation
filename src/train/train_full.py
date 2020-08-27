# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import sys
sys.path.append('../')
sys.path.append('../model')
from args import get_parser
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from model_full import get_model_full
from model_fomer import  get_model_fomer
from build_vocab import Vocabulary
from torchvision import transforms
import time
import torch.backends.cudnn as cudnn
from utils.tb_visualizer import Visualizer
import utils
from utils.utils_all import *
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'

#对模型的前半部分参数进行加载
def merge_models(args, model, ingr_vocab_size, action_vocab_size):
    load_args = pickle.load(open(os.path.join(args.save_dir, args.project_name,
                                              args.transfer_from, 'checkpoints/args_4_.pkl'), 'rb'))

    model_fomer_ = get_model_fomer(load_args, ingr_vocab_size, action_vocab_size)
    model_path = os.path.join(args.save_dir, args.project_name, args.transfer_from, 'checkpoints', 'model_4_.ckpt')

    # Load the trained model parameters
    model_fomer_.load_state_dict(torch.load(model_path, map_location=map_loc))
    model.ingredient_decoder = model_fomer_.ingredient_decoder
    model.action_decoder = model_fomer_.action_decoder
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    for p in model.ingredient_decoder.parameters():
        p.requires_grad = False
    for p in model.action_decoder.parameters():
        p.requires_grad = False
    return  model



def main(args):


    # Create model directory & other aux folders for logging
    where_to_save = os.path.join(args.save_dir, args.project_name, args.model_name)
    checkpoints_dir = os.path.join(where_to_save, 'checkpoints')
    logs_dir = os.path.join(where_to_save, 'logs')
    tb_logs = os.path.join(args.save_dir, args.project_name, 'tb_logs', args.model_name)
    make_dir(where_to_save)
    make_dir(logs_dir)
    make_dir(checkpoints_dir)
    make_dir(tb_logs)
    if args.tensorboard:
        logger = Visualizer(tb_logs, name='visual_results')

    #check if we want to resume from last checkpoint of current model
    if args.resume:
        args = pickle.load(open(os.path.join(checkpoints_dir, 'args_' + str(args.resume_epoch) + '_.pkl'), 'rb'))
        args.resume = True
    print(args)
    # logs to disk
    if not args.log_term:
        print ("Training logs will be saved to:", os.path.join(logs_dir, 'train.log'))
        sys.stdout = open(os.path.join(logs_dir, 'train.log'), 'w')
        sys.stderr = open(os.path.join(logs_dir, 'train.err'), 'w')

    # patience init
    curr_pat = 0

    # Build data loader
    data_loaders = {}
    datasets = {}

    for split in ['train', 'test']:

        transforms_list = []
        transforms_list.append(transforms.CenterCrop(args.crop_size))
        transforms_list.append(transforms.ToTensor())
        transform = transforms.Compose(transforms_list)

        max_num_samples = max(args.max_eval, args.batch_size) if split == 'test' else -1
        data_loaders[split], datasets[split] = get_loader(args.recipe_dir, args.aux_data_dir, args.lmdb_data_dir, split,
                                                          args.maxseqlen,
                                                          args.maxnuminstrs,
                                                          args.maxnumlabels,
                                                          args.maxnumactions,
                                                          args.maxnumims,
                                                          transform, args.batch_size,
                                                          shuffle=False, num_workers=args.num_workers,
                                                          drop_last=True,
                                                          max_num_samples=max_num_samples,
                                                          use_lmdb=args.use_lmdb,
                                                          suff=args.suff)

    ingr_vocab_size = datasets[split].get_ingrs_vocab_size()
    action_vocab_size = datasets[split].get_action_vocab_size()
    instrs_vocab_size = datasets[split].get_instrs_vocab_size()

    # Build the model
    model = get_model_full(args, ingr_vocab_size,action_vocab_size,instrs_vocab_size)
    keep_cnn_gradients = False
    decay_factor = 1.0

    #################################
    #这一部分用来加载之前训练的前半部分模型的参数
    model=merge_models(args, model, ingr_vocab_size, action_vocab_size)

    params = list(model.encoder_ingrs.parameters())+list(model.encoder_action.parameters())\
             +list(model.decoder_instr.parameters())

    print ("decoder + 2 encoder params:", sum(p.numel() for p in params if p.requires_grad))
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    if args.resume:
        print("model have been loading from checkpoints")

        model_path = os.path.join(args.save_dir, args.project_name, args.model_name, 'checkpoints',
                                  'model_' + str(args.resume_epoch) + '_.ckpt')
        optim_path = os.path.join(args.save_dir, args.project_name, args.model_name, 'checkpoints',
                                  'optim_' + str(args.resume_epoch) + '_.ckpt')
        optimizer.load_state_dict(torch.load(optim_path, map_location=map_loc))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        model.load_state_dict(torch.load(model_path, map_location=map_loc))

    if device != 'cpu' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    cudnn.benchmark = True



    es_best = 10000 if args.es_metric == 'loss' else 0
    # Train the model
    if args.resume:
        start = args.current_epoch+1
    else:
        start=0
    for epoch in range(start, args.num_epochs):

        # save current epoch for resuming
        if args.tensorboard:
            logger.reset()

        args.current_epoch = epoch
        # increase / decrase values for moving params
        if args.decay_lr:
            frac = epoch // args.lr_decay_every
            decay_factor = args.lr_decay_rate ** frac
            new_lr = args.learning_rate*decay_factor
            print ('Epoch %d. lr: %.5f'%(epoch, new_lr))
            set_lr(optimizer, decay_factor)

        for split in ['train', 'test']:

            if split == 'train':
                model.train()
            else:
                model.eval()
            total_loss_dict = {'recipe_loss': [], 'ingr_loss': [],'ingr_iou': [], 'action_loss': [],'action_iou': [], 'loss': [],
                                'perplexity': [], 'iou_sample': [], 'f1': [],
                              }

            torch.cuda.synchronize()
            start=time.time()
            i=0
            with torch.autograd.set_detect_anomaly(True):
                for img_inputs, captions, ingr_gt,action_gt in data_loaders[split]:
                    i=i+1
                    loss_dict = {}
                    length_steps=img_inputs.size(1)
                    img_inputs = img_inputs.to(device)
                    captions = captions.to(device)
                    ingr_gt = ingr_gt.to(device)
                    action_gt=action_gt.to(device)

                    for steps in range( length_steps):
                        image_step=img_inputs[:,steps,:,:,:]
                        caption_step=captions[:,steps,:]
                        ingrs_step=ingr_gt[:,steps,:]
                        action_step=action_gt[:,steps,:]

                        if split == 'train':
                            losses = model(image_step, ingrs_step,action_step,caption_step,
                                           keep_cnn_gradients=keep_cnn_gradients)
                        else:
                            with torch.no_grad():
                                losses = model(image_step,  ingrs_step,action_step,caption_step)

                        #recipe部分
                        recipe_loss = losses['recipe_loss']
                        recipe_loss = recipe_loss.mean()
                        loss_dict['recipe_loss'] = recipe_loss.item()

                        loss =  args.loss_weight[0] * recipe_loss
                        loss_dict['loss'] = loss.item()

                        for key in loss_dict.keys():
                            total_loss_dict[key].append(loss_dict[key])

                        if split == 'train':
                            model.zero_grad()
                            loss.backward()
                            optimizer.step()


                    if args.tensorboard:

                        logger.scalar_summary(mode=split + '_iter', epoch= len(data_loaders[split]) * epoch + i,
                                              **{k: np.mean(v[-args.log_step:]) for k, v in total_loss_dict.items()
                                                 if v})

            torch.cuda.synchronize()
            print(epoch,"epoch: ",time.time()-start)
            if args.tensorboard:
                logger.scalar_summary(mode=split,
                                      epoch=epoch,
                                      **{k: np.mean(v) for k, v in total_loss_dict.items() if v})

        # Save the model's best checkpoint if performance was improved
        es_value = np.mean(total_loss_dict[args.es_metric])

        # save current model as well
        save_model(model, optimizer, checkpoints_dir, str(int(epoch / 10)), suff='')
        pickle.dump(args, open(os.path.join(checkpoints_dir, 'args_' + str(int(epoch / 10)) + '_.pkl'), 'wb'))
        print('Saved checkpoint.')
        if (args.es_metric == 'loss' and es_value < es_best) or (args.es_metric == 'iou' and es_value > es_best):
            es_best = es_value
            save_model_best(model, optimizer, checkpoints_dir, suff='best')
            pickle.dump(args, open(os.path.join(checkpoints_dir, 'args_best.pkl'), 'wb'))
            curr_pat = 0
            print('Saved Best checkpoint.')
        else:
            curr_pat =curr_pat+ 1

        if curr_pat > args.patience:
            break

    if args.tensorboard:
        logger.close()


if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)
