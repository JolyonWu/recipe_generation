# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
import random
import json
import lmdb
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from torch.utils.data.distributed import DistributedSampler
from tqdm import *

class RecipeDataset(data.Dataset):

    def __init__(self, data_dir, aux_data_dir,lmdb_data_dir ,split, maxseqlen, maxnuminstrs, maxnumlabels,maxnumactions, maxnumims,
                 transform=None, max_num_samples=-1, use_lmdb=False, suff=''):

        self.ingrs_vocab = pickle.load(open(os.path.join(aux_data_dir,  'recipe_vocab_ingrs.pkl'), 'rb'))
        self.instrs_vocab = pickle.load(open(os.path.join(aux_data_dir, 'recipe_vocab_toks.pkl'), 'rb'))
        self.action_vocab=pickle.load(open(os.path.join(aux_data_dir, 'recipe_vocab_action.pkl'), 'rb'))
        self.dataset = json.load(open(os.path.join(aux_data_dir, split+'.json'), 'r'))

        self.label2word = self.get_ingrs_vocab()

        self.use_lmdb = use_lmdb
        if use_lmdb:
            self.image_file = lmdb.open(os.path.join(lmdb_data_dir, 'lmdb' ), max_readers=1, readonly=True,
                                        lock=False, readahead=False, meminit=False)
        self.split = split
        self.root = data_dir
        self.transform = transform
        self.max_num_labels = maxnumlabels
        self.max_num_actions=maxnumactions
        self.maxseqlen = maxseqlen
        self.max_num_instrs = maxnuminstrs
        self.maxseqlen = maxseqlen*maxnuminstrs
        self.maxnumims = maxnumims

    def get_instrs_vocab(self):
        return self.instrs_vocab

    def get_instrs_vocab_size(self):
        return len(self.instrs_vocab)

    def get_ingrs_vocab(self):
        return self.ingrs_vocab

    def get_ingrs_vocab_size(self):
        return len(self.ingrs_vocab)

    def get_action_vocab(self):
        return self.action_vocab

    def get_action_vocab_size(self):
        return len(self.action_vocab)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        sample = self.dataset[index]
        steps = sample['steps']
        min_length=min(len(steps),self.max_num_instrs)
        image_list=[torch.zeros(1)]*min_length
        caption_list = [torch.zeros(1)]*min_length
        ingredient_list = [torch.zeros(1)]*min_length
        action_list=[torch.zeros(1)]*min_length
        for k in range(min_length):

            paths = steps[k]['images']

            if len(paths)==0:
                image=Image.fromarray(np.zeros([400, 400, 3], np.uint8) + 254).convert('RGB')
            else:
                # if self.split == 'train':
                #     img_idx = np.random.randint(0, len(paths))
                # else:
                #     img_idx = 0
                img_idx = 0
                path = paths[img_idx]
                if self.use_lmdb:
                    with self.image_file.begin(write=False) as txn:
                        image = txn.get(path.encode())
                        image = np.fromstring(image, dtype=np.uint8)
                        image = np.reshape(image, (224, 224, 3))
                    image = Image.fromarray(image.astype('uint8'), 'RGB')
                else:
                    image = Image.open(os.path.join(self.root, path)).convert('RGB')

            ingrs_labels = steps[k]['step_ingredients']
            action_labels=steps[k]['step_action']
            captions = steps[k]['text']
            tokens = nltk.tokenize.word_tokenize(captions)
            ingrs_labels_gt = np.ones(self.max_num_labels) * self.ingrs_vocab('<pad>')
            action_labels_gt = np.ones(self.max_num_actions) * self.action_vocab('<pad>')


            #这个部分是将ingredient编成label
            pos = 0
            true_ingr_idxs = []
            for i in range(len( ingrs_labels)):
                true_ingr_idxs.append(self.ingrs_vocab( ingrs_labels[i]))
            for i in range(self.max_num_labels-1):
                if i >= len( ingrs_labels):
                    label = '<pad>'
                else:
                    label =  ingrs_labels[i]
                label_idx = self.ingrs_vocab(label)
                if label_idx not in ingrs_labels_gt:
                    ingrs_labels_gt[pos] = label_idx
                    pos += 1
            ingrs_labels_gt[pos ] = self.ingrs_vocab('<end>')
            ingrs_gt = torch.from_numpy(ingrs_labels_gt).long()

            # 这个部分是将action编成label
            pos = 0
            true_action_idxs = []
            for i in range(len(action_labels)):
                true_action_idxs.append(self.action_vocab(action_labels[i]))
            for i in range(self.max_num_actions - 1):
                if i >= len(action_labels):
                    label = '<pad>'
                else:
                    label = action_labels[i]
                label_idx = self.action_vocab(label)
                if label_idx not in action_labels_gt:
                    action_labels_gt[pos] = label_idx
                    pos += 1
            action_labels_gt[pos] = self.action_vocab('<end>')
            action_gt = torch.from_numpy(action_labels_gt).long()




            # Convert caption (string) to word ids.
            caption = []
            caption = self.caption_to_idxs(tokens, caption)
            caption = caption[0:self.maxseqlen-1]
            caption.append(self.instrs_vocab('<end>'))
            target = torch.Tensor(caption)

            if self.transform is not None:
                image = self.transform(image)
            image_input = image

            image_list[k]=image_input
            caption_list[k]=target
            ingredient_list[k]=ingrs_gt
            action_list[k]=action_gt

            ###############################
            # image_list.append(image_input)
            # caption_list.append(target)
            # ingredient_list.append(ingrs_gt)
            # action_list.append(action_gt)

        image_input=torch.stack(image_list,0)
        lengths = [cap.size(0) for cap in caption_list]
        captipn_input = torch.ones(len(caption_list), max(lengths)).long() * self.instrs_vocab('<pad>')
        for i, cap in enumerate(caption_list):
            end = lengths[i]
            captipn_input[i, :end] = cap[:end]

        ingrs_gt = torch.stack(ingredient_list, 0)
        action_gt = torch.stack(action_list, 0)

        return image_input, captipn_input, ingrs_gt, action_gt, self.instrs_vocab('<pad>'),self.ingrs_vocab('<pad>'),self.action_vocab('<pad>')

    def __len__(self):
        return len(self.dataset)

    def caption_to_idxs(self, tokens, caption):

        caption.append(self.instrs_vocab('<start>'))
        for token in tokens:
            caption.append(self.instrs_vocab(token))
        return caption


def collate_fn(data):

    # Sort a data list by caption length (descending order).
    # data.sort(key=lambda x: len(x[2]), reverse=True)
    image_input, captions, ingrs_gt,action_gt,  pad_value_instruction,pad_value_ingredient,pad_value_action = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).

    lengths_1d = [img.size(0) for img in image_input]
    image= torch.zeros(len(lengths_1d), max( lengths_1d),3,224,224)
    for i ,img in enumerate(image_input):
        image[i,:lengths_1d[i],:,:,:]=img[:lengths_1d[i],:,:,:]
    #这是ingredient的
    lengths_1d = [ingre.size(0) for ingre in ingrs_gt]
    ingredient=torch.ones(len( lengths_1d ), max( lengths_1d), ingrs_gt[0].size(1)).long() *pad_value_ingredient[0]
    for i,ingr  in enumerate( ingrs_gt):
        end_1d = lengths_1d[i]
        ingredient[i,:end_1d,:]=ingr[:end_1d,:]

    # 这是action的
    lengths_1d = [act.size(0) for act in action_gt]
    action = torch.ones(len(lengths_1d), max(lengths_1d), action_gt[0].size(1)).long() * pad_value_action[0]
    for i, act in enumerate(action_gt):
        end_1d = lengths_1d[i]
        action[i, :end_1d, :] = act[:end_1d, :]



    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths_1d = [cap.size(0) for cap in captions]
    lengths_2d=[cap.size(1) for cap in captions]
    targets = torch.ones(len(captions), max( lengths_1d), max( lengths_2d)).long() *pad_value_instruction[0]

    for i, cap in enumerate(captions):
        end_1d = lengths_1d[i]
        end_2d = lengths_2d[i]
        targets[i,:end_1d, :end_2d] = cap[:end_1d, :end_2d]

    return image, targets, ingredient,action


def get_loader(data_dir, aux_data_dir,lmdb_data_dir, split, maxseqlen,
               maxnuminstrs, maxnumlabels,maxnumactions, maxnumims, transform, batch_size,
               shuffle, num_workers, drop_last=False,
               max_num_samples=-1,
               use_lmdb=False,
               suff=''):

    dataset = RecipeDataset(data_dir=data_dir, aux_data_dir=aux_data_dir,lmdb_data_dir=lmdb_data_dir, split=split,
                              maxseqlen=maxseqlen, maxnumlabels=maxnumlabels,maxnumactions=maxnumactions, maxnuminstrs=maxnuminstrs,
                              maxnumims=maxnumims,
                              transform=transform,
                              max_num_samples=max_num_samples,
                              use_lmdb=use_lmdb,
                              suff=suff)


    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                               drop_last=drop_last, collate_fn=collate_fn, pin_memory=True)
    return data_loader, dataset



