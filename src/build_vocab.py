# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import nltk
import pickle
import argparse
from collections import Counter
import json
import os
from tqdm import *
import numpy as np
import re


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        return self.idx


    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<pad>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def return_word(self,idx):
        return self.idx2word[idx]





def get_instruction(instruction, replace_dict):
    instruction = instruction.lower()

    for rep, char_list in replace_dict.items():
        for c_ in char_list:
            if c_ in instruction:
                instruction = instruction.replace(c_, rep)
        instruction = instruction.strip()

    return instruction





def update_counter(list_, counter_toks):
    for sentence in list_:
        tokens = nltk.tokenize.word_tokenize(sentence)
        counter_toks.update(tokens)


def build_vocab_recipe(args):
    print ("Loading data...")

    dataset=json.load(open(os.path.join(args.recipe_dir, 'dataset_extract.json'), 'r'))


    print("Loaded data.")
    print("Found %d recipes in the dataset." % (len(dataset)))
    replace_dict_instrs = {'and': ['&', "'n"], ' ': ['%', ',', '.', '#', '[', ']', '!', '?','(',')']}



    ingrs_file = args.save_dir + 'all_ingrs_count.pkl'
    instrs_file = args.save_dir + 'all_instrs_count.pkl'
    action_file=args.save_dir + 'all_action_count.pkl'
    #####
    # 1. Count words in dataset and clean
    #####
    if os.path.exists(ingrs_file) and os.path.exists(instrs_file) :
        print ("loading pre-extracted word counters")
        counter_ingrs = pickle.load(open(args.save_dir + 'all_ingrs_count.pkl', 'rb'))
        counter_toks = pickle.load(open(args.save_dir + 'all_instrs_count.pkl', 'rb'))
        counter_action = pickle.load(open(args.save_dir + 'all_action_count.pkl', 'rb'))
    else:
        counter_toks = Counter()
        counter_ingrs = Counter()
        counter_ingrs_raw = Counter()
        counter_action= Counter()

        for i, entry in tqdm(enumerate(dataset)):

            # get all instructions for this recipe
            steps = entry['steps']
            instrs_list = []
            ingrs_list = []
            action_list=[]
            ingredients=[]

            for ingre in range(len(steps)):
                ingredients.extend(steps[ingre]['step_ingredients'])
                action_list.extend(steps[ingre]['step_action'])
            det_ingrs_filtered = []
            for j, ingrs in enumerate(ingredients):
                if len(ingrs) > 0 :
                    det_ingrs_filtered.append(ingrs)
                    ingrs_list.append(ingrs)

            # get raw text for instructions of this entry
            acc_len = 0
            for instr in steps:
                instr = instr['text']
                instr = get_instruction(instr, replace_dict_instrs)
                if len(instr) > 0:
                    instrs_list.append(instr)
                    acc_len += len(instr)

            # discard recipes with too few or too many ingredients or instruction words
            # if len(ingrs_list) < args.minnumingrs or len(instrs_list) < args.minnuminstrs \
            #         or len(instrs_list) >= args.maxnuminstrs or len(ingrs_list) >= args.maxnumingrs \
            #         or acc_len < args.minnumwords:
            #     continue

            # tokenize sentences and update counter
            update_counter(instrs_list, counter_toks)
            counter_ingrs.update(ingrs_list)
            counter_action.update(action_list)

        pickle.dump(counter_ingrs, open(args.save_dir + 'all_ingrs_count.pkl', 'wb'))
        pickle.dump(counter_toks, open(args.save_dir + 'all_instrs_count.pkl', 'wb'))
        pickle.dump(counter_ingrs_raw, open(args.save_dir + 'all_ingrs_raw_count.pkl', 'wb'))
        pickle.dump(counter_action, open(args.save_dir + 'all_action_count.pkl', 'wb'))

    # manually add missing entries for better clustering
    base_words = ['peppers', 'tomato', 'spinach_leaves', 'turkey_breast', 'lettuce_leaf',
                  'chicken_thighs', 'milk_powder', 'bread_crumbs', 'onion_flakes',
                  'red_pepper', 'pepper_flakes', 'juice_concentrate', 'cracker_crumbs', 'hot_chili',
                  'seasoning_mix', 'dill_weed', 'pepper_sauce', 'sprouts', 'cooking_spray', 'cheese_blend',
                  'basil_leaves', 'pineapple_chunks', 'marshmallow', 'chile_powder',
                  'cheese_blend', 'corn_kernels', 'tomato_sauce', 'chickens', 'cracker_crust',
                  'lemonade_concentrate', 'red_chili', 'mushroom_caps', 'mushroom_cap', 'breaded_chicken',
                  'frozen_pineapple', 'pineapple_chunks', 'seasoning_mix', 'seaweed', 'onion_flakes',
                  'bouillon_granules', 'lettuce_leaf', 'stuffing_mix', 'parsley_flakes', 'chicken_breast',
                  'basil_leaves', 'baguettes', 'green_tea', 'peanut_butter', 'green_onion', 'fresh_cilantro',
                  'breaded_chicken', 'hot_pepper', 'dried_lavender', 'white_chocolate',
                  'dill_weed', 'cake_mix', 'cheese_spread', 'turkey_breast', 'chucken_thighs', 'basil_leaves',
                  'mandarin_orange', 'laurel', 'cabbage_head', 'pistachio', 'cheese_dip',
                  'thyme_leave', 'boneless_pork', 'red_pepper', 'onion_dip', 'skinless_chicken', 'dark_chocolate',
                  'canned_corn', 'muffin', 'cracker_crust', 'bread_crumbs', 'frozen_broccoli',
                  'philadelphia', 'cracker_crust', 'chicken_breast']

    for base_word in base_words:
        base_word = base_word.replace('_', ' ')
        if base_word not in counter_ingrs.keys():
            counter_ingrs[base_word] = 1



    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter_toks.items() if cnt >= args.threshold_words]
    ingrs = [word for word, cnt in counter_ingrs.items() if cnt >= args.threshold_ingrs]
    action=[word for word, cnt in counter_action.items() if cnt >= args.threshold_action]

    # Recipe vocab
    # Create a vocab wrapper and add some special tokens.
    vocab_toks = Vocabulary()
    vocab_toks.add_word('<start>')
    vocab_toks.add_word('<end>')
    vocab_toks.add_word('<eoi>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab_toks.add_word(word)
    vocab_toks.add_word('<pad>')

    # Ingredient vocab
    # Create a vocab wrapper for ingredients
    vocab_ingrs = Vocabulary()
    vocab_ingrs.add_word('<end>')
    for i, word in enumerate(ingrs):
        vocab_ingrs.add_word(word)
    vocab_ingrs.add_word('<pad>')

    vocab_action = Vocabulary()
    vocab_action.add_word('<end>')
    for i, word in enumerate(action):
        vocab_action.add_word(word)
    vocab_action.add_word('<pad>')




    print("Total ingr vocabulary size: {}".format(len(vocab_ingrs)))
    print("Total token vocabulary size: {}".format(len(vocab_toks)))
    print("Total action vocabulary size: {}".format(len(vocab_action)))

    return vocab_ingrs, vocab_toks,vocab_action


def main(args):

    vocab_ingrs, vocab_toks ,vocab_action= build_vocab_recipe(args)

    with open(os.path.join(args.save_dir, args.suff+'recipe_vocab_ingrs.pkl'), 'wb') as f:
        pickle.dump(vocab_ingrs, f)
    with open(os.path.join(args.save_dir, args.suff+'recipe_vocab_toks.pkl'), 'wb') as f:
        pickle.dump(vocab_toks, f)
    with open(os.path.join(args.save_dir, args.suff + 'recipe_vocab_action.pkl'), 'wb') as f:
        pickle.dump(vocab_action, f)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe_dir', type=str,
                        default='/DATACENTER/3/wjl/Recipe_generation_our/',
                        help='recipe path')

    parser.add_argument('--save_dir', type=str, default='/DATACENTER/3/wjl/Recipe_generation_our/',
                        help='path for saving vocabulary wrapper')

    parser.add_argument('--suff', type=str, default='')

    parser.add_argument('--threshold_ingrs', type=int, default=3,
                        help='minimum ingr count threshold')

    parser.add_argument('--threshold_words', type=int, default=3,
                        help='minimum word count threshold')

    parser.add_argument('--threshold_action', type=int, default=3,
                        help='minimum action count threshold')

    parser.add_argument('--maxnuminstrs', type=int, default=40,
                        help='max number of instructions (sentences)')

    parser.add_argument('--maxnumingrs', type=int, default=20,
                        help='max number of ingredients')

    parser.add_argument('--minnuminstrs', type=int, default=2,
                        help='min number of instructions (sentences)')

    parser.add_argument('--minnumingrs', type=int, default=2,
                        help='min number of ingredients')

    parser.add_argument('--minnumwords', type=int, default=20,
                        help='minimum number of characters in recipe')

    args = parser.parse_args()
    main(args)
