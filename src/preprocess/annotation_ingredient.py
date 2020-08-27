import nltk
import pickle
import argparse
from collections import Counter
import json
import os
from tqdm import *
import numpy as np
import re
import sys
from vocabulary import *

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


def main(args):
    vocab_ingrs = Vocabulary()
    s_ingrs=set()
    with open(os.path.join(args.recipe_path+'recipe1m_vocab_ingrs.pkl'), 'rb') as f:
        vocab_ingrs=pickle.load(f)
    count=len(vocab_ingrs)
    for i in range(1, count):
        t = vocab_ingrs.return_word(i)
        for j in range(len(t)):
            word = t[j]
            word = word.replace('_', ' ')
            word = word.replace('-', '')
            s_ingrs.add(word)
    for word in base_words:
        word = word.replace('_', ' ')
        word = word.replace('-', '')
        s_ingrs.add(word)


    if args.start==0:
        add=os.path.join(args.recipe_path + 'oringin_dataset_new.json')
    else:
        add=os.path.join(args.recipe_path + 'oringin_dataset_new_anatation.json')
    with open(add, 'rb') as h:
        oringin_dataset=json.load(h)

        #match用来表示每个食谱里总共拥有的ingredient
        #anatation用来表示自己对每个recipe的标注
        #step_anatation表示每一个step的人工标注
        match = set()
        anatation=set()
        step_anatation=set()
        for i in range(args.start,50):
            print('这是第',i,'个recipe')
            recipe=oringin_dataset[i]['steps']
            length_step=len(recipe)
            match.clear()
            anatation.clear()
            for j in range(length_step):
                step_anatation.clear()
                text=recipe[j]['text']
                text=clean_data(text)
                print('NO.',j,'   ',text)

                step_ingredient=recipe[j]['ingredients']
                print('match:',step_ingredient )
                match=match.union(set(step_ingredient))
                while True:
                    ana=input("请输入ingredient:")
                    if ana =='' \
                             '':
                        break
                    step_anatation.add(ana)
                    anatation.add(ana)
                oringin_dataset[i]['steps'][j]['step_anatation']=list(step_anatation)
            oringin_dataset[i]['match']=list(match)
            oringin_dataset[i]['anatation'] = list(anatation)
            with open(os.path.join(args.save_path + 'oringin_dataset_new_anatation.json'), 'w') as l:
                json.dump(oringin_dataset, l, indent=4)

    with open(os.path.join(args.save_path + 'oringin_dataset_new_anatation.json'), 'w') as l:
        json.dump(oringin_dataset,l,indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe_path', type=str,
                        default='/DATACENTER/3/wjl/inversecooking/data/',
                        help='recipe1m path')
    parser.add_argument('--save_path', type=str, default='../data/',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--start', type=int, default=0,
                        help='为了快速恢复打标签的过程')
    args = parser.parse_args()
    main(args)