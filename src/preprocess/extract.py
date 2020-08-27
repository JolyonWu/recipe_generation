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
                  'philadelphia', 'cracker_crust', 'chicken_breast','tomatoes','juices','chilli','sauce','dough','cabage','soy_sauce','chilly_powder','beans','apple_sauce','applesauce',
              'sardine','noodles','noodle','applesauce']


def main(args):
    vocab_ingrs = Vocabulary()
    #s_ingrs是一个集合，里面包含着recipe_1M这个数据集工作，抽取出来的所有ingredient
    s_ingrs=set()
    with open(os.path.join(args.recipe_path+'recipe1m_vocab_ingrs.pkl'), 'rb') as f:
        vocab_ingrs=pickle.load(f)
    count=len(vocab_ingrs)
    for i in range(1,count):
        t=vocab_ingrs.return_word(i)
        for j in range(len(t)):
            word=t[j]
            word=word.replace('_',' ')
            word = word.replace('-', ' ')
            s_ingrs.add(word)
    for word in  base_words:
        word = word.replace('_', ' ')
        word=word.replace('-',' ')
        s_ingrs.add(word)

    #这个代码是在创建action,这个集合包含了所有的action
    if os.path.exists(os.path.join(args.recipe_path + 'recipe_action.pkl')):
        with open(os.path.join(args.recipe_path + 'recipe_action.pkl'), 'rb') as f:
            action=pickle.load(f)

    else:
        with open(os.path.join(args.recipe_path + 'dataset_action_anatation.json'), 'rb') as m:
            action=set()
            dataset_action=json.load(m)
            for i in range(31):
                recipe = dataset_action[i]['steps']
                length_step = len(recipe)
                for j in range(length_step):
                    action.update(recipe[j]['step_action'])

            with open(os.path.join(args.recipe_path + 'recipe_action.pkl'), 'wb') as k:
                pickle.dump(action,k)




    ##########################################################
    # 这一步是做冗余step清除的工作
    with open(os.path.join(args.recipe_path + 'dataset.json'), 'rb') as h:
        with open(os.path.join(args.recipe_path + 'dataset_filter.json'), 'w') as g:
            oringin_dataset=json.load(h)
            length_dataset=len(oringin_dataset)
            for i in range(length_dataset):
                recipe = oringin_dataset[i]['steps']
                del oringin_dataset[i]['source']
                length_step = len(recipe)
                for j in reversed(range(length_step)):
                    if recipe[j]['images'][0]==-1:
                        del recipe[j]
                    else:
                        del recipe[j]['ID']

                for j in range(len(recipe)):
                    if 0 in recipe[j]['images']  and j == 0:
                        recipe[j]['images']=[1]
                    if 0 in recipe[j]['images'] and j != 0:
                        recipe[j]['images'] = recipe[j - 1]['images']




                for j in range(len(recipe)):
                    recipe[j]['index']=j
            json.dump(oringin_dataset, g, indent=4)





    ##########################################################
    #这一步是做特征抽取的工作

    with open(os.path.join(args.recipe_path + 'dataset_filter.json'), 'rb') as h:
        oringin_dataset=json.load(h)
        length_dataset=len(oringin_dataset)
        for i in range(length_dataset):
            recipe=oringin_dataset[i]['steps']
            length_step=len(recipe)
            union_action=set()
            union = set()
            for j in range(length_step):
                text=recipe[j]['text']
                text=clean_data(text)
                text_list = nltk.tokenize.word_tokenize(text)

                union_action.clear()
                text_set=set(text_list)
                for k in range(len(text_list)-1):
                    word=text_list[k]+' '+text_list[k+1]
                    text_set.add(word)
                union_action = text_set.intersection(action)



                #以下这一段是采用set集合匹配的方法
                # text_set=set(text_list)
                # for k in range(len(text_list)-1):
                #     word=text_list[k]+' '+text_list[k+1]
                #     text_set.add(word)
                # union = text_set.intersection(s_ingrs)

                #以下这一段是采用其他方法
                union.clear()
                flag=0
                for k in range(len(text_list)-1):
                    if flag==1:
                        flag=0
                        continue
                    word=text_list[k]+' '+text_list[k+1]
                    if word in s_ingrs:
                        union.add(word)
                        flag=1
                    else:
                        word=text_list[k]
                        if word in s_ingrs:
                            union.add(word)
                        word = text_list[k+1]
                        if word in s_ingrs:
                            union.add(word)


                step_ingredient=list(union)
                step_action=list(union_action)
                oringin_dataset[i]['steps'][j]['step_ingredients']=step_ingredient
                oringin_dataset[i]['steps'][j]['step_action'] = step_action

    with open(os.path.join(args.save_path + 'dataset_extract.json'), 'w') as l:
        json.dump(oringin_dataset,l,indent=4)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe_path', type=str,
                        default='/DATACENTER/3/wjl/Recipe_generation_our/',
                        help='recipe path')
    parser.add_argument('--save_path', type=str,
                        default='/DATACENTER/3/wjl/Recipe_generation_our/',
                        help='save path')

    args = parser.parse_args()
    main(args)