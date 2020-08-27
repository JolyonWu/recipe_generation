import json
import argparse
import os
from tqdm import *
from random import shuffle
from vocabulary import clean_data


def main(args):
    file = open(os.path.join(args.recipe_path,  'dataset_extract.json'), 'r')
    train_file = open(os.path.join(args.save_path,  'train.json'), 'w')
    test_file = open(os.path.join(args.save_path,  'test.json'), 'w')
    oringin_dataset = json.load(file)
    length = len(oringin_dataset)
    image_dict=dict()
    for i in tqdm(reversed(range(length))):
        recipe=oringin_dataset[i]

        images=recipe['images']
        image_dict.clear()
        for j in range(len(images)):
            image_dict[images[j][0]]=images[j][1]
        for j in range(len(recipe['steps'])):
            step=recipe['steps'][j]
            step['text']=clean_data(step['text'])
            length_image=len(step['images'])
            temp=step['images'].copy()
            step['images'].clear()
            for k in range(length_image):
                if temp[k] in image_dict:
                    step['images'].append(image_dict[temp[k]])
            step['images']=step['images'][:args.max_num_image]
        del recipe['images']
        if len(recipe['steps'])==0:
            del oringin_dataset[i]


    shuffle(oringin_dataset)
    train = []
    test = []
    for i in range(int(len(oringin_dataset) / 10 * 9)):
        train.append(oringin_dataset[i])
    for i in range(int(len(oringin_dataset) / 10 * 9), len(oringin_dataset)):
        test.append(oringin_dataset[i])
    train=sorted(train, key=lambda recipe: len(recipe['steps']))
    test=sorted(test, key=lambda recipe: len(recipe['steps']))
    json.dump(train, train_file, indent=4)
    json.dump(test, test_file, indent=4)
    train_file.close()
    test_file.close()
    file.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe_path', type=str,
                        default='/DATACENTER/3/wjl/Recipe_generation_our/',
                        help='recipe path')

    parser.add_argument('--save_path', type=str, default='/DATACENTER/3/wjl/Recipe_generation_our/',
                        help='path for saving ')

    parser.add_argument('--max_num_image', type=int, default=3,
                        help='max numbers of images every step ')

    args = parser.parse_args()
    main(args)