import json
import argparse
import os
from tqdm import *


def main(args):
    train_file = open(os.path.join(args.save_dir,  'train.json'), 'r')
    test_file = open(os.path.join(args.save_dir, 'test.json'), 'r')


    for split in  [train_file, test_file]:
        oringin_dataset = json.load(split)
        length = len(oringin_dataset)
        all_image = 0
        miss_image = 0
        all_step = 0
        miss_step = 0
        for i in tqdm(range(length)):
            recipe=oringin_dataset[i]
            for j in range(len(recipe['steps'])):
                images=recipe['steps'][j]['images']
                length_i=len(images)
                all_step=all_step+1


                for k in reversed(range(length_i)):
                    all_image=all_image+1
                    if os.path.exists(os.path.join(args.recipe_dir,images[k])) == False:
                        del oringin_dataset[i]['steps'][j]['images'][k]
                        miss_image=miss_image+1
                if len(oringin_dataset[i]['steps'][j]['images'])==0:
                    miss_step+=1
        # if split==train_file:
        #     json.dump(oringin_dataset, train_file, indent=4)
        # else:
        #     json.dump(oringin_dataset, test_file, indent=4)

        print('all_image:',all_image)
        print('miss_image:',miss_image)
        if split==train_file:
            print('train_dataset:')
        else:
            print('test_dataset')
        print('all_step:',all_step)
        print('miss_step:',miss_step)

    file = open(os.path.join(args.save_dir, 'dataset.json'), 'r')
    oringin_dataset = json.load(file)
    length = len(oringin_dataset)
    all_step = 0
    miss_step = 0
    for i in tqdm(range(length)):
        recipe = oringin_dataset[i]
        for j in range(len(recipe['steps'])):
            images = recipe['steps'][j]['images']
            all_step = all_step + 1
            if images[0]<1:
                miss_step+=1
    # print('all_step:', all_step)
    # print('miss_step:', miss_step)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe_dir', type=str,
                        default='/DATACENTER/3/wjl/Recipe_generation_our/',
                        help='recipe path')

    parser.add_argument('--save_dir', type=str, default='/DATACENTER/3/wjl/Recipe_generation_our/',
                        help='path for saving ')
    args = parser.parse_args()
    main(args)