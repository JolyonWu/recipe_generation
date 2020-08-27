import os
import pickle
import argparse
import json


def main(args):

    with open(os.path.join(args.recipe_path + 'dataset_extract.json'), 'rb') as f:
        dataset = json.load(f)
        #这是ingredient的准确度
        with open(os.path.join(args.recipe_path + 'dataset_ingredient_anatation.json'), 'rb') as h:
            ingredient_anatation=json.load(h)
            correct = 0
            total_ground = 0
            total_pre = 0
            match=set()
            anatation = set()
            for i in range(50):
                length_step=len(dataset[i]['steps'])
                for j  in range(length_step):
                    match.clear()
                    anatation.clear()
                    match=set(dataset[i]['steps'][j]['step_ingredients'])
                    anatation = set( ingredient_anatation[i]['steps'][j]['step_anatation'])
                    total_ground=total_ground+len(anatation)
                    total_pre=total_pre+len(match)
                    correct=correct+len(match.intersection(anatation))
            print('ingredient准确率：',correct/total_pre*100,'%')
            print('ingredient召回率：', correct /total_ground  * 100, '%')


        #这是action的准确度
        with open(os.path.join(args.recipe_path + 'dataset_action_anatation.json'), 'rb') as h:
            action_anatation=json.load(h)
            correct = 0
            total_ground = 0
            total_pre = 0
            match=set()
            anatation = set()
            for i in range(30):
                length_step=len(dataset[i]['steps'])
                for j  in range(length_step):
                    match.clear()
                    anatation.clear()
                    match=set(dataset[i]['steps'][j]['step_action'])
                    anatation = set( action_anatation[i]['steps'][j]['step_action'])
                    total_ground=total_ground+len(anatation)
                    total_pre=total_pre+len(match)
                    correct=correct+len(match.intersection(anatation))
            print('action准确率：',correct/total_pre*100,'%')
            print('action召回率：', correct /total_ground  * 100, '%')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe_path', type=str,
                        default='/DATACENTER/3/wjl/Recipe_generation_our/',
                        help='recipe path')



    args = parser.parse_args()
    main(args)