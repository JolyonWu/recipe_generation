
import argparse
import json
import os




def main(args):

    if args.start == 0:
        add = os.path.join(args.recipe_path + 'dataset.json')
    else:
        add = os.path.join(args.recipe_path + 'dataset(action).json')
    with open(add, 'rb') as h:
        oringin_dataset=json.load(h)


        #anatation用来表示自己对每个recipe的标注
        #step_anatation表示每一个step的人工标注

        anatation=set()
        step_anatation=set()
        for i in range(args.start,50):
            print('这是第',i,'个recipe')
            recipe=oringin_dataset[i]['steps']
            length_step=len(recipe)

            anatation.clear()
            for j in range(length_step):
                step_anatation.clear()
                text=recipe[j]['text']

                print('NO.',j,'   ',text)
                while True:
                    ana=input("请输入action:")
                    if ana =='' \
                             '':
                        break
                    step_anatation.add(ana)
                    anatation.add(ana)
                oringin_dataset[i]['steps'][j]['step_action']=list(step_anatation)

            oringin_dataset[i]['action'] = list(anatation)
            with open(os.path.join(args.save_path + 'dataset(action).json'), 'w') as l:
                json.dump(oringin_dataset, l, indent=4)

    with open(os.path.join(args.save_path + 'dataset(action).json'), 'w') as l:
        json.dump(oringin_dataset,l,indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe_path', type=str,
                        default='/DATACENTER/3/wjl/inversecooking/data/',
                        help='recipe1m path')
    parser.add_argument('--save_path', type=str, default='/DATACENTER/3/wjl/inversecooking/data/',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--start', type=int, default=0,
                        help='为了快速恢复打标签的过程')
    args = parser.parse_args()
    main(args)