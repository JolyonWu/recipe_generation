# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import pickle
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import argparse
import lmdb
from torchvision import transforms
import json
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
MAX_SIZE = 1e12


def load_and_resize(root, path, imscale,flag):
    try:
        if os.path.exists(os.path.join(root, path))==True:
            img = Image.open(os.path.join(root, path)).convert('RGB')
        else:
            img=Image.fromarray(np.zeros([400, 400, 3], np.uint8)+254).convert('RGB')
            flag=flag+1
    except:
        img=Image.fromarray(np.zeros([400, 400, 3], np.uint8)+254).convert('RGB')
        flag=flag+1

    transf_list = []
    transf_list.append(transforms.Resize(imscale))
    transf_list.append(transforms.CenterCrop(imscale))
    transform = transforms.Compose(transf_list)
    img = transform(img)

    return img,flag


def main(args):


    imname2pos = {}
    datasets = json.load(open(os.path.join(args.recipe_dir,  'dataset.json'), 'r'))

    parts= lmdb.open(os.path.join(args.save_dir, 'lmdb'), map_size=int(MAX_SIZE))
    with parts.begin() as txn:
        present_entries = [key for key, _ in txn.cursor()]
    j = 0
    impaths=[]
    flag=0
    count_all=0
    for i, recipe in tqdm(enumerate(datasets)):
        impaths.clear()
        count_all=count_all+len(recipe['images'])
        for k in range(len(recipe['images'])):
            impaths.append(recipe['images'][k][1])
        for n, p in enumerate(impaths):
            if p.encode() not in present_entries:
                im,flag = load_and_resize(args.recipe_dir, p, args.imscale,flag)
                im = np.array(im).astype(np.uint8)
                with parts.begin(write=True) as txn:
                    txn.put(p.encode(), im)
            imname2pos[p] = j
            j += 1
    print("数据集中的图片一共有" , count_all , '张')
    print("在数据集中地址存在但是缺失的图片一共有", flag, '张')
    pickle.dump(imname2pos, open(os.path.join(args.save_dir, 'imname2pos_2.pkl'), 'wb'))



def test(args):

    imname2pos = pickle.load(open(os.path.join(args.save_dir, 'imname2pos_1.pkl'), 'rb'))
    paths = imname2pos

    for k, v in paths.items():
        path = k
        break
    image_file = lmdb.open(os.path.join(args.save_dir, 'lmdb'), max_readers=1, readonly=True,
                           lock=False, readahead=False, meminit=False)
    with image_file.begin(write=False) as txn:
        image = txn.get(path.encode())
        image = np.fromstring(image, dtype=np.uint8)
        image = np.reshape(image, (args.imscale, args.imscale, 3))
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    print (np.shape(image))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe_dir', type=str, default='/DATACENTER/3/wjl/Recipe_generation_our/',
                        help='path to the recipe dataset')
    parser.add_argument('--save_dir', type=str, default='/DATACENTER/3/wjl/Recipe_generation_our/',
                        help='path where the lmdbs will be saved')
    parser.add_argument('--imscale', type=int, default=224,
                        help='size of images (will be rescaled and center cropped)')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly or center cropping images')
    parser.add_argument('--image_size', type=int, default=224, help='size to rescale images')
    parser.add_argument('--test_only', dest='test_only', action='store_true')
    parser.set_defaults(test_only=False)
    args = parser.parse_args()

    if not args.test_only:
        main(args)
    #test(args)
