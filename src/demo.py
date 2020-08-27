import matplotlib.pyplot as plt
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import torch
import sys

sys.path.append('./model')
import torch.nn as nn
import numpy as np
import os
from args import get_parser
import pickle
from model_full import get_model_full
from model_fomer import  get_model_fomer
from torchvision import transforms
from utils.output_utils import prepare_output
from PIL import Image
import time
from data_loader import get_loader
from  data_loader import *
data_dir = '/DATACENTER/3/wjl/Recipe_generation_our/'
model_dir='/DATACENTER/3/wjl/Recipe_generation_our/recipe_generation/full/checkpoints/'
# code will run in gpu if available and if the flag is set to True, else it will run on cpu
use_gpu = True
device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'


ingrs_vocab = pickle.load(open(os.path.join(data_dir, 'recipe_vocab_ingrs.pkl'), 'rb'))
action_vocab = pickle.load(open(os.path.join(data_dir, 'recipe_vocab_action.pkl'), 'rb'))
insts_vocab = pickle.load(open(os.path.join(data_dir, 'recipe_vocab_toks.pkl'), 'rb'))
t = time.time()
import sys; sys.argv=['']; del sys
args = get_parser()
args.maxseqlen = 50


greedy = [True, False, False, False]
beam = [-1, -1, -1, -1]
temperature = 1.0
numgens = len(greedy)
datasets={}
data_loaders={}
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
output_dim = instrs_vocab_size
print (instrs_vocab_size, ingr_vocab_size,action_vocab_size)
model = get_model_full(args, ingr_vocab_size,action_vocab_size,instrs_vocab_size)
# Load the trained model parameters
model_path = os.path.join(model_dir, 'model_0_.ckpt')
model.load_state_dict(torch.load(model_path, map_location=map_loc))

model.to(device)
model.eval()

print ('loaded model')
print ("Elapsed time:", time.time() -t)
print(args)

for img_inputs, captions, ingr_gt,action_gt in data_loaders['test']:

    length_steps = img_inputs.size(1)
    img_inputs = img_inputs.to(device)
    captions = captions.to(device)
    ingr_gt = ingr_gt.to(device)
    action_gt = action_gt.to(device)
    for steps in range(length_steps):
        image_step = img_inputs[:, steps, :, :, :]
        caption_step = captions[:, steps, :]
        ingrs_step = ingr_gt[:, steps, :]
        action_step = action_gt[:, steps, :]
        true_caps = caption_step.clone()[:, 1:].contiguous()
        #new_img_PIL = transforms.ToPILImage()(image_step.cpu().squeeze(0)).convert('RGB')
        #new_img_PIL.show()  # 处理后的PIL图片

        with torch.no_grad():
            outputs = model.sample(image_step, true_ingrs=None)
            ingr_ids = outputs['ingr_ids'].cpu().numpy()
            action_ids = outputs['action_ids'].cpu().numpy()
            recipe_ids = outputs['recipe_ids'].cpu().numpy()
            outs, valid = prepare_output(recipe_ids[0], ingr_ids[0],action_ids[0], insts_vocab,ingrs_vocab, action_vocab)

            BOLD = '\033[1m'
            END = '\033[0m'

            print(BOLD + '\nInstructions:' + END)
            print('-' + ' '.join(outs['recipe']))

            print(BOLD + '\nIngredients:' + END)
            print(', '.join(outs['ingrs']))

            print(BOLD + '\nActions:' + END)
            print(', '.join(outs['action']))


    print('=' * 20)
    plt.axis('off')

    plt.close()
