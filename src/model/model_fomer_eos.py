# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import sys
sys.path.append('../')

import torch
import torch.nn as nn
from modules.encoder import EncoderCNN, EncoderLabels
from modules.transformer_decoder import DecoderTransformer
from utils.metrics import softIoU, MaskedCrossEntropyCriterion
from utils.utils_all import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#用在train函数中构造整个模型
def get_model_fomer(args, ingr_vocab_size,action_vocab_size):


    # build image model
    encoder_image = EncoderCNN(args.embed_size, args.dropout_encoder, args.image_model)
    ingr_decoder = DecoderTransformer(args.embed_size, ingr_vocab_size, dropout=args.dropout_decoder_i,
                                      seq_length=args.maxnumlabels,
                                      num_instrs=1, attention_nheads=args.n_att_ingrs,
                                      pos_embeddings=False,
                                      num_layers=args.transf_layers_ingrs,
                                      learned=False,
                                      normalize_before=True,
                                      normalize_inputs=True,
                                      last_ln=True,
                                      scale_embed_grad=False)
    action_decoder = DecoderTransformer(args.embed_size, action_vocab_size, dropout=args.dropout_decoder_i,
                                      seq_length=args.maxnumactions,
                                      num_instrs=1, attention_nheads=args.n_att_ingrs,
                                      pos_embeddings=False,
                                      num_layers=args.transf_layers_ingrs,
                                      learned=False,
                                      normalize_before=True,
                                      normalize_inputs=True,
                                      last_ln=True,
                                      scale_embed_grad=False)

    # ingredients loss
    label_loss = nn.BCELoss(reduce=False)

    model = IngredientModel( encoder_image,ingr_decoder,action_decoder, crit=label_loss,
                             pad_value_ingrs=ingr_vocab_size-1,pad_value_action=action_vocab_size-1,label_smoothing=args.label_smoothing_ingr)

    return model

#模型
class IngredientModel(nn.Module):
    def __init__(self,  image_encoder,ingr_decoder,action_decoder,
                  crit=None,  pad_value_ingrs=0, pad_value_action=0,  label_smoothing=0.0):

        super(IngredientModel, self).__init__()
        self.image_encoder = image_encoder
        self.ingredient_decoder = ingr_decoder
        self.action_decoder = action_decoder
        self.crit = crit
        self.pad_value_ingrs = pad_value_ingrs
        self.pad_value_action = pad_value_action
        self.label_smoothing = label_smoothing

    def forward(self, img_inputs, target_ingrs,target_action,
                sample=False, keep_cnn_gradients=False):
        if sample:
            return self.sample(img_inputs, greedy=True)
        img_features = self.image_encoder(img_inputs, keep_cnn_gradients)

        losses = {}

        #######################################################################
        #这一部分是ingredient生成one-hot
        target_one_hot_ingrs = label2onehot(target_ingrs, self.pad_value_ingrs)
        target_one_hot_smooth_ingrs = label2onehot(target_ingrs, self.pad_value_ingrs)
        #label_smooth
        target_one_hot_smooth_ingrs[target_one_hot_smooth_ingrs == 1] = (1-self.label_smoothing)
        target_one_hot_smooth_ingrs[target_one_hot_smooth_ingrs == 0] = self.label_smoothing / target_one_hot_smooth_ingrs.size(-1)
        ingr_ids, ingr_logits = self.ingredient_decoder.sample(None, None, greedy=True,
                                                               temperature=1.0, img_features=img_features,
                                                               first_token_value=0, replacement=False)
        ingr_logits = torch.nn.functional.softmax(ingr_logits, dim=-1)
        ############################
        #这一部分是ingredient_eos_loss的计算
        ingr_eos = ingr_logits[:, :, 0]
        target_ingr_eos = ((target_ingrs == 0) ^ (target_ingrs == self.pad_value_ingrs))
        target_ingr_eos=target_ingr_eos.float()
        ingr_eos_loss=self.crit(ingr_eos,target_ingr_eos)
        ingr_eos_loss = torch.mean(ingr_eos_loss, dim=-1)
        losses['ingr_eos_loss']=ingr_eos_loss
        #########################################
        # 这一部分是ingredient_loss的计算
        mask_perminv_ingrs = mask_from_eos(target_ingrs, eos_value=0, mult_before=False)
        ingr_probs = ingr_logits * mask_perminv_ingrs.float().unsqueeze(-1)
        ingr_probs, _ = torch.max(ingr_probs, dim=1)
        ingr_ids[mask_perminv_ingrs == 0] = self.pad_value_ingrs
        ingr_loss = self.crit(ingr_probs, target_one_hot_ingrs)
        ingr_loss = torch.mean(ingr_loss, dim=-1)
        losses['ingr_loss'] = ingr_loss

        # iou
        pred_one_hot_ingrs = label2onehot(ingr_ids, self.pad_value_ingrs)
        losses['ingr_iou'] = softIoU(pred_one_hot_ingrs, target_one_hot_ingrs)

        ################################################################################
        # #这一部分是action生成one-hot
        target_one_hot_action = label2onehot(target_action, self.pad_value_action)
        target_one_hot_smooth_action = label2onehot(target_action, self.pad_value_action)
        target_one_hot_smooth_action[target_one_hot_smooth_action == 1] = (1 - self.label_smoothing)
        target_one_hot_smooth_action[target_one_hot_smooth_action == 0] = self.label_smoothing / target_one_hot_smooth_action.size(-1)
        action_ids, action_logits = self.action_decoder.sample(None, None, greedy=True,
                                                               temperature=1.0, img_features=img_features,
                                                               first_token_value=0, replacement=False)
        action_logits = torch.nn.functional.softmax(action_logits, dim=-1)

        ############################
        # 这一部分是action_eos_loss的计算
        action_eos = action_logits[:, :, 0]
        target_action_eos = ((target_action == 0) ^ (target_action == self.pad_value_action))
        target_action_eos=target_action_eos.float()
        action_eos_loss = self.crit(action_eos, target_action_eos)
        action_eos_loss = torch.mean(action_eos_loss, dim=-1)
        losses['action_eos_loss']= action_eos_loss
        ########################################
        # 这一部分是ingredient_loss的计算
        mask_perminv_action = mask_from_eos(target_action, eos_value=0, mult_before=False)
        action_probs = action_logits * mask_perminv_action.float().unsqueeze(-1)
        action_probs, _ = torch.max(action_probs, dim=1)
        action_ids[mask_perminv_action == 0] = self.pad_value_action
        action_loss = self.crit(action_probs, target_one_hot_action)
        action_loss = torch.mean(action_loss, dim=-1)
        losses['action_loss'] = action_loss
        # iou
        pred_one_hot_action = label2onehot(action_ids, self.pad_value_action)
        losses['action_iou'] = softIoU(pred_one_hot_action, target_one_hot_action)

        return losses

    def sample(self, img_inputs, temperature=1.0):

        outputs = dict()

        img_features = self.image_encoder(img_inputs)
        ####################################
        #这个部分是ingredient部分
        ingr_ids, ingr_probs = self.ingredient_decoder.sample(None, None, greedy=True, temperature=temperature,
                                                              beam=-1,
                                                              img_features=img_features, first_token_value=0,
                                                              replacement=False)
        sample_mask = mask_from_eos(ingr_ids, eos_value=0, mult_before=False)
        ingr_ids[sample_mask == 0] = self.pad_value_ingrs
        outputs['ingr_ids'] = ingr_ids
        outputs['ingr_probs'] = ingr_probs.data
        ###############################
        #这个部分是action部分
        action_ids, action_probs = self.action_decoder.sample(None, None, greedy=True, temperature=temperature,
                                                              beam=-1,
                                                              img_features=img_features, first_token_value=0,
                                                              replacement=False)
        sample_mask = mask_from_eos(action_ids, eos_value=0, mult_before=False)
        action_ids[sample_mask == 0] = self.pad_value_action
        outputs['action_ids'] = action_ids
        outputs['action_probs'] = action_probs.data

        return outputs

