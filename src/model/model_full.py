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



def get_model_full(args, ingr_vocab_size,action_vocab_size, instrs_vocab_size):


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

    # build ingredients embedding
    encoder_ingrs = EncoderLabels(args.embed_size, ingr_vocab_size,
                                  args.dropout_encoder, scale_grad=False).to(device)

    encoder_action = EncoderLabels(args.embed_size, action_vocab_size,
                                  args.dropout_encoder, scale_grad=False).to(device)

    decoder_instr = DecoderTransformer(args.embed_size, instrs_vocab_size,
                                 dropout=args.dropout_decoder_r, seq_length=args.maxseqlen,
                                 num_instrs=args.maxnuminstrs,
                                 attention_nheads=args.n_att, num_layers=args.transf_layers,
                                 normalize_before=True,
                                 normalize_inputs=False,
                                 last_ln=False,
                                 scale_embed_grad=False)


    # ingredients loss
    label_loss = nn.BCELoss(reduce=False)
    # recipe loss
    criterion = MaskedCrossEntropyCriterion(ignore_index=[instrs_vocab_size - 1], reduce=False)

    model = IngredientModel( encoder_image,ingr_decoder,action_decoder,
                             encoder_ingrs,encoder_action,decoder_instr,
                             crit=label_loss,crit_recipe=criterion,
                             pad_value_ingrs=ingr_vocab_size-1,pad_value_action=action_vocab_size-1,label_smoothing=args.label_smoothing_ingr)

    return model


class IngredientModel(nn.Module):
    def __init__(self,  image_encoder,ingr_decoder,action_decoder,encoder_ingrs,encoder_action,decoder_instr,
                  crit=None,crit_recipe=None,  pad_value_ingrs=0, pad_value_action=0,  label_smoothing=0.0):

        super(IngredientModel, self).__init__()
        self.image_encoder = image_encoder
        self.ingredient_decoder = ingr_decoder
        self.action_decoder = action_decoder
        self.encoder_ingrs=encoder_ingrs
        self.encoder_action=encoder_action
        self.decoder_instr=decoder_instr
        self.crit = crit
        self.crit_recipe=crit_recipe
        self.pad_value_ingrs = pad_value_ingrs
        self.pad_value_action = pad_value_action
        self.label_smoothing = label_smoothing

    def forward(self, img_inputs, target_ingrs,target_action,target_caption,
                sample=False, keep_cnn_gradients=False):
        if sample:
            return self.sample(img_inputs, greedy=True)
        img_features = self.image_encoder(img_inputs, keep_cnn_gradients)
        losses = {}
        ingr_ids, ingr_logits = self.ingredient_decoder.sample(None, None, greedy=True,
                                                               temperature=1.0, img_features=img_features,
                                                               first_token_value=0, replacement=False)

        action_ids, action_logits = self.action_decoder.sample(None, None, greedy=True,
                                                               temperature=1.0, img_features=img_features,
                                                               first_token_value=0, replacement=False)


        #########################################################
        # encode ingredients and action
        target_ingr_feats = self.encoder_ingrs(ingr_ids)
        target_ingr_mask = mask_from_eos(ingr_ids, eos_value=0, mult_before=False)
        target_ingr_mask = target_ingr_mask.float().unsqueeze(1)

        target_action_feats = self.encoder_action(action_ids)
        target_action_mask = mask_from_eos(action_ids, eos_value=0, mult_before=False)
        target_action_mask = target_action_mask.float().unsqueeze(1)

        #将ingredient和aaction进行拼接
        feats=torch.cat((target_ingr_feats ,target_action_feats),2)
        mask=torch.cat((target_ingr_mask , target_action_mask),2)

        outputs, ids = self.decoder_instr(feats,mask,target_caption, img_features)

        outputs = outputs[:, :-1, :].contiguous()
        x_axis = outputs.size(0)
        y_axis = outputs.size(1)
        outputs = outputs.view(x_axis * y_axis, -1)

        targets = target_caption[:, 1:]
        targets = targets.contiguous().view(-1)
        loss = self.crit_recipe(outputs,targets)

        losses['recipe_loss'] = loss


        return losses

    def sample(self, img_inputs,  greedy=True, temperature=1.0, beam=-1.0, true_ingrs=None):

        outputs = dict()
        img_features = self.image_encoder(img_inputs)
        #这个部分是ingredient部分
        ingr_ids, ingr_probs = self.ingredient_decoder.sample(None, None, greedy=True, temperature=1.0,
                                                              beam=-1,
                                                              img_features=img_features, first_token_value=0,
                                                              replacement=False)
        sample_mask = mask_from_eos(ingr_ids, eos_value=0, mult_before=False)
        ingr_ids[sample_mask == 0] = self.pad_value_ingrs
        outputs['ingr_ids'] = ingr_ids
        outputs['ingr_probs'] = ingr_probs.data

        #这个部分是action部分
        action_ids, action_probs = self.action_decoder.sample(None, None, greedy=True, temperature=temperature,
                                                              beam=-1,
                                                              img_features=img_features, first_token_value=0,
                                                              replacement=False)
        sample_mask = mask_from_eos(action_ids, eos_value=0, mult_before=False)
        action_ids[sample_mask == 0] = self.pad_value_action
        outputs['action_ids'] = action_ids
        outputs['action_probs'] = action_probs.data

        #############################
        #接下来是recipe部分
        ingr_mask = mask_from_eos(ingr_ids, eos_value=0, mult_before=False)
        ingr_ids[ingr_mask == 0] = self.pad_value_ingrs

        ingr_mask = ingr_mask.float().unsqueeze(1)
        ingr_feats = self.encoder_ingrs(ingr_ids)

        action_mask = mask_from_eos(action_ids, eos_value=0, mult_before=False)
        action_ids[action_mask == 0] = self.pad_value_action

        action_mask = action_mask.float().unsqueeze(1)
        action_feats = self.encoder_action(action_ids)

        # 将ingredient和aaction进行拼接
        feats = torch.cat((ingr_feats,action_feats), 2)
        mask = torch.cat((ingr_mask, action_mask), 2)

        ids, probs = self.decoder_instr.sample(feats, mask, greedy, temperature, beam, img_features, first_token_value= 0,
                                                last_token_value=1)

        outputs['recipe_probs'] = probs.data
        outputs['recipe_ids'] = ids

        return outputs

