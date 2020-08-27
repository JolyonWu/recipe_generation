# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

replace_dict = {' .': '.',
                ' ,': ',',
                ' ;': ';',
                ' :': ':',
                '( ': '(',
                ' )': ')',
               " '": "'"}


def get_recipe(ids, vocab):
    toks = []
    for id_ in ids:
        toks.append(vocab.return_word(id_))
    return toks


def get_ingrs(ids, ingr_vocab_list):
    gen_ingrs = []
    for ingr_idx in ids:
        ingr_name = ingr_vocab_list.return_word(ingr_idx)
        if ingr_name == '<pad>':
            break
        gen_ingrs.append(ingr_name)
    return gen_ingrs


def prettify(toks, replace_dict):
    toks = ' '.join(toks)
    sentence = toks.split('<end>')[0]
    sentence = sentence.strip()
    sentence = sentence.capitalize()
    for k, v in replace_dict.items():
        sentence = sentence.replace(k, v)
    return sentence


def colorized_list(ingrs, ingrs_gt, colorize=False):
    if colorize:
        colorized_list = []
        for word in ingrs:
            if word in ingrs_gt:
                word = '\033[1;30;42m ' + word + ' \x1b[0m'
            else:
                word = '\033[1;30;41m ' + word + ' \x1b[0m'
            colorized_list.append(word)
        return colorized_list
    else:
        return ingrs


def prepare_output(gen_instr, gen_ingrs,gen_action,  insts_vocab,ingrs_vocab, action_vocab):

    toks = get_recipe(gen_instr,  insts_vocab)
    is_valid = True
    reason = 'All ok.'
    try:
        cut = toks.index('<end>')
        toks_trunc = toks[0:cut]
    except:
        toks_trunc = toks
        is_valid = False
        reason = 'no eos found'

    # repetition score
    score = float(len(set(toks_trunc))) / float(len(toks_trunc))
    toks = prettify(toks, replace_dict)

    if gen_ingrs is not None:
        gen_ingrs = get_ingrs(gen_ingrs,ingrs_vocab)
    if gen_action is not None:
        gen_action = get_ingrs(gen_action,action_vocab)

    if score <= 0.3:
        reason = 'Diversity score.'
        is_valid = False


    valid = {'is_valid': is_valid, 'reason': reason, 'score': score}
    outs = {'recipe': toks, 'ingrs': gen_ingrs,'action':gen_action}

    return outs, valid
