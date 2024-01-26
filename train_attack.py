'''
    Train adversarial phrase
'''

import random
import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import json

from src.tools.args import core_args, attack_args
from src.data.load_data import load_data
from src.models.load_model import load_model
from src.tools.tools import get_default_device, set_seeds
from src.attacker.selector import select_train_attacker
from src.tools.saving import base_path_creator, attack_base_path_creator_train

if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()

    # set seeds
    set_seeds(core_args.seed)
    base_path = base_path_creator(core_args)
    attack_base_path = attack_base_path_creator_train(attack_args, base_path)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if core_args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device(core_args.gpu_id)
    print(device)

    # load training data
    data, _ = load_data(core_args)

    # load model
    model = load_model(core_args, device=device)

    # load the vocab
    fpath = 'experiments/words.txt'
    if os.path.isfile(fpath):
        with open(fpath, 'r') as f:
            word_list = json.load(f)
    else:
        import nltk
        nltk.download('words')
        from nltk.corpus import words
        word_list = words.words()
        word_list = list(set(word_list))[:20000]

        with open(fpath, 'w') as f:
            json.dump(word_list, f)
    
    # select vocab segment if array job
    if attack_args.array_job_id != -1:
        start = attack_args.array_job_id*attack_args.array_word_size
        end = start+attack_args.array_word_size
        word_list = word_list[start:end]

    # save scores for each word as the next word in the uni adv phrase
    attacker = select_train_attacker(attack_args, core_args, model, word_list=word_list)
    prev, word_2_score = attacker.next_word_score(data, attack_args.prev_phrase, attack_base_path, array_job_id=attack_args.array_job_id)
