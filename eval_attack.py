'''
    Evaluate attack
'''

import sys
import os
import torch
import numpy as np

from src.tools.tools import get_default_device, set_seeds
from src.tools.args import core_args, attack_args
from src.tools.saving import base_path_creator, attack_base_path_creator_eval, attack_base_path_creator_train
from src.data.load_data import load_data
from src.models.load_model import load_model
from src.attacker.selector import select_eval_attacker

if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()

    print(core_args)
    print(attack_args)
    
    set_seeds(core_args.seed)
    if not attack_args.transfer:
        base_path = base_path_creator(core_args)
        attack_base_path = attack_base_path_creator_eval(attack_args, base_path)
    else:
        base_path = None
        attack_base_path = None

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if core_args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device(core_args.gpu_id)
    print(device)

    # Load the data
    train_data, test_data = load_data(core_args)
    if attack_args.eval_train:
        test_data = train_data

    # Load the model
    model = load_model(core_args, device=device)

    # load attacker for evaluation
    attacker = select_eval_attacker(attack_args, core_args, model, device=device)

    # evaluate

    if attack_args.attack_method == 'mel-whitebox':
        # Mel softprompt attack

        if not attack_args.transfer:
            softprompt_model_dir = f'{attack_base_path_creator_train(attack_args, base_path)}/softprompt_models'
        else:
            softprompt_model_dir = attack_args.softprompt_model_dir

        # 1) No attack
        if not attack_args.not_none:
            print('No attack')
            result = attacker.eval_uni_attack(test_data, softprompt_model_dir=softprompt_model_dir, attack_epoch=-1, cache_dir=attack_base_path, force_run=attack_args.force_run)
            print(result)
            print()

        # 2) Attack
        print('Attack')
        result = attacker.eval_uni_attack(test_data, softprompt_model_dir=softprompt_model_dir, attack_epoch=attack_args.attack_epoch, k_scale=attack_args.k_scale, cache_dir=attack_base_path, force_run=attack_args.force_run)
        print(result)
        print()
    
    else:
        # greedy word-based blackbox attack

        # 1) No attack
        if not attack_args.not_none:
            print('No attack')
            result = attacker.eval_uni_attack(test_data, adv_phrase='', cache_dir=base_path, force_run=attack_args.force_run, do_tqdm=True, perf=True)
            print(result)
            print()

        # 2) Attack
        print('Attack')
        result = attacker.eval_uni_attack(test_data, adv_phrase=attacker.adv_phrase, cache_dir=attack_base_path, force_run=attack_args.force_run, do_tqdm=True, perf=True)
        print(result)
        print()

    

