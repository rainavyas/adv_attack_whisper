from .base import BaseAttacker
from .greedy import GreedyAttacker
from .NMT.base import NMTBaseAttacker
from .NMT.greedy import NMTGreedyAttacker
from .whitebox.soft_prompt import SoftPromptAttack

def select_eval_attacker(attack_args, core_args, model):
    if attack_args.attack_method == 'greedy':
        # Blackbox ASR atttack
        return BaseAttacker(attack_args, model)
    elif attack_args.attack_method == 'greedy-nmt':
        # Blackbox NMT atttack
        dname = core_args.data_name
        parts = dname.split('-')
        return NMTBaseAttacker(attack_args, model, src_lang=parts[1], tgt_lang=parts[2])


def select_train_attacker(attack_args, core_args, model, word_list=None, device=None):
    if attack_args.attack_method == 'greedy':
        # Blackbox ASR greedy atttack
        return GreedyAttacker(attack_args, model, word_list)
    elif attack_args.attack_method == 'mel-whitebox':
        return SoftPromptAttack(attack_args, model, device)
    elif attack_args.attack_method == 'greedy-nmt':
        # Blackbox NMT atttack
        dname = core_args.data_name
        parts = dname.split('-')
        return NMTGreedyAttacker(attack_args, model, word_list, src_lang=parts[1], tgt_lang=parts[2])
