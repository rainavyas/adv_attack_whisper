from .base import BaseAttacker
from .greedy import GreedyAttacker

def select_eval_attacker(attack_args, core_args, model):
    return BaseAttacker(attack_args, model)

def select_train_attacker(attack_args, core_args, model, word_list=None):
    return GreedyAttacker(attack_args, model, word_list)

