from .base import BaseAttacker
from src.tools.tools import eval_wer
from src.tools.saving import next_dir
from src.attacker.greedy import GreedyAttacker
from .base import NMTBaseAttacker

import os
import json
from tqdm import tqdm

class NMTGreedyAttacker(NMTBaseAttacker, GreedyAttacker):
    def __init__(self, attack_args, model, word_list, src_lang='english', tgt_lang='french'):
        NMTBaseAttacker.__init__(self, attack_args, model, src_lang=src_lang, tgt_lang=tgt_lang)
        self.word_list = word_list
        self.max_new_tokens = 8 # this only applies to mistral / llama models

    def trn_evaluate_uni_attack(self, data, adv_phrase=''):
        '''
            Returns the nsl across the dataset with adv attack
        '''
        return self.eval_uni_attack(data, adv_phrase=adv_phrase, max_new_tokens=self.max_new_tokens)