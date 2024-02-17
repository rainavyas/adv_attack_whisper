from .base import BaseAttacker
from src.tools.tools import eval_wer
from src.tools.saving import next_dir
from src.attacker.greedy import GreedyAttacker
from .base import NMTBaseAttacker

import os
import json
from tqdm import tqdm

class NMTGreedyAttacker(NMTBaseAttacker, GreedyAttacker):
    def __init__(self, attack_args, model, word_list):
        NMTBaseAttacker.__init__(self, attack_args, model)
        self.word_list = word_list