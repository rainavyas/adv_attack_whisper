from .base import BaseAttacker
from src.tools.tools import eval_wer
from src.tools.saving import next_dir

import os
import json
from tqdm import tqdm

class GreedyAttacker(BaseAttacker):
    def __init__(self, attack_args, model, word_list):
        BaseAttacker.__init__(self, attack_args, model)
        self.word_list = word_list

    def next_word_score(self, data, curr_adv_phrase, cache_path, array_job_id=-1):
        '''
            curr_adv_phrase: current universal adversarial phrase
            Returns the WER for each word in word list as next uni adv word
        '''
        # check for cache
        pos = len(curr_adv_phrase.split(' '))+1 if curr_adv_phrase != '' else 1
        path = next_dir(cache_path, f'pos{pos}')
        if array_job_id != -1:
            path = next_dir(path, f'array_job{array_job_id}')

        fpath_prev = f'{path}/prev.txt'
        fpath_scores = f'{path}/scores.txt'
        if os.path.isfile(fpath_prev):
            with open(fpath_prev, 'r') as f:
                prev = json.load(f)
            with open(fpath_scores, 'r') as f:
                word_2_score = json.load(f)

            return prev, word_2_score

        score_no_attack = self.trn_evaluate_uni_attack_seen(data, curr_adv_phrase)
        word_2_score = {}
        for word in tqdm(self.word_list):
            if curr_adv_phrase == '':
                adv_phrase = word + '.'
            else:
                adv_phrase = curr_adv_phrase + ' ' + word + '.'
            score = self.trn_evaluate_uni_attack_seen(data, adv_phrase)
            word_2_score[word] = score
        
        # cache
        with open(fpath_prev, 'w') as f:
            prev = {'prev-adv-phrase': curr_adv_phrase, 'score':score_no_attack}
            json.dump(prev, f)
        with open(fpath_scores, 'w') as f:
            json.dump(word_2_score, f)
        
        return prev, word_2_score

    
    def trn_evaluate_uni_attack(self, data, adv_phrase=''):
        '''
            Returns the WER across the dataset with adv attack
        '''
        return self.eval_uni_attack(data, adv_phrase=adv_phrase)


    @staticmethod
    def next_best_word(base_path, pos=1):
        '''
            base_path: directory with scores.txt and prev.txt (or array_job files)
            Give the next best word from output saved files
        '''

        def best_from_dict(word_2_score, pos=1):
            prev = [None, 0]
            best = [None, 0]

            for k,v in word_2_score.items():
                if v>best[1]:
                    prev[0] = best[0]
                    prev[1] = best[1]
                    best[0]=k
                    best[1]=v
                elif v>prev[1]:
                    prev[0]=k
                    prev[1]=v
            if pos==1:
                return best[0], best[1]
            elif pos==2:
                return prev[0], prev[1]
            else:
                print("Not supported pos")

        if os.path.isfile(f'{base_path}/scores.txt'):
            with open(f'{base_path}/scores.txt', 'r') as f:
                word_2_score = json.load(f)
            return best_from_dict(word_2_score, pos=pos)
        
        elif os.path.isdir(f'{base_path}/array_job2'):
            combined = {}
            for i in range(200):
                try:
                    with open(f'{base_path}/array_job{i}/scores.txt', 'r') as f:
                        word_2_score = json.load(f)
                except:
                    continue
                combined = {**combined, **word_2_score}
            
            return best_from_dict(combined, pos=pos)

        else:
            raise ValueError("No cached scores") 