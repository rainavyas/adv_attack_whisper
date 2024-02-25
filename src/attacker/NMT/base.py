import os
import json
from tqdm import tqdm

from src.attacker.base import BaseAttacker
from src.tools.tools import eval_bleu, eval_neg_seq_len

class NMTBaseAttacker(BaseAttacker):
    '''
    Base class for adversarial attacks on LLM evaluation systems
    '''
    def __init__(self, attack_args, model, src_lang='english', tgt_lang='french'):
        BaseAttacker.__init__(self, attack_args, model)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def eval_uni_attack(self, data, adv_phrase='', cache_dir=None, force_run=False, do_tqdm=False, perf=False, max_new_tokens=512):
        '''
            Generates predictions with adv_phrase (saves to cache)
            Computes the (negative average sequence length) = -1*mean(len(prediction))

            max_new_tokens limited when training attack
        '''

        if cache_dir is not None:
            # check for cache
            fpath = f'{cache_dir}/predictions.json'
            if os.path.isfile(fpath) and not force_run:
                with open(fpath, 'r') as f:
                    hyps = json.load(f)
                
                nsl = eval_neg_seq_len(hyps)
                return nsl

        hyps = []
        if do_tqdm:
            for sample in tqdm(data):
                prompt = self._prep_prompt(sample, adv_phrase=adv_phrase)
                hyp = self.model.predict(prompt, max_new_tokens=max_new_tokens).split('\n')[0] # often notes after translation to not be included
                hyps.append(hyp)
        else:
            for sample in data:
                prompt = self._prep_prompt(sample, adv_phrase=adv_phrase)
                hyp = self.model.predict(prompt, max_new_tokens=max_new_tokens).split('\n')[0]
                hyps.append(hyp)
        nsl = eval_neg_seq_len(hyps)

        if cache_dir is not None:
            with open(fpath, 'w') as f:
                json.dump(hyps, f)
        
        if perf:
            refs = [d['tgt_sentence'] for d in data]
            bleu_score = eval_bleu(hyps, refs)
            return {'nsl':nsl, 'bleu':bleu_score}
        return nsl
    
    def _prep_prompt(self, sample, adv_phrase=''):
        prompt = (
            f"Give only the {self.tgt_lang} translation of:\n\n"
            f"{sample['src_sentence']} {adv_phrase}\n\n"
            # f"Only give the translated {self.tgt_lang} sentence.\n"
        )
        return prompt