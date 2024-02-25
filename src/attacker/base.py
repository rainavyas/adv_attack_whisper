import os
import json
from tqdm import tqdm

from src.tools.tools import eval_wer

class BaseAttacker():
    '''
    Base class for adversarial attacks on LLM evaluation systems
    '''
    def __init__(self, attack_args, model):
        self.attack_args = attack_args
        self.model = model
        self.adv_phrase = self._load_phrase(self.attack_args.attack_phrase)
    
    def _load_phrase(self, phrase_name):
        if phrase_name=='fwhisper-tiny-greedy-librispeech':
            phrase = 'aonach'
            return ' '.join(phrase.split()[:self.attack_args.num_greedy_phrase_words])
        if phrase_name=='fwhisper-tiny-greedy-librispeech-spelt':
            phrase = 'A O N A C H'
            return ' '.join(phrase.split()[:self.attack_args.num_greedy_phrase_words])
        if phrase_name=='fwhisper-tiny-greedy3-librispeech':
            phrase = 'tocologist'
            return ' '.join(phrase.split()[:self.attack_args.num_greedy_phrase_words])
        if phrase_name=='fwhisper-tiny-greedy2k-librispeech':
            phrase = 'luctiferous'
            return ' '.join(phrase.split()[:self.attack_args.num_greedy_phrase_words])

    def eval_uni_attack(self, data, adv_phrase='', cache_dir=None, force_run=False, do_tqdm=False, **kwargs):
        '''
            Generates predictions with adv_phrase (saves to cache)
            Computes the WER against the references
        '''
        refs = [d['ref'] for d in data]

        if cache_dir is not None:
            # check for cache
            fpath = f'{cache_dir}/predictions.json'
            if os.path.isfile(fpath) and not force_run:
                with open(fpath, 'r') as f:
                    hyps = json.load(f)
                wer = eval_wer(hyps, refs)
                return wer

        hyps = []
        if do_tqdm:
            for sample in tqdm(data):
                hyp = self.model.predict(audio=sample['audio'], decoder_text=adv_phrase)
                hyps.append(hyp)
        else:
            for sample in data:
                hyp = self.model.predict(audio=sample['audio'], decoder_text=adv_phrase)
                hyps.append(hyp)
        wer = eval_wer(hyps, refs)

        if cache_dir is not None:
            with open(fpath, 'w') as f:
                json.dump(hyps, f)
        
        return wer
    
    
    
