import os
import json

from .nmt import load_flores
from .speech import _librispeech


def load_data(core_args):
    '''
        Return data as train_data, test_data
        Each data is a list (over data samples), where each sample is a dictionary
            sample = {
                        'audio':    <path to utterance audio file>,
                        'ref':      <Reference transcription>,
                    }
    '''
    if core_args.data_name == 'librispeech':
        # 80 random samples used for training purposes
        return select_samples(_librispeech('dev_other')), _librispeech('test_other')
    
    elif 'flores' in core_args.data_name:
        dname = core_args.data_name
        parts = dname.split('-')
        dev, test = load_flores(parts[1], parts[2])
        return select_samples(dev, cache_dir='experiments/samples/flores'), test


def select_samples(data, samples=80, cache_dir='experiments'):
    # load randomly selected samples
    fpath = f'{cache_dir}/sample_inds.txt'
    if os.path.isfile(fpath):
        with open(fpath, 'r') as f:
            inds = json.load(f)
    else:
        import random
        inds = random.sample(range(len(data)), samples)
        with open(fpath, 'w') as f:
            json.dump(inds, f)
    sampled_data = [data[ind] for ind in inds]
    return sampled_data