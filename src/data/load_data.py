import os
import json
import ffmpeg
from tqdm import tqdm

from .nmt import load_flores
from .speech import _librispeech


def load_data(core_args, attack_method='greedy'):
    '''
        Return data as train_data, test_data
        Each data is a list (over data samples), where each sample is a dictionary
            sample = {
                        'audio':    <path to utterance audio file>,
                        'ref':      <Reference transcription>,
                    }
    '''
    if core_args.data_name == 'librispeech':

        #blackbox attack
        if 'greedy' in attack_method:
            # 80 random samples used for training purposes
            return select_samples(_librispeech('dev_other')), _librispeech('test_other')
        
        
        # whitebox attack mel space
        elif attack_method=='mel-whitebox':
            # filter to only keep audio segments less than 29s
            return time_filtered_samples(_librispeech('dev_other'), max_audio=29, part='train', cache_dir='experiments/samples/librispeech'), time_filtered_samples(_librispeech('test_other'), max_audio=29, part='test', cache_dir='experiments/samples/librispeech')

        # whitebox attack audio space
        elif attack_method=='whitebox-audio':
            return _librispeech('dev_other'), _librispeech('test_other')

    
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



def get_audio_duration(fpath):
    # returns duration in seconds
    probe = ffmpeg.probe(fpath)
    stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    return float(stream['duration'])


def time_filtered_samples(samples, max_audio=29, part='train', cache_dir='experiments'):
    # Filter to only keep audio segments less than 29seconds
    fpath = f'{cache_dir}/{part}_time_inds.txt'
    if os.path.isfile(fpath):
        with open(fpath, 'r') as f:
            inds = json.load(f)
        return [samples[i] for i in inds]
    else:
        new_samples = []
        inds = []
        print(f"Loading audio files shorter than {max_audio}s")
        for i, sample in tqdm(enumerate(samples), total=len(samples)):
            duration = get_audio_duration(sample['audio'])
            if duration < max_audio:
                new_samples.append(sample)
                inds.append(i)
        with open(fpath, 'w') as f:
            json.dump(inds, f)
        return new_samples
