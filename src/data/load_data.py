import os
import json

LIBRISPEECH_DIR = '/home/vr313/rds/rds-altaslp-8YSp2LXTlkY/data/librispeech'


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
        # create random 100 samples later
        return select_samples(_librispeech('dev_other')), _librispeech('test_other')


def _librispeech(sub_dir):
    '''
        for clean audio, set `sub_dir' to dev_clean/test_clean as dev/test sets
        for noisy audio, set `sub_dir' to dev_other/test_other as dev/test sets
    '''
    audio_transcript_pair_list = []
    with open(f'{LIBRISPEECH_DIR}/{sub_dir}/audio_ref_pair_list', 'r') as fin:
        for line in fin:
            _, audio, ref = line.split(None, 2)
            ref = ref.rstrip('\n')

            # change audio path as per user
            audio = audio.replace('rm2114', 'vr313')

            sample = {
                    'audio': audio,
                    'ref': ref
                }
            audio_transcript_pair_list.append(sample) 
    return audio_transcript_pair_list

def select_samples(data, samples=80):
    # load randomly selected samples
    fpath = 'experiments/sample_inds.txt'
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