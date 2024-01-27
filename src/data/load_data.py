LIBRISPEECH_DIR = '~/rds/rds-altaslp-8YSp2LXTlkY/data/librispeech'


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
        return _librispeech()


def _librispeech(sub_dir):
    '''
        for clean audio, set `sub_dir' to dev_clean/test_clean as dev/test sets
        for noisy audio, set `sub_dir' to dev_other/test_other as dev/test sets
    '''
    audio_transcript_pair_list = []
    with open(f'{LIBRISPEECH_DIR}/{sub_dir}/audio_ref_pair_list') as fin:
        for line in fin:
            _, audio, ref = line.split(None, 2)
            sample = {
                    'audio': audio,
                    'ref': ref
                }
            audio_transcript_pair_list.append(sample)
                
    return audio_transcript_pair_list
