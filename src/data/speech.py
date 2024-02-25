LIBRISPEECH_DIR = '/home/vr313/rds/rds-altaslp-8YSp2LXTlkY/data/librispeech'

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