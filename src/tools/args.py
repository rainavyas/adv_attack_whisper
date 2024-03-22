import argparse

def core_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--model_name', type=str, default='whisper-small', help='ASR model')
    commandLineParser.add_argument('--gpu_id', type=int, default=0, help='select specific gpu')
    commandLineParser.add_argument('--data_name', type=str, default='librispeech', help='dataset for exps; for flores: flores-english-french')
    commandLineParser.add_argument('--seed', type=int, default=1, help='select seed')
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    return commandLineParser.parse_known_args()

def attack_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    # train attack args
    commandLineParser.add_argument('--attack_method', type=str, default='greedy', choices=['greedy', 'greedy-nmt', 'mel-whitebox', 'audio-whitebox'], help='Adversarial attack approach for training')
    commandLineParser.add_argument('--prev_phrase', default='', type=str, help='previously learnt adv phrase for greedy approach')
    commandLineParser.add_argument('--array_job_id', type=int, default=-1, help='-1 means not to run as an array job')
    commandLineParser.add_argument('--array_word_size', type=int, default=400, help='number of words to test for each array job in greedy attack')
    commandLineParser.add_argument('--max_epochs', type=int, default=20, help='Training epochs for soft-prompt mel-space attack')
    commandLineParser.add_argument('--save_freq', type=int, default=1, help='Epoch frequency for saving adv mel vectors for soft-prompt mel-space attack')
    commandLineParser.add_argument('--bs', type=int, default=16, help='Batch size for soft-prompt mel-space attack')
    commandLineParser.add_argument('--clip_mel', action='store_true', help='Clip softprompt log mel vectors to be clip_mel_val at maximum')
    commandLineParser.add_argument('--clip_mel_val', type=float, default=0, help='Value (maximum) to clip the log mel vectors')




    # eval attack args
    commandLineParser.add_argument('--attack_phrase', type=str, default='', help='Specifc adversarial attack phrase to evaluate, e.g. fwhisper-tiny-greedy-librispeech')
    commandLineParser.add_argument('--attack_epoch', type=int, default=-1, help='Specify which training epoch of mel softprompt attack to evaluate; -1 means no attack')
    commandLineParser.add_argument('--k_scale', type=float, default=1, help='Scale adv mel vectors for whitebox attack down by k_scale')
    commandLineParser.add_argument('--num_greedy_phrase_words', type=int, default=-1, help='for greedy phrase select only first k words')
    commandLineParser.add_argument('--force_run', action='store_true', help='Do not load from cache')
    commandLineParser.add_argument('--not_none', action='store_true', help='Do not evaluate the none attack')
    commandLineParser.add_argument('--eval_train', action='store_true', help='Evaluate attack on the train split')

    # eval attack args for attack transferability
    commandLineParser.add_argument('--transfer', action='store_true', help='Indicate it is a transferability attack (across model or dataset) for mel whitebox attack')
    commandLineParser.add_argument('--softprompt_model_dir', type=str, default='', help='path to trained mel attack vectors to evaluate, e.g. experiments/librispeech/whisper-tiny/attack_train/mel-whitebox/softprompt_models')
    return commandLineParser.parse_known_args()

def analysis_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--compare_with_audio', action='store_true', help='Include a real audio file')
    return commandLineParser.parse_known_args()


