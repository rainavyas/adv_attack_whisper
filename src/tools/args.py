import argparse

def core_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--model_name', type=str, default='whisper-small', help='ASR model')
    commandLineParser.add_argument('--gpu_id', type=int, default=0, help='select specific gpu')
    commandLineParser.add_argument('--data_name', type=str, default='librispeech', help='dataset for exps')
    commandLineParser.add_argument('--seed', type=int, default=1, help='select seed')
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    return commandLineParser.parse_known_args()

def attack_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    # train attack args
    commandLineParser.add_argument('--attack_method', type=str, default='greedy', choices=['greedy'], help='Adversarial attack approach for training')
    commandLineParser.add_argument('--prev_phrase', default='', type=str, help='previously learnt adv phrase for greedy approach')
    commandLineParser.add_argument('--array_job_id', type=int, default=-1, help='-1 means not to run as an array job')
    commandLineParser.add_argument('--array_word_size', type=int, default=400, help='number of words to test for each array job in greedy attack')

    # eval attack args
    commandLineParser.add_argument('--attack_phrase', type=str, default='greedy-librispeech', help='Specifc adversarial attack phrase to evaluate')
    commandLineParser.add_argument('--num_greedy_phrase_words', type=int, default=-1, help='for greedy phrase select only first k words')
    commandLineParser.add_argument('--force_run', action='store_true', help='Do not load from cache')
    commandLineParser.add_argument('--not_none', action='store_true', help='Do not evaluate the none attack')
    commandLineParser.add_argument('--eval_train', action='store_true', help='Evaluate attack on the train split')
    return commandLineParser.parse_known_args()


