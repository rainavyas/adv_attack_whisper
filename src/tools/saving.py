import os

def base_path_creator(core_args, create=True):
    path = '.'
    path = next_dir(path, 'experiments', create=create)
    path = next_dir(path, core_args.data_name, create=create)
    path = next_dir(path, core_args.model_name, create=create)
    return path

def attack_base_path_creator_train(attack_args, path='.', create=True):
    path = next_dir(path, 'attack_train', create=create)
    path = next_dir(path, attack_args.attack_method, create=create)
    return path

def attack_base_path_creator_eval(attack_args, path='.', create=True):
    path = next_dir(path, 'attack_eval', create=create)
    path = next_dir(path, attack_args.attack_phrase, create=create)
    if 'greedy' in attack_args.attack_phrase:
        path =  next_dir(path, f'num_words-{attack_args.num_greedy_phrase_words}', create=create)
    return path


def next_dir(path, dir_name, create=True):
    if not os.path.isdir(f'{path}/{dir_name}'):
        try:
            if create:
                os.mkdir(f'{path}/{dir_name}')
            else:
                raise ValueError ("provided args do not give a valid model path")
        except:
            # path has already been created in parallel
            pass
    path += f'/{dir_name}'
    return path