

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


def _librispeech():
    pass
    #TODO