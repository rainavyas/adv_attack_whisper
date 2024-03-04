import torch
from whisper.audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)

class BaseAttacker():
    '''
        Base class for whitebox attack on Whisper Model in mel-vector space
    '''
    def __init__(self, attack_args, model):
        self.attack_args = attack_args
        self.whisper_model = model # assume it is a whisper model

    def audio_to_mel(self, audio, padding=False):
        '''
            Get sequence of mel-vectors
        '''
        if padding:
            mel = log_mel_spectrogram(audio, self.whisper_model.model.dims.n_mels, padding=N_SAMPLES)
        else:
            mel = log_mel_spectrogram(audio, self.whisper_model.model.dims.n_mels)
        return mel
    
    def eval_uni_attack(self, data, softprompt_model, cache_dir=None, force_run=False):
        '''
            Generates transcriptions with softprompt_model learnt softprompts (saves to cache)
            Computes the (negative average sequence length) = -1*mean(len(prediction))
        '''
        raise NotImplementedError