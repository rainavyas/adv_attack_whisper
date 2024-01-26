import torch

class WhisperModel:
    '''
        Wrapper for Whisper ASR Transcription
    '''
    def __init__(self, model_name, device=torch.device('cpu')):
        self.model = #TODO
    
    def predict(self, audio='', decoder_text=''):
        '''
            Whisper decoder output here
        '''
        pass
