import torch
from faster_whisper import WhisperModel


CACHE_DIR = '/home/vr313/rds/rds-altaslp-8YSp2LXTlkY/experiments/rm2114/.cache'

MODEL_NAME_MAPPER = {
    'fast-whisper-tiny'  : 'tiny.en',
    'fast-whisper-base'  : 'base.en',
    'fast-whisper-small' : 'small.en',
    'fast-whisper-medium'  : 'medium.en',
    'fast-whisper-large'  : 'large',
}

class FastWhisperModel:
    '''
        Wrapper for Whisper ASR Transcription
    '''
    def __init__(self, model_name='whisper-small', device=torch.device('cpu')):
        self.model = WhisperModel(MODEL_NAME_MAPPER[model_name], device='cuda', compute_type="float16", download_root=CACHE_DIR)
    
    def predict(self, audio='', decoder_text=''):
        '''
            Whisper decoder output here
        '''
        # decode_options = {'beam_size':None}
        if decoder_text=='':
            decoder_text = None
        segments, _ = self.model.transcribe(audio, language='en', initial_prompt=decoder_text, beam_size=1)
        text_segments = []
        for segment in list(segments):
            text = segment.text
            text_segments.append(text.strip())
        return ' '.join(text_segments)

