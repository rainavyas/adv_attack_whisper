import torch
import whisper
import editdistance

# from whisper.normalizers import EnglishTextNormalizer

CACHE_DIR = '/home/vr313/rds/rds-altaslp-8YSp2LXTlkY/experiments/rm2114/.cache'

MODEL_NAME_MAPPER = {
    'whisper-tiny'  : 'tiny.en',
    'whisper-base'  : 'base.en',
    'whisper-small' : 'small.en',
    'whisper-medium'  : 'medium.en',
    'whisper-large'  : 'large',
}

class WhisperModel:
    '''
        Wrapper for Whisper ASR Transcription
    '''
    def __init__(self, model_name='whisper-small', device=torch.device('cpu')):
        self.model = whisper.load_model(MODEL_NAME_MAPPER[model_name], device=device, download_root=CACHE_DIR)
        # self.std = EnglishTextNormalizer()
    
    def predict(self, audio='', decoder_text=''):
        '''
            Whisper decoder output here
        '''
        result = self.model.transcribe(audio, language='en', initial_prompt=decoder_text, verbose=False)
        segments = []
        for segment in result['segments']:
            segments.append(segment['text'].strip())
            # print(segment['start'], segment['end'], segment['text'])
        return ' '.join(segments)

    # def calc_wer(self, hyp_list, ref_list):
    #     '''
    #         Calculate WER on the entire set, apply standard text normalisation before calculation
    #     '''
    #     errors, refs = 0, 0
    #     for hyp, ref in hyp_list, ref_list:
    #         hyp_tn, ref_tn = self.std(hyp), self.std(ref)
    #         errors += editdistance.eval(hyp_tn.split(), ref_tn.split())
    #         refs += len(ref_tn.split())
    #     return errors / refs
