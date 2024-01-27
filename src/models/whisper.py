import torch
import whisper
import editdistance

from whisper.normalizers import EnglishTextNormalizer

CACHE_DIR = '~/rds/rds-altaslp-8YSp2LXTlkY/experiments/rm2114/.cache'

class WhisperModel:
    '''
        Wrapper for Whisper ASR Transcription
    '''
    def __init__(self, model_name='small.en', device=torch.device('cpu')):
        self.model = whisper.load_model(model_name, device=device, download_root=CACHE_DIR)
        self.std = EnglishTextNormalizer()
    
    def predict(self, audio='', decoder_text=''):
        '''
            Whisper decoder output here
        '''
        result = self.model.transcribe(audio, language='en', initial_prompt=decoder_text)
        segments = []
        for segment in result['segments']:
            segments.append(segment['text'].strip())
            print(segment['start'], segment['end'], segment['text'])
        return ' '.join(segments)

    def calc_wer(self, hyp_list, ref_list):
        '''
            Calculate WER on the entire set, apply standard text normalisation before calculation
        '''
        errors, refs = 0, 0
        for hyp, ref in hyp_list, ref_list:
            hyp_tn, ref_tn = self.std(hyp), self.std(ref)
            errors += editdistance.eval(hyp_tn.split(), ref_tn.split())
            refs += len(ref_tn.split())
        return errors / refs
