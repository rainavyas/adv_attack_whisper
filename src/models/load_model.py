from .whisper import WhisperModel
from .fast_whisper import FastWhisperModel

def load_model(core_args, device=None):
    if 'fast-whisper' in core_args.model_name:
        return FastWhisperModel(core_args.model_name, device=device)
    if 'whisper' in core_args.model_name:
        return WhisperModel(core_args.model_name, device=device)