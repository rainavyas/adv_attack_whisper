from .whisper import WhisperModel
from .fast_whisper import FastWhisperModel
from .text_decoder import HFBase, GemmaModel

def load_model(core_args, device=None):
    if 'fast-whisper' in core_args.model_name:
        return FastWhisperModel(core_args.model_name, device=device)
    elif 'whisper' in core_args.model_name:
        return WhisperModel(core_args.model_name, device=device, task=core_args.task, language=core_args.language)
    elif 'mistral' in core_args.model_name:
        return HFBase(core_args.model_name, device)
    elif 'gemma' in core_args.model_name:
        return GemmaModel(core_args.model_name, device)