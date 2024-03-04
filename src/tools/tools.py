from whisper.normalizers import EnglishTextNormalizer
import editdistance
import torch
import random
from nltk.translate.bleu_score import sentence_bleu

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)

def get_default_device(gpu_id=0):
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device(f'cuda:{gpu_id}')
    else:
        print("No CUDA found")
        return torch.device('cpu')

def eval_wer(hyps, refs):
    # assuming the texts are already aligned
    # WER
    std = EnglishTextNormalizer()
    errors = 0
    crefs = 0
    for hyp, ref, in zip(hyps, refs):
        a = std(' '.join(hyp.split()[1:]))
        b = std(' '.join(ref.split()[1:]))
        errors += editdistance.eval(a.split(), b.split())
        crefs += len(b.split())
    return errors/crefs

def eval_neg_seq_len(hyps):
    '''
        Average sequence length (negative)
    '''
    nlens = 0
    for hyp in hyps:
        nlens += (len(hyp.split()))
    return (-1)*nlens/len(hyps)

def eval_bleu(hyps, refs):
    '''
        BLEU score
    '''
    scores = 0
    for hyp, ref in zip(hyps, refs):
        hyp = hyp.rstrip('\n')
        ref = ref.rstrip('\n')
        scores += sentence_bleu([ref.split()], hyp.split())
    return scores/len(hyps)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
