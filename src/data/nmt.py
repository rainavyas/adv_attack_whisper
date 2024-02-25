'''
    NMT Dataset
'''

from datasets import load_dataset

language_mapper = {
    'english'   :   'eng_Latn',
    'french'    :   'fra_Latn'
}

def load_flores(src='english', tgt='french'):
    src_dataset = load_dataset("facebook/flores", language_mapper[src])
    src_dev = src_dataset['dev']
    src_test = src_dataset['devtest']

    tgt_dataset = load_dataset("facebook/flores", language_mapper[tgt])
    tgt_dev = tgt_dataset['dev']
    tgt_test = tgt_dataset['devtest']

    # merge
    def combine(data_src, data_tgt):
        combined = []
        for s,t in zip(data_src, data_tgt):
            sample = {
                'id'    :   s['id'],
                'src_sentence'  :   s['sentence'],
                'tgt_sentence'  :   t['sentence']
            }
            combined.append(sample)
        return combined
    
    return combine(src_dev, tgt_dev), combine(src_test, tgt_test)
