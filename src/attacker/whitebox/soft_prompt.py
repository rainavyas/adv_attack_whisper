import torch
import torch.nn as nn
import random
import os
from tqdm import tqdm

from .base import BaseAttacker
from src.tools.tools import set_seeds
from whisper.tokenizer import get_tokenizer

class SoftPromptAttack(BaseAttacker):
    '''
        Soft-prompting style adversarial attack in mel-vector space

        Learn a short sequence of mel-vectors pre-pended to the input-audio mel-vectors to maximise the probability of the end-of-transcript token
    '''
    def __init__(self, attack_args, whisper_model, device, lr=1e-3):
        BaseAttacker.__init__(self, attack_args, whisper_model)

        self.tokenizer = get_tokenizer(self.whisper_model.model.is_multilingual, num_languages=self.whisper_model.model.num_languages, task="trancribe")
        self.device = device

        self.softprompt_model = SoftPromptModelWrapper(self.tokenizer).to(device)
        self.optimizer = torch.optim.AdamW(self.softprompt_model.parameters(), lr=lr, eps=1e-8)

    def _loss(self, logits):
        '''
        The average negative probability of the end of transcript token

        logits: Torch.tensor [batch x vocab_size]
        '''
        eot_id = self.tokenizer.eot

        sf = nn.Softmax(dim=1)
        probs = sf(logits)
        eot_probs = probs[:,eot_id].squeeze()
        return -1*torch.mean(eot_probs)

    def train_step(self, train_loader, print_freq=25):
        '''
        Run one train epoch
        '''
        losses = AverageMeter()

        # switch to train mode
        model.train()

        for i, mels in enumerate(train_loader):
            mels = [mel.to(self.device) for mel in mels]

            # Forward pass
            logits = self.softprompt_model(mels, self.whisper_model)
            loss = self._loss(logits) # check logits are in correct shape for _loss method

            # Backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            losses.update(loss.item(), ids.size(0))
            if i % print_freq == 0:
                logging.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {losses.val:.4f} ({losses.avg:.4f})')
        

    def _prep_dl(self, data, bs=16, shuffle=False):
        '''
        Create batches of lists of mel vectors
        '''
        if shuffle:
            random.shuffle(data)

        print('Creating mel vectors from audio files')
        mels = []
        for d in tqdm(data):
            mels.append(self.audio_to_mel(d['audio']))

        batches = [mels[i:i+bs] for i in range(0, len(mels), bs)]
        breakpoint()
        return batches


    def train_process(self, train_data, cache_dir, max_epochs=10, bs=16):
        set_seeds(1)

        fpath = f'{cache_dir}/softprompt_models'
        if not os.path.isdir(fpath):
            os.mkdir(fpath)

        train_dl = self._prep_dl(train_data, bs=bs, shuffle=True) # encode audio into mel vectors and each batch is a list of mel vectors

        for epoch in range(max_epochs):
            # train for one epoch
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            self.train_step(train_dl)

            # save model at this epoch
            if not os.path.isdir(f'{fpath}/epoch{epoch+1}'):
                os.mkdir(f'{fpath}/epoch{epoch+1}')
            state = self.softprompt_model.state_dict()
            torch.save(state, f'{fpath}/epoch{epoch+1}/model.th')





class SoftPromptModelWrapper(nn.Module):
    '''
        Model with the only learnable parameters being the soft prompt vectors in the mel-space
    '''
    def __init__(self, tokenizer, num_dim=80, num_vectors=32):
        super(SoftPromptModelWrapper, self).__init__()
        self.tokenizer = tokenizer

        self.softprompt = nn.Parameter(torch.rand(num_vectors, num_dim))

    def forward(mels, whisper_model):
        '''
            mels: List of audio mel vectors
            whisper_model: encoder-decoder model

            Returns the logits for the first transcribed token
        '''

        breakpoint()
        # sp_mels = # concatenate soft prompt mels to each item in list
        # padded_mels = # pad the above to 30s audio and make into batch
        # return self._mel_to_logit(padded_mels, whisper_model)

    def _mel_to_logit(self, mel: torch.Tensor, whisper_model):
        '''
            Forward pass through the whisper model of the mel vectors
            expect mel vectors passed as a batch and padded to 30s of audio length
        '''
        # create batch of start of transcript tokens
        sot_ids = self.tokenizer.sot_sequence_including_notimestamps

        #TODO convert above to torch tensor and repeat to be the same batch size as mel
        # move above to init to happen only once
        # may need to move the sot_ids to device as well

        return whisper_model.model.forward(mel, sot_ids_torch)
    

    # def transcribe(
    #     model: "Whisper",
    #     audio: Union[str, np.ndarray, torch.Tensor],
    #     *,
    #     verbose: Optional[bool] = None,
    #     temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    #     compression_ratio_threshold: Optional[float] = 2.4,
    #     logprob_threshold: Optional[float] = -1.0,
    #     no_speech_threshold: Optional[float] = 0.6,
    #     condition_on_previous_text: bool = True,
    #     initial_prompt: Optional[str] = None,
    #     word_timestamps: bool = False,
    #     prepend_punctuations: str = "\"'“¿([{-",
    #     append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    #     clip_timestamps: Union[str, List[float]] = "0",
    #     hallucination_silence_threshold: Optional[float] = None,
    #     **decode_options,
    # ):
    #     '''
    #         Mimics the original Whisper transcribe functions but prepends the softprompt vectors
    #         in the mel-vector space
    #     '''
    #     raise NotImplementedError


