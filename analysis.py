import torch
import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from whisper.audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from math import log

from src.tools.tools import get_default_device
from src.tools.args import core_args, attack_args, analysis_args
from src.tools.saving import base_path_creator, attack_base_path_creator_eval, attack_base_path_creator_train
from src.models.load_model import load_model
from src.data.load_data import load_data
from src.attacker.selector import select_eval_attacker

if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()
    analysis_args, _ = analysis_args()

    print(core_args)
    print(attack_args)
    print(analysis_args)

    if not attack_args.transfer:
        base_path = base_path_creator(core_args)
        attack_base_path = attack_base_path_creator_eval(attack_args, base_path)
    else:
        base_path = None
        attack_base_path = None


    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/analysis.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if core_args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device(core_args.gpu_id)
    print(device)

    # Load the model
    model = load_model(core_args, device=device)

    # load attacker for evaluation
    attacker = select_eval_attacker(attack_args, core_args, model, device=device)

    # extract the softprompt model attack mel vectors
    softprompt_model = attacker.softprompt_model
    if not attack_args.transfer:
        softprompt_model_dir = f'{attack_base_path_creator_train(attack_args, base_path)}/softprompt_models'
    else:
        softprompt_model_dir = attack_args.softprompt_model_dir
    softprompt_model.load_state_dict(torch.load(f'{softprompt_model_dir}/epoch{attack_args.attack_epoch}/model.th'))

    log_adv_mel = softprompt_model.softprompt.cpu().detach()
    log_adv_mel = log_adv_mel - log(attack_args.k_scale)

    if analysis_args.compare_with_audio:
        _, data = load_data(core_args)
        log_audio_mel = log_mel_spectrogram(data[7]['audio'], model.model.dims.n_mels)
        log_adv_mel = torch.cat((log_adv_mel, log_audio_mel), dim=1)

    adv_mel = np.exp(log_adv_mel.numpy())

    # plot the heatmap
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    img = librosa.display.specshow(adv_mel, y_axis='linear', x_axis='time', hop_length=HOP_LENGTH,
                                sr=SAMPLE_RATE)
    # ax[0].set(title='Linear-frequency power spectrogram')
    # ax[0].label_outer()

    # librosa.display.specshow(adv_mel, y_axis='log', sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
    #                         x_axis='time', ax=ax[1])
    # ax[1].set(title='Log-frequency power spectrogram')
    # ax[1].label_outer()
    fig.colorbar(img, ax=ax, format="%.2f dB")

    # save image
    add_for_k = ''
    if attack_args.k_scale > 1:
        add_for_k = f'k{attack_args.k_scale}'

    if analysis_args.compare_with_audio:
        ax.set_xlim(right=3)
        if attack_args.transfer:
            save_path = 'experiments/transfer_spectrogram.png'
        else:
            save_path = f'{attack_base_path}/comparison_spectrogram{add_for_k}.png'
    else:
        save_path = f'{attack_base_path}/spectrogram{add_for_k}.png'
    fig.savefig(save_path, bbox_inches='tight')

