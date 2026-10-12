

import torch
from stable_audio_3 import AutoencoderModel

from .base import Codec


class Same(Codec):
    """
    Wrapper around EncodecModel, instantiated with config 48kHz & 24kbps, for autoencoding batches of files.
    """
    def __init__(self, version='same-l', device=None):
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')

        self.model = AutoencoderModel.from_pretrained(version)

        self.sr = self.model.sample_rate

    def autoencode(self,
                   audio_in: torch.Tensor | list[torch.Tensor],
                   sr_in: int | list[int]) \
            -> torch.Tensor | list[torch.Tensor]:
        """
        Auto-encode a batch of waveforms, implicitly expected to be sampled at 48kHz.

        Args:
            audio_in (`torch.Tensor`, `list[torch.Tensor]`):
                The input waveform or batch of waveforms to be autoencoded.

            sr_in (`int`, `list[int]`):
                Sampling rate of the input waveforms.

        Returns:
            audio_out (`torch.Tensor`, list[torch.Tensor]):
                The output waveform or batch of waveforms, with same shapes than the inputs.
        """
        with torch.no_grad():
            # Pre-process & encode
            encoder_out = self.model.encode(audio_in, sr=sr_in)

            # Decode
            audio_out = self.model.decode(encoder_out).to('cpu')

            # Un-pad
            if isinstance(audio_in, (list, tuple)):
                return [audio_out[idx, :audio_in[idx].shape[-1]] for idx in range(len(audio_in))]
            else:
                return audio_out[:, :audio_in.shape[-1]]
