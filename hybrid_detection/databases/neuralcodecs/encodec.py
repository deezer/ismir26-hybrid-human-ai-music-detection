
import logging

import numpy as np
import torch
from transformers import AutoProcessor, EncodecModel

from .base import Codec


# Reduce verbosity of AutoProcessor.from_pretrained & EncodecModel.from_pretrained methods
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class Encodec24(Codec):
    """
    Wrapper around EncodecModel, instantiated with config 48kHz & 24kbps, for autoencoding batches of files.
    """
    def __init__(self, device=None):
        super().__init__(device=device)

        self.batch_preprocessor = AutoProcessor.from_pretrained("facebook/encodec_48khz", force_download=False)
        self.model = EncodecModel.from_pretrained("facebook/encodec_48khz", force_download=False)
        self.model.eval()
        self.model.to(self.device)

        self.sr = self.batch_preprocessor.sampling_rate
        self.bandwidth = 24


    def autoencode(self,
                   audio_in: np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor],
                   sr_in: int | list[int]) \
            -> torch.Tensor | list[torch.Tensor]:
        """
        Auto-encode a batch of waveforms, implicitly expected to be sampled at 48kHz.

        Args:
            audio_in (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`):
                The input waveform or batch of waveforms to be autoencoded, with types and shapes as supported by
                EncodecFeatureExtractor.

            sr_in (`int`, `list[int]`):
                Sampling rate of the input waveforms.
                Ignored! Assumed all waveforms are sampled at 48kHz.

        Returns:
            audio_out (`torch.Tensor`, list[torch.Tensor]):
                The output waveform or batch of waveforms, with same shapes than the inputs.
        """
        with torch.no_grad():
            # Prepare batch of audio samples
            if isinstance(audio_in, (list, tuple)): # handle batch_preprocessor crash with list of tf.Tensors...
                for sample_id in range(len(audio_in)):
                    if isinstance(audio_in[sample_id], torch.Tensor):
                        audio_in[sample_id] = audio_in[sample_id].numpy()
            audio_batch = self.batch_preprocessor(audio_in,
                                                  sampling_rate=self.sr,
                                                  padding="longest",
                                                  return_tensors="pt").to(self.device)
            batch_size = audio_batch["input_values"].shape[0]

            # Encode
            encoder_out = self.model.encode(audio_batch["input_values"],
                                            audio_batch["padding_mask"],
                                            bandwidth=self.bandwidth)

            # Decode
            audio_out = self.model.decode(encoder_out.audio_codes,
                                          encoder_out.audio_scales,
                                          audio_batch["padding_mask"])[0].to('cpu')
            # Un-pad
            if isinstance(audio_in, (list, tuple)):
                return [audio_out[idx, :, :audio_in[idx].shape[-1]] for idx in range(batch_size)]
            else:
                return audio_out[0, :, :audio_in.shape[-1]]
