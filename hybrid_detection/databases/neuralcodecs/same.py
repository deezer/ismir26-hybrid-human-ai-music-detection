

import torch

# WORKAROUND: Disable Triton compiler for flex_attention.
# ROOT CAUSE: Using a newer PyTorch version than what stable-audio-3 requires
# triggers a flex_attention compilation that exceeds consumer GPU hardware limits (64KB SRAM).
# Overriding fa.compile forces "Eager" mode, bypassing the crash.
try:
    import torch.nn.attention.flex_attention as fa
    fa.compile = lambda f, *args, **kwargs: f
except (ImportError, AttributeError):
    pass

from stable_audio_3 import AutoencoderModel

from .base import Codec


class Same(Codec):
    """
    Wrapper around SAME stable audio 3 encoder for autoencoding batches of files.
    """
    def __init__(self, version='same-l', chunked=False, chunk_size=128, overlap=32, device=None):
        super().__init__(device=device)

        self.model = AutoencoderModel.from_pretrained(version)

        self.chunked = chunked
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.sr = self.model.sample_rate  # 44.1kHz


    def autoencode(self,
                   audio_in: torch.Tensor | list[torch.Tensor],
                   sr_in: int | list[int]) \
            -> torch.Tensor | list[torch.Tensor]:
        """
        Auto-encode a batch of waveforms.

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
            encoder_out = self.model.encode(
                audio_in, sr=sr_in, chunked=self.chunked, chunk_size=self.chunk_size, overlap=self.overlap
            )

            # Decode
            audio_out = self.model.decode(
                encoder_out, chunked=self.chunked, chunk_size=self.chunk_size, overlap=self.overlap
            ).to('cpu')

            # Un-pad
            if isinstance(audio_in, (list, tuple)):
                return [audio_out[idx, :, :audio_in[idx].shape[-1]] for idx in range(len(audio_in))]
            else:
                return audio_out[0, :, :audio_in.shape[-1]]
