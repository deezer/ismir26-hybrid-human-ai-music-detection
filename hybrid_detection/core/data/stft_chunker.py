
import logging

import numpy as np
import torch
from torch import Tensor
from torchaudio.transforms import Spectrogram
from torchcodec.decoders import AudioDecoder
from torchaudio.functional import resample


class StftChunker:
    """
    Wrapper around torchaudio.transforms.Spectrogram to compute the STFT from a waveform or an audio file,
    and to slice it into overlapping chunks (temporal slices).

    Args:
        sr (`int`, *optional*, defaults to 44100):
            Sampling rate of expected waveforms. If an input audio file does not match it, a resampling is performed.
        **kwargs (`dict[str, Any]`, *optional*):
            Optional args passed to ``torchaudio.transforms.Spectrogram``.
    """
    def __init__(self, sr: int = 44100, **kwargs):
        self.sr = sr
        self.stft = Spectrogram(**kwargs).eval()

    @property
    def f_axis(self) -> np.ndarray:
        """STFT frequency axis in Hz"""
        return np.linspace(0, self.sr / 2, num=self.stft.n_fft // 2 + 1)

    def chunk_size(self, chunk_dur: float) -> int:
        """Convert a chunk duration (in seconds) to a chunk size (in number of stft frames)."""
        return max(1, int(np.floor(1 + (chunk_dur * self.sr - self.stft.win_length) / self.stft.hop_length)))

    def chunk_hop_size(self, chunk_dur: float, chunk_ov: float) -> int:
        """Convert a chunk duration (in seconds) + overlap (in [0, 1[) to a chunk hop size (in number of stft frames)."""
        assert 0 <= chunk_ov < 1
        chunk_size = self.chunk_size(chunk_dur)
        return max(1, int(chunk_size * (1 - chunk_ov)))

    def frames_to_sec(self, n: int) -> float:
        """Returns the duration (in seconds) spanned by n consecutive stft frames."""
        return ((n - 1) * self.stft.hop_length + self.stft.win_length) / self.sr

    def info_chunk_quantization(self, chunk_dur: float, chunk_ov: float):
        chunk_dur_quantized = self.frames_to_sec(self.chunk_size(chunk_dur))
        chunk_hop_quantized = self.frames_to_sec(self.chunk_hop_size(chunk_dur, chunk_ov))
        logging.info(f'Requested chunk duration {chunk_dur} will be quantized to '
                     f'{np.round(chunk_dur_quantized, 2)} to match the STFT configuration.')
        logging.info(f'Requested chunk overlap {chunk_ov} will be quantized to '
                     f'{np.round(1-chunk_hop_quantized/chunk_dur_quantized, 2)} '
                     f'to match the STFT configuration.')

    def t_axis_chunks(self, chunk_dur: float, chunk_ov: float, n_chunks: int) -> np.ndarray:
        """Returns the time grid (in sec) corresponding to n consecutive chunks."""
        chunk_quantized_dur = self.frames_to_sec(self.chunk_size(chunk_dur))
        chunk_quantized_hop_dur = self.frames_to_sec(self.chunk_hop_size(chunk_dur, chunk_ov))
        # Note: below-term self.stft.hop_length/self.sr account for the fact that stft chunks overlap in the time domain
        t_axis = np.arange(n_chunks) * (chunk_quantized_hop_dur - self.stft.hop_length/self.sr) + chunk_quantized_dur / 2
        if self.stft.center:
            t_axis -= ((self.stft.win_length/2) / self.sr)
        return t_axis

    def get_from_waveform(self,
                          sig: torch.Tensor,
                          chunk_dur: float | None = None,
                          chunk_ov: float = 0) -> torch.Tensor:
        """
        Compute the STFT from an input waveform and slice it into overlapping chunks.

        Args:
            sig (`torch.Tensor`):
                Input waveform of dimensions (n_channels, n_samples).
                Implicitly expected to be sampled at self.sr.
            chunk_dur (`float`, *optional*, defaults to None):
                Duration of each STFT chunk, in seconds.
                The duration will be quantized to the nearest valid number of STFT frames.
            chunk_ov (`float`, *optional*, defaults to 0):
                Overlap between consecutive chunks, in the interval ``[0, 1[``.
                The overlap will be quantized to the nearest valid number of STFT frames.

        Returns:
            Tensor: Sliced STFT of dimensions (n_chunks, n_channels, n_freqs, chunk_size)
        """

        # Compute STFT from full sig
        stft = self.stft(sig)  # (n_chan, n_freq, n_frames)

        # Slice into chunks
        if chunk_dur is None:
            chunk_size = stft.shape[2]
            chunk_hop = chunk_size
        else:
            chunk_size = self.chunk_size(chunk_dur)
            chunk_hop = self.chunk_hop_size(chunk_dur, chunk_ov)

        n_chunks = max(1, int(np.floor(1 + (stft.shape[2] - chunk_size) / chunk_hop)))
        stft_chunks = [stft[..., idx * chunk_hop: idx * chunk_hop + chunk_size] for idx in range(n_chunks)]

        return torch.stack(stft_chunks) # (n_chunks, n_chans, n_freq, chunk_size)

    def get_from_file(self,
                      fp: str,
                      chunk_dur: float | None = None,
                      chunk_ov: float = 0,
                      **kwargs) -> torch.Tensor:
        """
        Compute the STFT from an input audio file and slice it into overlapping chunks.

        Args:
            fp (`str`):
                Input filepath.
            chunk_dur (`float`, *optional*, defaults to None):
                Duration of each STFT chunk, in seconds.
                The duration will be quantized to the nearest valid number of STFT frames.
            chunk_ov (`float`, *optional*, defaults to 0):
                Overlap between consecutive chunks, in the interval ``[0, 1[``.
                The overlap will be quantized to the nearest valid number of STFT frames.
            **kwargs (`dict[str, Any]`, *optional*):
                Optional args passed to ``torchaudio.functional.resample``.

        Returns:
            Tensor: Sliced STFT of dimensions (n_chunks, n_channels, n_freqs, chunk_size)
        """
        sig = self.read_file(fp, **kwargs)

        return self.get_from_waveform(sig, chunk_dur, chunk_ov)

    def read_file(self, fp: str, **kwargs) -> torch.Tensor:
        """
        Read an input audio file and resample it if it doesn't match the expected sampling rate.

        Args:
            fp (`str`):
                Input filepath.
            **kwargs (`dict[str, Any]`, *optional*):
                Optional args passed to ``torchaudio.functional.resample``.

        Returns:
            Tensor: audio waveform of dimensions (n_channels, n_samples).

        """
        audio_samples = AudioDecoder(fp).get_all_samples()
        sig = audio_samples.data
        sr = audio_samples.sample_rate

        if sr != self.sr:
            sig = resample(sig, sr, self.sr, **kwargs)

        return sig
