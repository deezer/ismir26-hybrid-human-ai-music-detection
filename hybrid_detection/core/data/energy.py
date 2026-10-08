
import numpy as np
import torch

from .stft_chunker import StftChunker


class BandSplitEnergy:
    """
    Compute band-split energy from overlapping STFT chunks.

    Args:
        sr (`int`, *optional*, defaults to 44100):
            Expected sampling rate of the input waveform.
            The audio is resampled if its sampling rate differs from ``sr``.
        n_fft (`int`, *optional*, defaults to 2**14):
            Size of FFT, creates ``n_fft // 2 + 1`` frequency bins.
        f_min (`float`, *optional*, defaults to 5000):
            Minimum frequency, in Hz, included in the energy representation.
        f_max (`float`, *optional*, defaults to 16000):
            Maximum frequency, in Hz, included in the energy representation.
        filters_bandwidth (`float`, *optional*, defaults to 1000):
            Bandwidth, in Hz, of band-pass energy filters.
        **kwargs (`dict[str, Any]`, *optional*):
            Optional additional args passed to ``StftChunker``.
    """
    def __init__(self,
                 sr: int = 44100,
                 n_fft: int = 2 ** 14,
                 f_min: float = 0,
                 f_max: float = 16000,
                 filters_bandwidth: float = 1000,
                 **kwargs):
        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max
        self.filters_bandwidth = filters_bandwidth
        self.stft_chunker = StftChunker(n_fft=n_fft, power=2, normalized=True, **kwargs)

    @property
    def filter_bank(self) -> np.ndarray:
        band_edges = np.arange(self.f_min, self.f_max+1, self.filters_bandwidth)
        n_bands = len(band_edges)-1
        stft_f_axis = self.stft_chunker.f_axis
        filt_mat = np.zeros((len(stft_f_axis), n_bands), dtype=np.float32)
        for filt_id, (f_min, f_max) in enumerate(zip(band_edges[:-1], band_edges[1:])):
            filt_mat[(stft_f_axis >= f_min) & (stft_f_axis < f_max), filt_id] = 1
        return filt_mat

    def compute_from_waveform(self,
                              sig: torch.Tensor,
                              chunk_dur: float | None = None,
                              chunk_ov: float = 0,
                              display: bool = False) -> np.ndarray:
        """
        Compute band-split energy on overlapping STFT chunks of an input waveform.

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
            display (`bool`, *optional*, defaults to False):
                Intended for illustration and debugging.

        Returns:
            `np.ndarray` of shape (n_chunks, n_bands)
            The band-split energy representation for all overlapping chunks.
        """
        # Get chunks energy vs freq
        # stft shape (n_chunks, n_chans, n_freq, chunk_size)
        stft = self.stft_chunker.get_from_waveform(sig, chunk_dur, chunk_ov).numpy()
        chunks_freq_energy = np.sum(stft, axis=(1, 3))   # mean over channels and time -> out shape (n_chunks, n_freq)

        # Apply filter bank
        filter_bank = self.filter_bank # (n_freq, n_filt)
        tf_energy = np.matmul(chunks_freq_energy, filter_bank) # (n_chunks, n_filt)

        if display:
            import matplotlib.pyplot as plt
            f_axis = self.stft_chunker.f_axis
            filter_bank = self.filter_bank

            plt.figure()
            plt.subplot(211)
            plt.title('Filter bank')
            plt.imshow(filter_bank, aspect='auto', origin='lower', interpolation='none',
                       extent=(0, filter_bank.shape[1]-1, f_axis[0], f_axis[-1]))
            plt.xlabel('band ids')
            plt.ylabel('frequency (Hz)')
            plt.colorbar()
            plt.subplot(212)
            plt.title('TF energy (dB)')
            plt.imshow(10*np.log10(tf_energy).T, aspect='auto', origin='lower', cmap='jet')
            plt.colorbar()
            plt.xlabel('chunk ids')
            plt.ylabel('filt ids')

        return tf_energy

    def compute_from_file(self,
                          fp: str,
                          chunk_dur: float | None = None,
                          chunk_ov: float = 0,
                          display: bool = False,
                          **kwargs) -> np.ndarray:
        """
        Compute band-split energy on overlapping STFT chunks of an input filepath.

        Args:
            fp (`str`):
                Input filepath.
            chunk_dur (`float`, *optional*, defaults to None):
                Duration of each STFT chunk, in seconds.
                The duration will be quantized to the nearest valid number of STFT frames.
            chunk_ov (`float`, *optional*, defaults to 0):
                Overlap between consecutive chunks, in the interval ``[0, 1[``.
                The overlap will be quantized to the nearest valid number of STFT frames.
            display (`bool`, *optional*, defaults to False):
                Intended for illustration and debugging.
            **kwargs (`dict[str, Any]`, *optional*):
                Optional args passed to ``torchaudio.functional.resample``.

        Returns:
            `np.ndarray` of shape (n_chunks, n_bands)
            The band-split energy representation for all overlapping chunks.
        """
        sig = self.stft_chunker.read_file(fp, **kwargs)

        return self.compute_from_waveform(sig, chunk_dur, chunk_ov, display)
