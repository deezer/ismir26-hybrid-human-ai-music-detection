
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import interp1d
import torch

from .stft_chunker import StftChunker


class FakePrint:
    """
    Fakeprint implementation (based on original from https://github.com/deezer/ismir25-ai-music-detector).
    This implementation extends the original method to local detection by computing representations
    from overlapping STFT chunks.

    Args:
        sr (`int`, *optional*, defaults to 44100):
            Expected sampling rate of the input waveform.
            The audio is resampled if its sampling rate differs from ``sr``.
        n_fft (`int`, *optional*, defaults to 2**14):
            Size of FFT, creates ``n_fft // 2 + 1`` frequency bins.
        f_min (`float`, *optional*, defaults to 5000):
            Minimum frequency, in Hz, included in the fakeprint representation.
        f_max (`float`, *optional*, defaults to 16000):
            Maximum frequency, in Hz, included in the fakeprint representation.
        env_filt_size (`int`, *optional*, defaults to 10):
            Filter size, as a number of frequency bins, used to detect local minima and to compute the spectral
            lower envelope ("lower hull").
        env_clip_min_db (`float`, *optional*, defaults to -60):
            Minimum envelope level relative to the envelope maximum, in dB.
            Envelope values below ``max(envelope) + env_clip_min_db`` are leading to fakeprint null values.
        clip_max_db (`float`, *optional*, defaults to 5):
            Final clipping applied to the fakeprint representation.
    """
    def __init__(self,
                 sr: int = 44100,
                 n_fft: int = 2**14,
                 f_min: float = 5000,
                 f_max: float = 16000,
                 env_filt_size: int = 10,
                 env_clip_min_db: float = -60,
                 clip_max_db: float = 5):
        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max
        self.env_filt_size = env_filt_size
        self.env_clip_min_db = env_clip_min_db
        self.clip_max_db = clip_max_db
        self.stft_chunker = StftChunker(n_fft=n_fft, power=2)

    @property
    def f_axis(self) -> np.ndarray:
        """Fakeprint frequency axis (in Hz)."""
        return self.stft_chunker.f_axis[self.f_mask]

    @property
    def f_mask(self) -> np.ndarray:
        """STFT frequency mask"""
        f_axis_full = self.stft_chunker.f_axis
        return (f_axis_full >= self.f_min) & (f_axis_full <= self.f_max)

    def compute_from_waveform(self,
                              sig: torch.Tensor,
                              chunk_dur: float | None = None,
                              chunk_ov: float = 0,
                              display: bool = False) -> np.ndarray:
        """
        Compute fakeprints on overlapping STFT chunks of an input waveform.

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
                If ``True``, display the intermediate steps of the fakeprint
                computation for a randomly selected chunk. Intended for
                illustration and debugging.

        Returns:
            fakeprint (`np.ndarray`) of shape (n_chunks, fakeprint_size)
            The fakeprint representation for all overlapping chunks.
        """
        # Pre-process
        sig /= (sig.abs().max() + 1e-6) # global peak normalization

        # Get STFT chunks
        stft = self.stft_chunker.get_from_waveform(sig, chunk_dur, chunk_ov).numpy()

        # Get average log magnitude profile on [f_min, f_max]
        spec = stft[:, :, self.f_mask, :]
        spec = 10 * np.log10(np.clip(np.abs(spec), 1e-10, 1e6))
        spec = np.mean(spec, axis=(1, 3))  # mean over channels and time -> out shape (n_chunks, fakeprint_size)

        # Compute lower spectral envelope ("lower hull")
        # Get local minima
        spec_sliding_f = sliding_window_view(spec, axis=1, window_shape=self.env_filt_size)
        spec_local_min_pos = np.argmin(spec_sliding_f, axis=-1) + np.arange(spec_sliding_f.shape[1])

        # Smooth envelope
        f_axis = self.f_axis
        smooth_env = np.zeros_like(spec)
        for chunk_id in range(smooth_env.shape[0]):
            local_min_pos = np.unique(spec_local_min_pos[chunk_id, :]).tolist()
            if local_min_pos[0] != 0:
                local_min_pos.insert(0, 0)
            if local_min_pos[-1] != len(f_axis)-1:
                local_min_pos.append(len(f_axis)-1)
            smooth_env[chunk_id, :] = interp1d(f_axis[local_min_pos],
                                               spec[chunk_id, local_min_pos],
                                               kind='quadratic')(f_axis)

        # Clip envelope
        env_clip_min = np.max(smooth_env, axis=1, keepdims=True) + self.env_clip_min_db
        smooth_env = np.clip(smooth_env, a_min=env_clip_min, a_max=None)

        # Subtract env, clip and normalize
        fakeprint = np.clip(spec - smooth_env, a_min=0, a_max=self.clip_max_db)
        fakeprint = fakeprint / (1e-6 + np.max(fakeprint, axis=1, keepdims=True))

        if display:
            import matplotlib.pyplot as plt

            rnd_chunk_id = np.random.randint(fakeprint.shape[0])
            local_min_pos = np.unique(spec_local_min_pos[rnd_chunk_id, :])

            plt.figure()
            plt.subplot(211)
            plt.title(f'chunk #{rnd_chunk_id}: spectral envelope processing')
            plt.plot(f_axis, spec[rnd_chunk_id, ...], label='mean log stft')
            plt.plot(f_axis[local_min_pos], spec[rnd_chunk_id, local_min_pos], 'g+', label='local minima')
            plt.plot(f_axis, smooth_env[rnd_chunk_id, ...], 'r--', label='smooth env')
            plt.grid()
            plt.legend()
            plt.subplot(212)
            plt.title('mean log stft to fakeprint')
            plt.plot(f_axis, (spec - smooth_env)[rnd_chunk_id, ...], label='whitened')
            plt.plot(f_axis, (fakeprint * self.clip_max_db)[rnd_chunk_id, ...], 'r', label='clipped')
            plt.grid()
            plt.legend()
            plt.xlabel('f (Hz)')

        return fakeprint

    def compute_from_file(self,
                          fp: str,
                          chunk_dur: float | None = None,
                          chunk_ov: float = 0,
                          display: bool = False,
                          **kwargs) -> np.ndarray:
        """
        Compute fakeprints on overlapping STFT chunks of an input filepath.

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
                If ``True``, display the intermediate steps of the fakeprint
                computation for a randomly selected chunk. Intended for
                illustration and debugging.
            **kwargs (`dict[str, Any]`, *optional*):
                Optional args passed to ``torchaudio.functional.resample``.

        Returns:
            fakeprint (`np.ndarray`) of shape (n_chunks, fakeprint_size)
            The fakeprint representation for all overlapping chunks.
        """
        sig = self.stft_chunker.read_file(fp, **kwargs)

        return self.compute_from_waveform(sig, chunk_dur, chunk_ov, display)
