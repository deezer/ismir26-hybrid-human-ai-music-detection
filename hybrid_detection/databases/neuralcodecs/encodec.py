
import os
import logging

import numpy as np
import torch
from torchcodec.encoders import AudioEncoder
from torchcodec.decoders import AudioDecoder
from torchaudio.functional import resample
from transformers import AutoProcessor, EncodecModel


# Reduce verbosity of AutoProcessor.from_pretrained & EncodecModel.from_pretrained methods
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class Encodec24:
    """
    Wrapper around EncodecModel, instantiated with config 48kHz & 24kbps, for autoencoding batches of files.
    """
    def __init__(self, device=None):
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')

        self.batch_preprocessor = AutoProcessor.from_pretrained("facebook/encodec_48khz", force_download=False)
        self.model = EncodecModel.from_pretrained("facebook/encodec_48khz", force_download=False)
        self.model.eval()
        self.model.to(self.device)

        self.sr = self.batch_preprocessor.sampling_rate
        self.bandwidth = 24


    def autoencode(self, audio_in: np.ndarray | torch.Tensor | list[np.ndarray]) -> torch.Tensor | list[torch.Tensor]:
        """
        Auto-encode a batch of waveforms, implicitly expected to be sampled at 48kHz.

        Args:
            audio_in (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`):
                The input waveform or batch of waveforms to be autoencoded, with types and shapes as supported by
                EncodecFeatureExtractor.

        Returns:
            audio_out (`torch.Tensor`, list[torch.Tensor]):
                The output waveform or batch of waveforms, with same shapes than the inputs.
        """
        with torch.no_grad():
            # Prepare batch of audio samples
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


    def autoencode_from_files(self,
                              fp_in: str | list[str],
                              fp_out: str | list[str],
                              sr_forced: int | None = 44100,
                              **kwargs):
        """
        Auto-encode and export a batch of files.

        Args:
            fp_in (`str`, `list[str]`):
                Input audio filepath or batch of audio filepaths.
            fp_out (`str`, `list[str]`):
                Output audio filepath or batch of audio filepaths.
            sr_forced (`int`, *optional*, defaults to 44100):
                Hack to allow explicitly defining a sampling frequency different from the one supported by encodec
                (when None, defaults to 48000).
                44100 default implies that 44.1kHz waveforms (potentially obtained through resampling) will be sent to
                encodec, thus seen as 48kHz waveforms.
                Datasets used for our experiments (FMA and MUSDB-HQ) are sampled at 44.1kHz, this trick avoids doing
                multiple resampling operations along the pipeline: 44.1->48 for the autoencoding, 48->44.1 for the
                "fakeprint" computation. While conducting experiments we tested with sr_forced both set to 44100
                and None (on a data subset): we did not see any impact on the AI detection performance (neither a
                clear audible impact on autoencoded audios).
            **kwargs (`dict[str, Any]`, *optional*):
                Optional args passed to torchaudio.functional.resample
        """
        fp_in = [fp_in] if isinstance(fp_in, str) else fp_in
        fp_out = [fp_out] if isinstance(fp_out, str) else fp_out
        assert len(fp_in) == len(fp_out)

        if sr_forced is None:
            sr_forced = self.sr

        # Load a batch of input audio files, resample if needed
        audio_samples_in = []
        for fp in fp_in:
            loader = AudioDecoder(fp, num_channels=2)
            sig = loader.get_all_samples().data
            sr = loader.metadata.sample_rate
            bit_rate = loader.metadata.bit_rate

            if sr != sr_forced:
                sig = resample(sig, sr, sr_forced, **kwargs)
                sr = sr_forced

            audio_samples_in.append({'sig': sig.numpy(), 'sr': sr, 'bit_rate': bit_rate})

        # Auto-encode
        waveforms_out = self.autoencode([x['sig'] for x in audio_samples_in])

        # Export audio files
        for wf, fp, audio_in in zip(waveforms_out, fp_out, audio_samples_in):
            audio_saver = AudioEncoder(samples=wf, sample_rate=audio_in['sr'])
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            audio_saver.to_file(dest=fp, bit_rate=audio_in['bit_rate'])
