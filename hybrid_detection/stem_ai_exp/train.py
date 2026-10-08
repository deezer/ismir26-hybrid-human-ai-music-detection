
import os
import argparse
import logging
import gc

N_MAX_THREADS = 2
os.environ["OMP_NUM_THREADS"] = f'{N_MAX_THREADS}'

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(N_MAX_THREADS)
torch.set_num_interop_threads(8)

from torch.utils.data import Dataset, DataLoader
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from model import AiStemDetector
from paths import StemAiExperimentPaths
from hybrid_detection.core.eval import tpr_fpr
from hybrid_detection.utils.argparse_type import float_or_none

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)

logging.basicConfig(level=logging.INFO)
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)


class AiStemDataset(Dataset):

    def __init__(self, inputs, outputs):
        assert inputs.shape[0] == outputs.shape[0]
        self.inputs = inputs.astype(np.float32)
        self.input_size = self.inputs.shape[1]
        self.outputs = outputs.astype(np.float32)
        return

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


def preprocess_data(data_df, target_stem, single_snr_band=False, display=False):

    # Compute target stem SNR input feature
    if not single_snr_band:  # default: 10 SNR bands between 0 and 16kHZ (1kHz bandwidth)
        data_df['target_snr_db'] = np.unstack(
            10 * (np.log10(np.clip(np.stack(data_df['vocals_energy'].to_list()), 1e-8, None)) -
                  np.log10(np.clip(np.stack(data_df['accomp_energy'].to_list()), 1e-8, None))),
        )
    else: # a single SNR band between 5 and 16kHZ
        data_df['target_snr_db'] = np.unstack(
            10 * (np.log10(np.clip(np.stack(data_df['vocals_energy'].to_list())[:, 5:].sum(axis=1), 1e-8, None)) -
                  np.log10(np.clip(np.stack(data_df['accomp_energy'].to_list())[:, 5:].sum(axis=1), 1e-8, None))),
        )

    if target_stem == 'accomp':
        data_df[['target_gain_db', 'target_snr_db']] = - data_df[['vocals_gain_db', 'target_snr_db']]
    elif target_stem == 'vocals':
        data_df[['target_gain_db', 'target_snr_db']] = data_df[['vocals_gain_db', 'target_snr_db']]
    else:
        raise ValueError(f"Unknown stem {target_stem}")

    data_df['target_stem'] = target_stem

    # Add target stem label
    label_map = {'vr_ar': {'vocals': 0, 'accomp': 0},
                 'vg_ar': {'vocals': 1, 'accomp': 0},
                 'vr_ag': {'vocals': 0, 'accomp': 1},
                 'vg_ag': {'vocals': 1, 'accomp': 1}}

    label_df = data_df[['track', 'mix_type', 'target_gain_db']].drop_duplicates()
    label_df['label'] = label_df.apply(lambda x: label_map[x['mix_type']][target_stem], axis=1)
    data_df = data_df.merge(label_df, how='inner', on=['track', 'mix_type', 'target_gain_db'])

    # Detect silences (mix chunks with energy < max energy)
    max_energy_q = 0.95
    silence_energy_ratio = 2/100

    data_df['mix_energy'] = np.stack(data_df['vocals_energy'].to_list()).sum(axis=1) \
                            + np.stack(data_df['accomp_energy'].to_list()).sum(axis=1)
    track_max_energy_df = data_df \
        .groupby(['track', 'mix_type', 'target_gain_db']) \
        .agg({'mix_energy': lambda x: np.quantile(x, max_energy_q)}) \
        .reset_index(drop=False) \
        .rename(columns={'mix_energy': 'mix_max_energy'})
    data_df = data_df \
        .merge(track_max_energy_df, how='left', on=['track', 'mix_type', 'target_gain_db'])
    data_df['is_silence'] = (data_df['mix_energy'] / data_df['mix_max_energy']) <= silence_energy_ratio

    # Display silence detection and features for a random track
    if display:
        import matplotlib.pyplot as plt

        # select a random hybrid track with a high number of silence chunk to illustrate influence of preproc
        track_df = data_df[(data_df['target_gain_db'] == 0) &
                            (data_df['label'] == 1) &
                            (data_df['mix_type'].isin(['vg_ar', 'vr_ag']))] \
            .groupby(['track', 'mix_type', 'target_gain_db']) \
            .agg({'is_silence': sum}) \
            .reset_index(drop=False) \
            .sort_values(by=['is_silence'], ascending=False) \
            .head(10).sample(1).drop(columns='is_silence')

        track_df = data_df \
            .merge(track_df, how='inner', on=['track', 'mix_type', 'target_gain_db']) \
            .sort_values(by='t')

        t_axis = track_df['t'].values
        silence_mask = track_df['is_silence'].values.astype(int)

        plt.figure()
        plt.subplot(311)
        plt.title(f'{track_df["track"].tolist()[0]} | '
                  f'mix_config = {track_df["mix_type"].tolist()[0]} | '
                  f'target = {target_stem} at {track_df["target_gain_db"].tolist()[0]} dB')
        plt.plot(t_axis, track_df['mix_energy'].values, label='chunk energy')
        plt.plot(t_axis, track_df['mix_max_energy'].values, label=f'energy q={max_energy_q}')
        plt.plot(t_axis, track_df['mix_max_energy'].values * silence_energy_ratio, label='silence thr')
        plt.fill_between(t_axis, track_df['mix_energy'].max() * silence_mask, color='k', alpha=0.15, label='silence')
        plt.legend()
        plt.xlim(t_axis[0], t_axis[-1])
        plt.grid()
        plt.ylabel('mix energy')
        plt.subplot(312)
        plt.plot(t_axis, track_df['full_ai_logit_score'].values)
        plt.fill_between(t_axis,
                         y1=track_df['full_ai_logit_score'].min() * (1-silence_mask) +
                            track_df['full_ai_logit_score'].max() * silence_mask,
                         y2=np.ones_like(t_axis) * track_df['full_ai_logit_score'].min(),
                         color='k', alpha=0.15)
        plt.xlim(t_axis[0], t_axis[-1])
        plt.grid()
        plt.ylabel('mix logit fake score')
        plt.subplot(313)
        plt.imshow(np.stack(track_df['target_snr_db'].values).T, aspect='auto', origin='lower', cmap='jet',
                   extent=(t_axis[0], t_axis[-1], 0., 16000.), vmin=-15, vmax=15)
        plt.grid()
        plt.ylabel('target SNR (dB)')
        plt.xlabel('time (s)')

    return data_df.drop(columns=['vocals_energy', 'accomp_energy'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", help="target stem", choices=['vocals', 'accomp'], type=str)
    parser.add_argument("--db", help="training database", choices=['musdb_oracle', 'musdb_demucs'], type=str)
    parser.add_argument("--codec", help="neural codec", choices=['encodec24'], type=str, default='encodec24')
    parser.add_argument("--chunk_dur", help="chunk duration (in sec)", type=float_or_none, default=5)
    parser.add_argument("--train_ratio", help="train proportion", type=float, default=0.7)
    parser.add_argument("--random_seed", help="for reproducibility", type=int, default=42)
    parser.add_argument("--single_snr_band", help="use a single SNR band [5, 16] kHz instead of the 16 bands in [0, 16] kHz", action='store_true')
    args = parser.parse_args()

    logging.info(f'Starting the training of the stem AI music detector with config:\n{args}')

    # Prepare experiments directory
    exp_paths = StemAiExperimentPaths()

    data_dir = exp_paths.data_dir(args.db, args.codec, args.chunk_dur)

    model_dir = exp_paths.model_dir(args.db, args.codec, args.stem, args.chunk_dur, args.random_seed,
                                    single_snr_band=args.single_snr_band, mkdir=True)
    out_predictions_path = os.path.join(model_dir, 'train_test_predictions.parquet')

    logging.info(f'Loading & pre-processing input data from {data_dir}')

    seed_everything(args.random_seed, workers=True)

    # Load and preprocess data
    full_data_df = pd.read_parquet(data_dir)
    full_data_df = preprocess_data(full_data_df, args.stem, args.single_snr_band)

    split_df = full_data_df[['track']] \
        .drop_duplicates() \
        .sort_values(by=['track']) \
        .sample(frac=args.train_ratio, replace=False, random_state=args.random_seed)
    split_df['train_set'] = True

    full_data_df = full_data_df \
        .merge(split_df, how='left', on='track') \
        .fillna({'train_set': False})

    full_data_df['seed'] = args.random_seed

    stats_df = full_data_df \
        .groupby(['train_set']) \
        .agg({'track': pd.Series.nunique,
              'mix_type': set,
              'target_gain_db': set,
              'chunk_id': len,
              'is_silence': sum}) \
        .rename(columns={'track': 'n_tracks', 'chunk_id': 'n_chunks_kept', 'is_silence': 'n_chunks_silence'})
    stats_df['n_chunks_kept'] -= stats_df['n_chunks_silence']
    logging.info(f"\n{stats_df}")

    logging.info(f'Initializing model and training')

    # Train model
    # Could probably benefit from some fine-tuning:
    # validation early stopping, LR reduction on plateau, model archi fine-tuning, ...

    def get_input_features(df):
        return np.hstack(
            [np.vstack(df[col].to_list()) for col in ['full_ai_logit_score', 'target_snr_db']]  # stack input features
        ).astype(np.float32)

    train_df = full_data_df[full_data_df['train_set'] & ~full_data_df['is_silence']]  # drop silences

    train_dataset = AiStemDataset(inputs=get_input_features(train_df),
                                  outputs=train_df['label'].values[:, None])

    train_loader = DataLoader(train_dataset,
                              batch_size=1024,
                              shuffle=True,
                              num_workers=4,
                              persistent_workers=True)

    stem_detector = AiStemDetector(input_size=train_dataset.input_size)

    callbacks = [
        ModelCheckpoint(dirpath=model_dir,
                        filename='model-{epoch}-{train_loss:.2f}-{train_tpr:.2f}-{train_fpr:.2f}',
                        monitor='train_fpr',
                        mode='min',
                        save_weights_only=True)
        ,
        TQDMProgressBar(refresh_rate=200, leave=True)
    ]

    logger = CSVLogger(save_dir=model_dir)

    trainer = Trainer(max_epochs=50,
                      accelerator="cpu", # faster than gpu for such small model & dataset
                      deterministic="warn",
                      callbacks=callbacks,
                      logger=logger)

    trainer.fit(stem_detector, train_loader)

    # Free some memory
    train_df = None
    train_dataset = None
    train_loader = None
    gc.collect()

    # Run predictions and export
    logging.info(f'Running model prediction and saving to {out_predictions_path}')

    X = get_input_features(full_data_df)

    stem_detector.eval()
    with torch.no_grad():
        ai_score = stem_detector(torch.tensor(X)).detach().cpu().numpy()

    out_data_df = full_data_df[['track', 'mix_type', 'vocals_gain_db', 'codec', 'target_stem', 'target_gain_db',
                                'chunk_dur', 'chunk_ov', 'chunk_id', 't', 'label', 'is_silence', 'train_set',  'seed']]
    out_data_df['stem_ai_score'] = ai_score

    out_data_df.to_parquet(out_predictions_path, index=False)

    # Display eval
    exp_config_cols = ['seed', 'chunk_dur', 'target_stem', 'train_set', 'target_gain_db']
    track_cols = ['track', 'mix_type', 'target_stem', 'target_gain_db']

    eval_df = out_data_df[~out_data_df['is_silence']] \
        .groupby(exp_config_cols, dropna=False) \
        .apply(lambda x: tpr_fpr(x['label'], x['stem_ai_score'], as_pandas=True)) \
        .reset_index(drop=False) \
        .pivot(index=exp_config_cols[:-1], columns=exp_config_cols[-1], values=['tpr', 'fpr'])

    logging.info(f"Chunk-level evaluation \n{eval_df}")

    eval_df = out_data_df[~out_data_df['is_silence']] \
        .groupby(exp_config_cols + track_cols) \
        .agg({'stem_ai_score': 'mean', 'label': 'mean'}) \
        .groupby(exp_config_cols, dropna=False) \
        .apply(lambda x: tpr_fpr(x['label'], x['stem_ai_score'], as_pandas=True)) \
        .reset_index(drop=False) \
        .pivot(index=exp_config_cols[:-1], columns=exp_config_cols[-1], values=['tpr', 'fpr'])

    logging.info(f"Track-level evaluation \n{eval_df}")
