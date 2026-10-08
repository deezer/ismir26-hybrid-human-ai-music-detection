
import os
import logging
from tqdm import tqdm

import pandas as pd
from torchcodec.decoders import AudioDecoder

from config import FMA_VERSIONS_PATHS
from neuralcodecs.encodec import Encodec24

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 2000)

logging.basicConfig(level=logging.INFO)


def load_dataset(version='real'):
    return pd.read_csv(FMA_VERSIONS_PATHS[version]['metadata'], sep=',')


def parse_dataset(version='real'):
    data_dir = FMA_VERSIONS_PATHS[version]['data']
    metadata_out_path = FMA_VERSIONS_PATHS[version]['metadata']

    logging.info(f'Parsing FMA-medium {version} dataset from {data_dir}')
    logging.info(f'metadata will be saved in {metadata_out_path}')
    dataset_metadata = []

    for split_dir in tqdm(sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])):
        for f_path in sorted([f.path for f in os.scandir(split_dir) if not f.name.startswith('.')]):
            try:
                audio_md = AudioDecoder(f_path).metadata
            except :
                logging.warning(f'Cannot load {f_path}, ignoring it!')
                continue

            dataset_metadata.append({'fname': os.path.basename(f_path),
                                     'sr': audio_md.sample_rate,
                                     'dur': audio_md.duration_seconds,
                                     'n_channels': audio_md.num_channels,
                                     'fpath': f_path})

    dataset_metadata_df = pd.DataFrame(dataset_metadata)
    os.makedirs(os.path.dirname(metadata_out_path), exist_ok=True)
    dataset_metadata_df.to_csv(metadata_out_path, sep=',', header=True, index=False)

    return


def autoencode_with_encodec24(batch_size=4, device=None):
    logging.info(f'Auto-encoding FMA-medium real dataset with encodec24')
    data_df = load_dataset(version='real')

    # Drop a few "outliers", for simplicity (from 24985 to 23542 samples)
    data_df = data_df[data_df['sr'] == 44100]
    data_df = data_df[data_df['dur'] >= 25]
    data_df = data_df[data_df['n_channels'] == 2]
    data_df = data_df.reset_index(drop=True)

    # Prepare file batches
    data_df['batch_id'] = data_df.index.values // batch_size
    data_df['fpath_out'] = data_df['fpath'] \
        .apply(lambda x: x.replace(FMA_VERSIONS_PATHS['real']['data'],
                                   FMA_VERSIONS_PATHS['encodec24']['data']))

    # Auto-encode and export
    encodec24 = Encodec24(device=device)
    for batch_id in tqdm(data_df['batch_id'].unique()):
        batch_df = data_df[data_df['batch_id'] == batch_id]
        encodec24.autoencode_from_files(fp_in=batch_df['fpath'].tolist(),
                                        fp_out=batch_df['fpath_out'].tolist())

    return


if __name__ == '__main__':

    parse_dataset(version='real')

    autoencode_with_encodec24(batch_size=128)  # can take a few hours
    parse_dataset(version='encodec24')
