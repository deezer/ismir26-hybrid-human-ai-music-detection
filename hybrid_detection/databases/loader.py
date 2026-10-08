
import logging

import pandas as pd

from .config import FMA_VERSIONS_PATHS, MUSDB_VERSIONS_PATHS


def fma() -> pd.DataFrame:
    """
    Get the list of files + some metadata for the dataset based on FMA-medium,
    providing fully real and fully generated (simulated through encodec24 autoencoding) tracks.
    """

    try:
        fma_real_df = pd.read_csv(FMA_VERSIONS_PATHS['real']['metadata'], sep=',')
        fma_fake_df = pd.read_csv(FMA_VERSIONS_PATHS['encodec24']['metadata'], sep=',')
    except FileNotFoundError as e:
        logging.error(f'Could not find FMA metadata files: '
                      f'{FMA_VERSIONS_PATHS["real"]["metadata"]} and {FMA_VERSIONS_PATHS["encodec24"]["metadata"]}.\n'
                      f'Make sure to set the database paths in "databases/config.py" '
                      f'and to run the data preparation script "databases/fma_medium.py"')
        raise FileNotFoundError(e)


    fma_real_df['codec'] = 'real'
    fma_fake_df['codec'] = 'encodec24'

    # Keep samples in common
    fma_real_df = fma_real_df.merge(fma_fake_df[['fname']], on='fname', how='inner')

    fma_all_df = pd.concat([fma_real_df, fma_fake_df], ignore_index=True) \
        .sort_values(by=['fname', 'codec']) \
        .reset_index(drop=True)

    return fma_all_df


def musdb_oracle() -> pd.DataFrame:
    """
    Get the list of files + some metadata for the dataset based on MUSDB18-HQ,
    providing hybrid, fully real and fully generated mixtures together with their oracle sources.
    """
    try:
        data_df = pd.read_csv(MUSDB_VERSIONS_PATHS['2_stems_mix_real_encodec24']['metadata'], sep=',')
    except FileNotFoundError as e:
        logging.error(f'Could not find MUSDB oracle metadata file '
                      f'{MUSDB_VERSIONS_PATHS["2_stems_mix_real_encodec24"]["metadata"]}.\n'
                      f'Make sure to set the database paths in "databases/config.py" '
                      f'and to run the data preparation script "databases/musdb_hq.py"')
        raise FileNotFoundError(e)

    data_df['codec'] = 'encodec24'
    return data_df

def musdb_demucs() -> pd.DataFrame:
    """
    Get the list of files + some metadata for the dataset based on MUSDB18-HQ,
    providing hybrid, fully real and fully generated mixtures together with their sources extracted by ht-demucs.
    """
    try:
        data_df = pd.read_csv(MUSDB_VERSIONS_PATHS['2_stems_mix_real_encodec24_demucs']['metadata'], sep=',')
    except FileNotFoundError as e:
        logging.error(f'Could not find MUSDB oracle metadata files'
                      f'{MUSDB_VERSIONS_PATHS["2_stems_mix_real_encodec24_demucs"]["metadata"]}.\n'
                      f'Make sure to set the database paths in "databases/config.py" '
                      f'and to run the data preparation script "databases/musdb_hq.py"')
        raise FileNotFoundError(e)

    data_df['codec'] = 'encodec24'
    return data_df