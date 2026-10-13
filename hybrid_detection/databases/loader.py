
import logging

import pandas as pd

from .config import FMA_VERSIONS_PATHS, MUSDB_VERSIONS_PATHS


def fma(codec: str = 'encodec24') -> pd.DataFrame:
    """
    Get the list of files + some metadata for the dataset based on FMA-medium,
    providing fully real and fully generated tracks (simulated through autoencoding).
    """

    try:
        fma_real_df = pd.read_csv(FMA_VERSIONS_PATHS['real']['metadata'], sep=',')
        fma_fake_df = pd.read_csv(FMA_VERSIONS_PATHS[codec]['metadata'], sep=',')
    except FileNotFoundError as e:
        logging.error(f'Could not find FMA metadata files: '
                      f'{FMA_VERSIONS_PATHS["real"]["metadata"]} and {FMA_VERSIONS_PATHS[codec]["metadata"]}.\n'
                      f'Make sure to set the database paths in "databases/config.py" '
                      f'and to run the data preparation script "databases/fma_medium.py"')
        raise FileNotFoundError(e)


    fma_real_df['codec'] = 'real'
    fma_fake_df['codec'] = codec

    # Keep samples in common
    fma_real_df = fma_real_df.merge(fma_fake_df[['fname']], on='fname', how='inner')

    fma_all_df = pd.concat([fma_real_df, fma_fake_df], ignore_index=True) \
        .sort_values(by=['fname', 'codec']) \
        .reset_index(drop=True)

    return fma_all_df


def musdb_oracle(codec: str = 'encodec24') -> pd.DataFrame:
    """
    Get the list of files + some metadata for the dataset based on MUSDB18-HQ,
    providing hybrid, fully real and fully generated mixtures together with their oracle sources.
    """
    try:
        data_df = pd.read_csv(MUSDB_VERSIONS_PATHS[f'2_stems_mix_real_{codec}']['metadata'], sep=',')
    except FileNotFoundError as e:
        logging.error(f'Could not find MUSDB oracle metadata file '
                      f'{MUSDB_VERSIONS_PATHS[f"2_stems_mix_real_{codec}"]["metadata"]}.\n'
                      f'Make sure to set the database paths in "databases/config.py" '
                      f'and to run the data preparation script "databases/musdb_hq.py"')
        raise FileNotFoundError(e)

    data_df['codec'] = codec
    return data_df

def musdb_demucs(codec: str = 'encodec24') -> pd.DataFrame:
    """
    Get the list of files + some metadata for the dataset based on MUSDB18-HQ,
    providing hybrid, fully real and fully generated mixtures together with their sources extracted by ht-demucs.
    """
    try:
        data_df = pd.read_csv(MUSDB_VERSIONS_PATHS[f'2_stems_mix_real_{codec}_demucs']['metadata'], sep=',')
    except FileNotFoundError as e:
        logging.error(f'Could not find MUSDB oracle metadata files'
                      f'{MUSDB_VERSIONS_PATHS[f"2_stems_mix_real_{codec}_demucs"]["metadata"]}.\n'
                      f'Make sure to set the database paths in "databases/config.py" '
                      f'and to run the data preparation script "databases/musdb_hq.py"')
        raise FileNotFoundError(e)

    data_df['codec'] = codec
    return data_df
