
import os
import logging
from tqdm import tqdm

import pandas as pd
import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from config import MUSDB_VERSIONS_PATHS, DEMUCS_DOCKER_VOLUMES, DOCKER_DEMUCS_RUN_CMD, DEMUCS_RUN_CMD
from neuralcodecs.encodec import Encodec24

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 2000)

logging.basicConfig(level=logging.INFO)


def load_dataset(version='4_stems_real'):
    data_df = pd.read_csv(MUSDB_VERSIONS_PATHS[version]['metadata'], sep=',')
    return data_df.reset_index(drop=True)


def parse_stem_dataset(version='4_stems_real'):
    assert version in ('4_stems_real', '2_stems_real', '2_stems_encodec24')

    data_dir = MUSDB_VERSIONS_PATHS[version]['data']
    metadata_out_path = MUSDB_VERSIONS_PATHS[version]['metadata']

    logging.info(f'Parsing MUSDB-HQ {version} dataset from {data_dir}')
    logging.info(f'metadata will be saved in {metadata_out_path}')
    dataset_metadata = []

    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        for track_dir in tqdm(sorted([f.path for f in os.scandir(split_dir) if f.is_dir()])):
            for f_path in sorted([f.path for f in os.scandir(track_dir) if not f.name.startswith('.')]):
                try:
                    audio_md = AudioDecoder(f_path).metadata
                except :
                    logging.warning(f'Cannot load {f_path}, ignoring it!')
                    continue

                dataset_metadata.append({'track': os.path.basename(track_dir),
                                         'stem': os.path.splitext(os.path.basename(f_path))[0],
                                         'sr': audio_md.sample_rate,
                                         'dur': audio_md.duration_seconds,
                                         'n_channels': audio_md.num_channels,
                                         'fpath': f_path})

    dataset_metadata_df = pd.DataFrame(dataset_metadata)
    os.makedirs(os.path.dirname(metadata_out_path), exist_ok=True)
    dataset_metadata_df.to_csv(metadata_out_path, sep=',', header=True, index=False)

    return


def downmix_to_2stems():
    logging.info(f'Downmixing MUSDB-HQ dataset from 4 to 2 stems')
    data_4stems_df = load_dataset(version='4_stems_real')

    # Add 2-stem downmix mapping
    downmix_df = pd.DataFrame.from_dict({'stem': ['vocals', 'drums', 'bass', 'other'],
                                         'stem_out': ['vocals'] + ['accompaniment'] * 3})
    data_4stems_df = data_4stems_df.merge(downmix_df, on='stem', how='inner')
    # Prepare 2-stem out file paths
    data_4stems_df['fpath_out'] = data_4stems_df \
        .apply(lambda x: x['fpath']. \
               replace(MUSDB_VERSIONS_PATHS['4_stems_real']['data'], MUSDB_VERSIONS_PATHS['2_stems_real']['data']). \
               replace(x['stem'], x['stem_out']), axis=1)

    # Iterate over tracks and output stems
    for (_, _, fpath_out), stems_in_df in tqdm(data_4stems_df.groupby(['track', 'stem_out', 'fpath_out'])):
        assert len(stems_in_df[['sr', 'dur', 'n_channels']].drop_duplicates()) == 1, \
            f'metadata mismatch: {print(stems_in_df)}'

        # Downmix
        audio_out = torch.stack([AudioDecoder(fpath).get_all_samples().data
                                 for fpath in stems_in_df['fpath']]).sum(dim=0)
        sr = stems_in_df['sr'].unique()[0]

        # Export
        os.makedirs(os.path.dirname(fpath_out), exist_ok=True)
        AudioEncoder(samples=audio_out, sample_rate=sr).to_file(dest=fpath_out)

    return


def autoencode_with_encodec24(batch_size=4, device=None):
    logging.info(f'Auto-encoding MUSDB "2_stems_real" dataset with encodec24')
    data_df = load_dataset(version='2_stems_real')

    # Prepare file batches
    data_df['batch_id'] = data_df.index.values // batch_size
    data_df['fpath_out'] = data_df['fpath'] \
        .apply(lambda x: x.replace(MUSDB_VERSIONS_PATHS['2_stems_real']['data'],
                                   MUSDB_VERSIONS_PATHS['2_stems_encodec24']['data']))

    # Auto-encode and export
    encodec24 = Encodec24(device=device)
    for batch_id in tqdm(data_df['batch_id'].unique()):
        batch_df = data_df[data_df['batch_id'] == batch_id]
        encodec24.autoencode_from_files(fp_in=batch_df['fpath'].tolist(),
                                        fp_out=batch_df['fpath_out'].tolist())

    return


def generate_mixtures():
    metadata_out_path = MUSDB_VERSIONS_PATHS['2_stems_mix_real_encodec24']['metadata']

    logging.info(f'Generate MUSDB 2_stems_mix_real_encodec24 dataset')
    logging.info(f'metadata will be saved in {metadata_out_path}')

    # Get list of real and encoded stems
    real_stems_df = load_dataset(version='2_stems_real')
    fake_stems_df = load_dataset(version='2_stems_encodec24')
    real_stems_df['codec'] = 'real'
    fake_stems_df['codec'] = 'encodec24'
    all_stems_df = pd.concat([real_stems_df, fake_stems_df], ignore_index=True)

    # Define all mixing configurations
    mix_def = {
        'vg_ar': '((stem == "vocals") & (codec == "encodec24")) | ((stem == "accompaniment") & (codec == "real"))',
        'vr_ag': '((stem == "vocals") & (codec == "real"))      | ((stem == "accompaniment") & (codec == "encodec24"))',
        'vg_ag': '((stem == "vocals") & (codec == "encodec24")) | ((stem == "accompaniment") & (codec == "encodec24"))',
        'vr_ar': '((stem == "vocals") & (codec == "real"))      | ((stem == "accompaniment") & (codec == "real"))',
    }
    vocals_gain_db_list = [-12, -6, 0, 6, 12]

    # Generate all mixtures
    dataset_metadata = []
    for track, track_df in tqdm(all_stems_df.groupby('track')):

        for mix_type in mix_def:
            mix_df = track_df.query(mix_def[mix_type])
            assert len(mix_df) == 2, f'more than 2 stems found in {print(mix_df)}'

            vocals_fp = mix_df[mix_df['stem'] == 'vocals']['fpath'].tolist()[0]
            accomp_fp = mix_df[mix_df['stem'] == 'accompaniment']['fpath'].tolist()[0]

            vocals_dec = AudioDecoder(vocals_fp)
            accomp_dec = AudioDecoder(accomp_fp)

            assert ((vocals_dec.metadata.sample_rate == accomp_dec.metadata.sample_rate) &
                    (vocals_dec.metadata.num_channels == accomp_dec.metadata.num_channels)), \
                f'metadata mismatch for stems in {print(mix_df)}'

            vocals_sig = vocals_dec.get_all_samples().data
            accomp_sig = accomp_dec.get_all_samples().data
            sr = vocals_dec.metadata.sample_rate

            for vocals_gain in vocals_gain_db_list:

                mix_sig = vocals_sig * 10 ** (vocals_gain / 20) + accomp_sig
                mix_sig = mix_sig / mix_sig.abs().max()

                # Prepare output path
                data_out_dir = MUSDB_VERSIONS_PATHS['2_stems_mix_real_encodec24']['data']
                mix_fp = vocals_fp \
                    .replace(MUSDB_VERSIONS_PATHS['2_stems_real']['data'], data_out_dir) \
                    .replace(MUSDB_VERSIONS_PATHS['2_stems_encodec24']['data'], data_out_dir) \
                    .replace('vocals.wav', f'{mix_type}_{vocals_gain}dB.wav')

                # Export mix
                os.makedirs(os.path.dirname(mix_fp), exist_ok=True)
                AudioEncoder(samples=mix_sig, sample_rate=sr).to_file(dest=mix_fp)

                # Keep metadata
                audio_md = AudioDecoder(mix_fp).metadata
                dataset_metadata.append({'track': track,
                                         'mix_type': mix_type,
                                         'vocals_gain_db': vocals_gain,
                                         'sr': audio_md.sample_rate,
                                         'dur': audio_md.duration_seconds,
                                         'n_channels': audio_md.num_channels,
                                         'mix_path': mix_fp,
                                         'vocals_path': vocals_fp,
                                         'accomp_path': accomp_fp})

    # Export metadata file
    dataset_metadata_df = pd.DataFrame(dataset_metadata)
    os.makedirs(os.path.dirname(metadata_out_path), exist_ok=True)
    dataset_metadata_df.to_csv(metadata_out_path, sep=',', header=True, index=False)

    return


def prepare_demucs_commands():
    demucs_bash_out_path = f'{DEMUCS_DOCKER_VOLUMES["/data/output"]}/run_demucs.sh'
    metadata_out_path = MUSDB_VERSIONS_PATHS['2_stems_mix_real_encodec24_demucs']['metadata']

    logging.info(f'Preparing demucs commands to run inside your "docker-facebook-demucs" container instance:\n  '
                 f'A bash script will be saved in {demucs_bash_out_path}\n  '
                 f'1. Run the container with the following command:\n    '
                 f'{DOCKER_DEMUCS_RUN_CMD}\n  '
                 f'2. Run the bash script:\n    '
                 f'cd /data/output\n    '
                 f'./run_demucs.sh')

    mixes_df = load_dataset(version='2_stems_mix_real_encodec24')

    # Prepare out paths, according to demucs default conventions cf.
    # https://github.com/facebookresearch/demucs/blob/main/demucs/separate.py
    mixes_df['demucs_out_arg'] = mixes_df['mix_path'].apply(
        lambda x: os.path.dirname(x).replace(MUSDB_VERSIONS_PATHS['2_stems_mix_real_encodec24']['data'],
                                             MUSDB_VERSIONS_PATHS['2_stems_mix_real_encodec24_demucs']['data'])
    )
    mixes_df['demucs_out_dir'] = mixes_df.apply(
        lambda x: f'{x["demucs_out_arg"]}/htdemucs/{os.path.basename(os.path.splitext(x["mix_path"])[0])}', axis=1
    )
    mixes_df['vocals_path'] = mixes_df['demucs_out_dir'].apply(lambda x: f'{x}/vocals.wav')
    mixes_df['accomp_path'] = mixes_df['demucs_out_dir'].apply(lambda x: f'{x}/no_vocals.wav')

    # Build demucs commands
    mixes_df['demucs_cmd'] = mixes_df.apply(
        lambda x: DEMUCS_RUN_CMD\
            .format(demucs_in=x['mix_path'].replace(DEMUCS_DOCKER_VOLUMES['/data/input'], '/data/input'),
                    demucs_out=x['demucs_out_arg'].replace(DEMUCS_DOCKER_VOLUMES['/data/output'], '/data/output')),
        axis=1
    )

    # Export bash script
    with open(demucs_bash_out_path, 'w') as f:
        f.writelines([cmd + '\n' for cmd in mixes_df['demucs_cmd'].tolist()])
    os.chmod(demucs_bash_out_path, 0o755)

    # Export metadata file now (even if demucs has yet to be run)
    logging.info(f'metadata will be saved in {metadata_out_path}')
    mixes_df\
        .drop(columns=['demucs_out_arg', 'demucs_out_dir', 'demucs_cmd']) \
        .to_csv(metadata_out_path, sep=',', header=True, index=False)

    return


if __name__ == '__main__':

    parse_stem_dataset(version='4_stems_real')

    downmix_to_2stems()
    parse_stem_dataset(version='2_stems_real')

    autoencode_with_encodec24(batch_size=4)
    parse_stem_dataset(version='2_stems_encodec24')

    generate_mixtures()

    prepare_demucs_commands()

    # then run demucs from your docker container (cf. prepare_demucs_commands() instructions)
