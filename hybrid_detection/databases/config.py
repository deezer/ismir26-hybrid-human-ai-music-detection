
import os


METADATA_DIR = os.path.join(os.path.dirname(__file__), 'metadata')

"""
Download FMA-medium from https://github.com/mdeff/fma
Then set:
  - FMA_REAL_DIR: path to your copy of FMA-medium dataset
  - FMA_AUTOENCODED_DIR: path where the autoencoded versions of the dataset will be stored when running ./fma_medium.py
"""

FMA_REAL_DIR = '/data/nfs/analysis/audio_database/genres/FMA/fma_medium'  # e.g. '/path/to/FMA/fma_medium'
FMA_AUTOENCODED_DIR = '/data/nfs/analysis/frigaud/ismir26/datasets/fma_medium_autoencoded'  # e.g. '/path/to/ismir26_hybrid/datasets/fma_medium_autoencoded'

assert FMA_REAL_DIR is not None, "Please set FMA_REAL_DIR environment variable"
assert FMA_AUTOENCODED_DIR is not None, "Please set FMA_ENCODED_DIR environment variable"

FMA_VERSIONS_PATHS = {
    # Initial dataset
    'real': {'data': FMA_REAL_DIR,
             'metadata': f'{METADATA_DIR}/fma_medium/real.csv'},
    # Auto-encoded with encodec
    'encodec24': {'data': f'{FMA_AUTOENCODED_DIR}/encodec24',
                  'metadata': f'{METADATA_DIR}/fma_medium/encodec24.csv'},
    # Auto-encodec with SAME
    'same-l': {'data': f'{FMA_AUTOENCODED_DIR}/same-l',
               'metadata': f'{METADATA_DIR}/fma_medium/same-l.csv'}
}


"""
# Download MUSDB18-HQ dataset from https://zenodo.org/records/3338373
Then set:
  - MUSDB_4STEMS_DIR: path to your copy of musdb_hq (4 stems) dataset
  - MUSDB_2STEMS_DIR: path where the 2-stem datasets (real stems, autoencoded stems, hybrid mixtures and demucs separation)
                   will be stored when running ./musdb_hq.py
"""

MUSDB_4STEMS_DIR = '/data/nfs/analysis/audio_database/separated_sources/musdb_hq'  # e.g. '/path/to/musdb_hq'
MUSDB_2STEMS_DIR = '/data/nfs/analysis/frigaud/ismir26/datasets/musdb_hq_2stems'  # e.g. '/path/to/ismir26_hybrid/datasets/musdb_hq_2stems'

assert MUSDB_4STEMS_DIR is not None, "Please set MUSDB_4STEMS_DIR environment variable"
assert MUSDB_2STEMS_DIR is not None, "Please set MUSDB_2STEMS_DIR environment variable"

MUSDB_VERSIONS_PATHS = {
    # Initial dataset
    '4_stems_real': {'data': MUSDB_4STEMS_DIR,
                     'metadata': f'{METADATA_DIR}/musdb/4stems_real.csv'},
    # Downmixed to 2 stems
    '2_stems_real': {'data': f'{MUSDB_2STEMS_DIR}/real',
                     'metadata': f'{METADATA_DIR}/musdb/2stems_real.csv'},
    # -------- Auto-encoded with encodec  --------
    '2_stems_encodec24': {'data': f'{MUSDB_2STEMS_DIR}/encodec24',
                          'metadata': f'{METADATA_DIR}/musdb/2stems_encodec24.csv'},
    # Hybrid mixtures
    '2_stems_mix_real_encodec24': {'data': f'{MUSDB_2STEMS_DIR}/real_encodec24_mix',
                                   'metadata': f'{METADATA_DIR}/musdb/2stems_mix_real_encodec24.csv'},
    # Demucs separation of hybrid mixtures
    '2_stems_mix_real_encodec24_demucs': {'data': f'{MUSDB_2STEMS_DIR}/real_encodec24_mix_demucs',
                                          'metadata': f'{METADATA_DIR}/musdb/2stems_mix_real_encodec24_demucs.csv'},
    # -------- Auto-encoded with same-l  --------
    '2_stems_same-l': {'data': f'{MUSDB_2STEMS_DIR}/same-l',
                       'metadata': f'{METADATA_DIR}/musdb/2stems_same-l.csv'},
    # Hybrid mixtures
    '2_stems_mix_real_same-l': {'data': f'{MUSDB_2STEMS_DIR}/real_same-l_mix',
                                   'metadata': f'{METADATA_DIR}/musdb/2stems_mix_real_same-l.csv'},
    # Demucs separation of hybrid mixtures
    '2_stems_mix_real_same-l_demucs': {'data': f'{MUSDB_2STEMS_DIR}/real_same-l_mix_demucs',
                                       'metadata': f'{METADATA_DIR}/musdb/2stems_mix_real_same-l_demucs.csv'}
}


"""
Clone demucs docker repo from https://github.com/xserrat/docker-facebook-demucs
then build the image: `make build`
Note that it requires cuda > 12.6.2 to run on GPU.
Then set below:
  - DEMUCS_REPO: the path to your clone of docker-facebook-demucs repo
"""

DEMUCS_REPO = '/data/home/deezer-gpu3/frigaud/code/docker-facebook-demucs'  # e.g. '/path/to/code/docker-facebook-demucs'

assert DEMUCS_REPO is not None, "Please set DEMUCS_REPO environment variable"

DEMUCS_DOCKER_VOLUMES = {
    codec: {
        '/data/input': MUSDB_VERSIONS_PATHS[f'2_stems_mix_real_{codec}']['data'],
        '/data/output': MUSDB_VERSIONS_PATHS[f'2_stems_mix_real_{codec}_demucs']['data'],
        '/data/models': f'{DEMUCS_REPO}/models'
    }
    for codec in ['encodec24', 'same-l']
}

DOCKER_DEMUCS_RUN_CMD = {
    codec: (
        'docker run --rm -it'
        ' --name=demucs'
        ' --gpus all'
        f' -v "{DEMUCS_DOCKER_VOLUMES[codec]["/data/input"]}":/data/input'
        f' -v "{DEMUCS_DOCKER_VOLUMES[codec]["/data/output"]}":/data/output'
        f' -v "{DEMUCS_DOCKER_VOLUMES[codec]["/data/models"]}":/data/models'
        ' xserrat/facebook-demucs:latest'
        ' bash')
    for codec in ['encodec24', 'same-l']
}

DEMUCS_RUN_CMD = (
    'python3 -m demucs'
    ' -n htdemucs'
    ' -j 1'
    ' --two-stems vocals'
    ' --clip-mode clamp'
	' --shifts 1'
	' --overlap 0.25'
    ' --out "{demucs_out}"'
	' "{demucs_in}"'
)
