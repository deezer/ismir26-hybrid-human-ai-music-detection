
import os


METADATA_DIR = os.path.join(os.path.dirname(__file__), 'metadata')

"""
Download FMA-medium from https://github.com/mdeff/fma
Then set:
  - FMA_REAL_DIR: path to your copy of FMA-medium dataset
  - FMA_ENCODEC_DIR: path where the autoencoded version of the dataset will be stored when running ./fma_medium.py
"""

FMA_REAL_DIR = None  # e.g. '/path/to/FMA/fma_medium'
FMA_ENCODEC_DIR = None  # e.g. '/path/to/ismir26_hybrid/datasets/fma_medium_encodec24'

assert FMA_REAL_DIR is not None, "Please set FMA_REAL_DIR environment variable"
assert FMA_ENCODEC_DIR is not None, "Please set FMA_ENCODEC_DIR environment variable"

FMA_VERSIONS_PATHS = {
    # Initial dataset
    'real': {'data': FMA_REAL_DIR,
             'metadata': f'{METADATA_DIR}/fma_medium_real.csv'},
    # Auto-encoded with encodec
    'encodec24': {'data': FMA_ENCODEC_DIR,
                  'metadata': f'{METADATA_DIR}/fma_medium_encodec24.csv'}
}


"""
# Download MUSDB18-HQ dataset from https://zenodo.org/records/3338373
Then set:
  - MUSDB_4STEMS_DIR: path to your copy of musdb_hq (4 stems) dataset
  - MUSDB_2STEMS_DIR: path where the 2-stem datasets (real stems, autoencoded stems, hybrid mixtures and demucs separation)
                   will be stored when running ./musdb_hq.py
"""

MUSDB_4STEMS_DIR = None  # e.g. '/path/to/musdb_hq'
MUSDB_2STEMS_DIR = None  # e.g. '/path/to/ismir26_hybrid/datasets/musdb_hq_2stems'

assert MUSDB_4STEMS_DIR is not None, "Please set MUSDB_4STEMS_DIR environment variable"
assert MUSDB_2STEMS_DIR is not None, "Please set MUSDB_2STEMS_DIR environment variable"

MUSDB_VERSIONS_PATHS = {
    # Initial dataset
    '4_stems_real': {'data': MUSDB_4STEMS_DIR,
                     'metadata': f'{METADATA_DIR}/musdb_4stems_real.csv'},
    # Downmixed to 2 stems
    '2_stems_real': {'data': f'{MUSDB_2STEMS_DIR}/real',
                     'metadata': f'{METADATA_DIR}/musdb_2stems_real.csv'},
    # Downmixed to 2 stems, then auto-encoded with encodec
    '2_stems_encodec24': {'data': f'{MUSDB_2STEMS_DIR}/encodec24',
                          'metadata': f'{METADATA_DIR}/musdb_2stems_encodec24.csv'},
    # Hybrid mixtures
    '2_stems_mix_real_encodec24': {'data': f'{MUSDB_2STEMS_DIR}/real_encodec24_mix',
                                   'metadata': f'{METADATA_DIR}/musdb_2stems_mix_real_encodec24.csv'},
    # Demucs separation of hybrid mixtures
    '2_stems_mix_real_encodec24_demucs': {'data': f'{MUSDB_2STEMS_DIR}/real_encodec24_mix_demucs',
                                          'metadata': f'{METADATA_DIR}/musdb_2stems_mix_real_encodec24_demucs.csv'}
}


"""
Clone demucs docker repo from https://github.com/xserrat/docker-facebook-demucs
then build the image: `make build`
Note that it requires cuda > 12.6.2 to run on GPU.
Then set below:
  - DEMUCS_REPO: the path to your clone of docker-facebook-demucs repo
"""

DEMUCS_REPO = None  # e.g. '/path/to/code/docker-facebook-demucs'

assert DEMUCS_REPO is not None, "Please set DEMUCS_REPO environment variable"

DEMUCS_DOCKER_VOLUMES = {
    '/data/input': MUSDB_VERSIONS_PATHS['2_stems_mix_real_encodec24']['data'],
    '/data/output': MUSDB_VERSIONS_PATHS['2_stems_mix_real_encodec24_demucs']['data'],
    '/data/models': f'{DEMUCS_REPO}/models'
}

DOCKER_DEMUCS_RUN_CMD = (
    'docker run --rm -it'
	' --name=demucs'
	' --gpus all'
	f' -v "{DEMUCS_DOCKER_VOLUMES["/data/input"]}":/data/input'
	f' -v "{DEMUCS_DOCKER_VOLUMES["/data/output"]}":/data/output'
	f' -v "{DEMUCS_DOCKER_VOLUMES["/data/models"]}":/data/models'
	' xserrat/facebook-demucs:latest'
    ' bash'
)

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
