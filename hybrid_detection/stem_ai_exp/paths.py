
import os
import sys

# Dirty python path manip
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')))
from hybrid_detection.config import EXP_ROOT_DIR


class StemAiExperimentPaths(object):

    root_dir = f'{EXP_ROOT_DIR}/stem_ai_detection'

    @classmethod
    def data_dir(cls, db, codec, chunk_dur, mkdir=False, clean=False):
        dir_ = f'{cls.root_dir}/data/{cls.db_config_str(db, codec)}/{cls.chunk_config_str(chunk_dur)}'
        if mkdir:
            os.makedirs(dir_, exist_ok=True)
        if clean:
            for f in os.listdir(dir_):
                if f.endswith('.parquet'):
                    os.remove(os.path.join(dir_, f))
        return dir_

    @classmethod
    def model_dir(cls, db, codec, stem, chunk_dur, random_seed, single_snr_band=False, mkdir=False):
        if not single_snr_band:
            snr_config = ''
        else:
            snr_config = '_single_snr_band'
        dir_ = f'{cls.root_dir}/models/{cls.db_config_str(db, codec)}_{stem}{snr_config}/{cls.chunk_config_str(chunk_dur)}/seed_{random_seed}'
        if mkdir:
            os.makedirs(dir_, exist_ok=True)
        return dir_

    @classmethod
    def db_config_str(cls, db, codec):
        return f'{db}_{codec}'

    @classmethod
    def chunk_config_str(cls, chunk_dur):
        if chunk_dur is None:
            return 'chunk_off'
        else:
            if int(chunk_dur) == chunk_dur:
                chunk_dur = str(int(chunk_dur))
            else:
                chunk_dur = str(round(chunk_dur, 2)).replace('.', 'dot')
            return f'chunk_{chunk_dur}s'
