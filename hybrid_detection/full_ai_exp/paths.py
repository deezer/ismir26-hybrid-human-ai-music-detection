
import os
import sys

# Dirty python path manip
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')))
from hybrid_detection.config import EXP_ROOT_DIR


class FullAiExperimentPaths(object):

    root_dir = f'{EXP_ROOT_DIR}/full_ai_detection'

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
    def model_dir(cls, db, codec, chunk_dur, mkdir=False):
        dir_ = f'{cls.root_dir}/models/{cls.db_config_str(db, codec)}/{cls.chunk_config_str(chunk_dur)}'
        if mkdir:
            os.makedirs(dir_, exist_ok=True)
        return dir_

    @classmethod
    def predictions_dir(cls, db_predict, db_train, codec, chunk_dur, mkdir=False, clean=False):
        model_dir = cls.model_dir(db_train, codec, chunk_dur, mkdir=mkdir)
        dir_ = f'{model_dir}/predictions/{db_predict}'
        if mkdir:
            os.makedirs(dir_, exist_ok=True)
        if clean:
            for f in os.listdir(dir_):
                if f.endswith('.parquet'):
                    os.remove(os.path.join(dir_, f))
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
