
import os

N_MAX_THREADS = 2
os.environ["OMP_NUM_THREADS"] = f'{N_MAX_THREADS}'

import argparse
import logging

import numpy as np
from scipy.special import expit
import pandas as pd
import joblib

from paths import FullAiExperimentPaths
import hybrid_detection.databases.loader as db_loader
from hybrid_detection.core.data.fakeprint import FakePrint
from hybrid_detection.utils.argparse_type import float_or_none
from hybrid_detection.utils.mp_wrapper import multiprocess_run


logging.basicConfig(level=logging.INFO)


def predict_batch(data_df: pd.DataFrame,
                  batch_id: int,
                  batch_model_dir: str,
                  batch_out_dir: str) -> None:

    try:
        # Init pipeline
        fakeprint_inst = FakePrint()
        model = joblib.load(f'{batch_model_dir}/model.pkl')

        # Run iteratively over files
        out_data = []
        for sample_id, sample_df in data_df.iterrows():

            # Compute fakeprints
            f_path = sample_df['fpath']
            chunk_dur = sample_df['chunk_dur']
            chunk_ov = sample_df['chunk_ov']

            fakeprints = fakeprint_inst.compute_from_file(f_path, chunk_dur=chunk_dur, chunk_ov=chunk_ov)

            n_chunks = fakeprints.shape[0]
            chunk_ids = np.arange(n_chunks)
            if chunk_dur is not None:
                t_axis = fakeprint_inst.stft_chunker.t_axis_chunks(chunk_dur, chunk_ov, n_chunks)
            else:
                t_axis = np.zeros((1,))

            # Run model predictions
            full_ai_logit_score = model.decision_function(fakeprints)
            full_ai_score = expit(full_ai_logit_score)

            # Store predictions together with file metadata
            sample_data = [sample_df.to_dict() |
                           {'chunk_id': chunk_id,
                            't': t_axis[chunk_id],
                            'full_ai_logit_score': full_ai_logit_score[chunk_id],
                            'full_ai_score': full_ai_score[chunk_id]}
                           for chunk_id in chunk_ids]
            out_data += sample_data

        # Export data to parquet file
        out_part = f'0000{batch_id}'[-5:]
        out_fp = f'{batch_out_dir}/part-{out_part}.parquet'
        pd.DataFrame(out_data).to_parquet(out_fp, index=False)

    except Exception:
        logging.exception(f'Worker failed while processing batch {batch_id}')
        raise


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", help="prediction database", choices=['musdb_mixtures', 'musdb_demucs_sources'], type=str, default='musdb_mixtures')
    parser.add_argument("--db_train", help="training database", choices=['fma'], type=str, default='fma')
    parser.add_argument("--codec", help="neural codec", choices=['encodec24'], type=str, default='encodec24')
    parser.add_argument("--chunk_dur", help="chunk duration (in sec)", type=float_or_none, default=None)
    parser.add_argument("--n_batches", help="num of batches to split the full dataset", type=int, default=100)
    parser.add_argument("--n_workers", help="num of worker processes", type=int, default=None)

    args = parser.parse_args()

    logging.info(f'Starting full AI prediction with config:\n{args}')

    exp_paths = FullAiExperimentPaths()

    # Get detector dir
    model_dir = exp_paths.model_dir(args.db_train, args.codec, args.chunk_dur)

    # Prepare output data dir
    out_dir = exp_paths.predictions_dir(args.db, args.db_train, args.codec, args.chunk_dur, mkdir=True, clean=True)
    logging.info(f'Output data will be exported as parquet part files in {out_dir}')

    # Get list of files to process
    if args.db == 'musdb_mixtures':
        database_df = db_loader.musdb_demucs() \
            .drop(columns=['vocals_path', 'accomp_path']) \
            .rename(columns={'mix_path': 'fpath'})

    elif args.db == 'musdb_demucs_sources':
        db_vocals_df = db_loader.musdb_demucs()
        db_vocals_df['target_stem'] = 'vocals'
        db_vocals_df['target_gain_db'] = db_vocals_df['vocals_gain_db']
        db_vocals_df['fpath'] = db_vocals_df['vocals_path']

        db_accomp_df = db_loader.musdb_demucs()
        db_accomp_df['target_stem'] = 'accomp'
        db_accomp_df['target_gain_db'] = - db_accomp_df['vocals_gain_db']
        db_accomp_df['fpath'] = db_accomp_df['accomp_path']

        database_df = pd.concat([db_vocals_df, db_accomp_df], ignore_index=True) \
            .drop(columns=['mix_path', 'vocals_path', 'accomp_path'])

        # Add target stem label
        label_map = {'vr_ar': {'vocals': 0, 'accomp': 0},
                     'vg_ar': {'vocals': 1, 'accomp': 0},
                     'vr_ag': {'vocals': 0, 'accomp': 1},
                     'vg_ag': {'vocals': 1, 'accomp': 1}}

        database_df['label'] = database_df.apply(lambda x: label_map[x['mix_type']][x['target_stem']], axis=1)


    else:
        raise ValueError(f"Unknown database {args.db}")

    database_df = database_df[database_df['codec'] == args.codec].reset_index(drop=True)
    assert len(database_df) > 0, 'no files found for db={args.db} and codec={args.codec}'

    database_df['chunk_dur'] = args.chunk_dur
    database_df['chunk_ov'] = 0.95  # Fix chunk overlap to 95 %

    # Split into batches and run parallel processing
    database_df['batch_id'] =  database_df.index.values % args.n_batches
    batch_ids = database_df["batch_id"].unique()

    multiprocess_run(
        func=predict_batch,
        worker_args=[(database_df[database_df['batch_id'] == batch_id], int(batch_id), model_dir, out_dir)
                     for batch_id in batch_ids],
        n_workers=args.n_workers or int(os.cpu_count() / N_MAX_THREADS)
    )
