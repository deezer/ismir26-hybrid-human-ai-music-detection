
import os

N_MAX_THREADS = 2
os.environ["OMP_NUM_THREADS"] = f'{N_MAX_THREADS}'

import argparse
import logging

import numpy as np
from scipy.special import expit
import joblib
import pandas as pd

from paths import StemAiExperimentPaths
from hybrid_detection.full_ai_exp.paths import FullAiExperimentPaths
import hybrid_detection.databases.loader as db_loader
from hybrid_detection.core.data.fakeprint import FakePrint
from hybrid_detection.core.data.energy import BandSplitEnergy
from hybrid_detection.utils.argparse_type import float_or_none
from hybrid_detection.utils.mp_wrapper import multiprocess_run


logging.basicConfig(level=logging.INFO)


def compute_hybrid_mix_data_batch(data_df: pd.DataFrame,
                                  batch_id: int,
                                  batch_model_dir: str,
                                  batch_out_dir: str) -> None:
    try:
        # Init full AI detection pipeline
        fakeprint_inst = FakePrint()
        full_ai_model = joblib.load(f'{batch_model_dir}/model.pkl')

        # Init BandSplitEnergy
        band_split_energy = BandSplitEnergy()

        # Run iteratively over files
        out_data = []
        for sample_id, sample_df in data_df.iterrows():

            # Get sample config
            mix_path = sample_df['mix_path']
            vocals_path = sample_df['vocals_path']
            accomp_path = sample_df['accomp_path']

            chunk_dur = sample_df['chunk_dur']
            chunk_ov = sample_df['chunk_ov']

            # Compute mix fakeprint
            fakeprints = fakeprint_inst.compute_from_file(mix_path, chunk_dur=chunk_dur, chunk_ov=chunk_ov)

            n_chunks = fakeprints.shape[0]
            chunk_ids = np.arange(n_chunks)
            if chunk_dur is not None:
                t_axis = fakeprint_inst.stft_chunker.t_axis_chunks(chunk_dur, chunk_ov, n_chunks)
            else:
                t_axis = np.zeros((1,))

            # Get full AI-model predictions
            full_ai_logit_score = full_ai_model.decision_function(fakeprints)
            full_ai_score = expit(full_ai_logit_score)

            # Get sources energy TF representation
            vocals_energy = band_split_energy.compute_from_file(vocals_path, chunk_dur=chunk_dur, chunk_ov=chunk_ov)
            accomp_energy = band_split_energy.compute_from_file(accomp_path, chunk_dur=chunk_dur, chunk_ov=chunk_ov)

            assert vocals_energy.shape == accomp_energy.shape, \
                'vocals and accomp BandSplitEnergy representation shape mismatch'
            assert vocals_energy.shape[0] == len(full_ai_score), \
                'fully fake prediction and BandSplitEnergy representation shape mismatch'

            # Store data together with file metadata
            sample_data = [sample_df.to_dict() |
                           {'chunk_id': chunk_id,
                            't': t_axis[chunk_id],
                            'full_ai_logit_score': full_ai_logit_score[chunk_id],
                            'full_ai_score': full_ai_score[chunk_id],
                            'vocals_energy': vocals_energy[chunk_id, :].copy(),
                            'accomp_energy': accomp_energy[chunk_id, :].copy()}
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
    parser.add_argument("--db",  help="select database", choices=['musdb_oracle', 'musdb_demucs'], type=str)
    parser.add_argument("--codec", help="select neural codec", choices=['encodec24'], type=str, default='encodec24')
    parser.add_argument("--chunk_dur", help="chunk duration (in sec)", type=float_or_none, default=None)
    parser.add_argument("--n_batches", help="num of batches to split the full dataset", type=int, default=100)
    parser.add_argument("--n_workers", help="num of worker processes", type=int, default=None)

    args = parser.parse_args()

    logging.info(f'Starting the data computation for stem AI detection with config:\n{args}')

    # Prepare output data dir
    out_dir = StemAiExperimentPaths().data_dir(args.db, args.codec, args.chunk_dur, mkdir=True, clean=True)
    logging.info(f'Output data will be exported as parquet part files in {out_dir}')

    # Get full-ai detection model
    full_ai_model_dir = FullAiExperimentPaths().model_dir('fma', args.codec, args.chunk_dur)
    logging.info(f'Selected full AI-detection model: {out_dir}')

    # Get list of files to process
    if args.db == 'musdb_oracle':
        database_df = db_loader.musdb_oracle()
    elif args.db == 'musdb_demucs':
        database_df = db_loader.musdb_demucs()
    else:
        raise ValueError(f"Unknown database {args.db}")

    database_df = database_df[database_df['codec'] == args.codec].reset_index(drop=True)
    assert len(database_df) > 0, 'no files found for db={args.db} and codec={args.codec}'

    database_df['chunk_dur'] = args.chunk_dur
    database_df['chunk_ov'] = 0.9  # Fix chunk overlap

    # Split into batches and run parallel processing
    database_df['batch_id'] =  database_df.index.values % args.n_batches
    batch_ids = database_df["batch_id"].unique()

    multiprocess_run(
        func=compute_hybrid_mix_data_batch,
        worker_args=[(database_df[database_df['batch_id'] == batch_id], int(batch_id), full_ai_model_dir, out_dir)
                      for batch_id in batch_ids],
        n_workers=args.n_workers or int(os.cpu_count() / N_MAX_THREADS)
    )
