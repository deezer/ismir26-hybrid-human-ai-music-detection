
import os

N_MAX_THREADS = 2
os.environ["OMP_NUM_THREADS"] = f'{N_MAX_THREADS}'

import argparse
import logging

import numpy as np
import pandas as pd

from paths import FullAiExperimentPaths
import hybrid_detection.databases.loader as db_loader
from hybrid_detection.core.data.fakeprint import FakePrint
from hybrid_detection.utils.argparse_type import float_or_none
from hybrid_detection.utils.mp_wrapper import multiprocess_run


logging.basicConfig(level=logging.INFO)


def compute_fakeprint_batch(data_df: pd.DataFrame,
                            batch_id: int,
                            batch_out_dir: str,
                            max_chunks_per_file: int | None = 50) -> None:

    try:
        # Fix random seed for 'n_max_chunks' subsampling option
        rng = np.random.default_rng(42)

        # Init FakePrint transform
        fakeprint_inst = FakePrint()

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

            # Subsample
            if max_chunks_per_file is not None and n_chunks > max_chunks_per_file:
                chunk_ids = np.sort(rng.choice(chunk_ids, size=max_chunks_per_file, replace=False))

            # Store fakeprints together with file metadata
            sample_data = [sample_df.to_dict() |
                           {'chunk_id': chunk_id,
                            'fakeprint': fakeprints[chunk_id, :].copy()}
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
    parser.add_argument("--db",  help="select database", choices=['fma'], type=str, default='fma')
    parser.add_argument("--codec", help="select neural codec", choices=['real', 'encodec24'], type=str, default='encodec24')
    parser.add_argument("--chunk_dur", help="chunk duration (in sec)", type=float_or_none, default=None)
    parser.add_argument("--n_batches", help="num of batches to split the full dataset", type=int, default=100)
    parser.add_argument("--n_workers", help="num of worker processes", type=int, default=None)

    args = parser.parse_args()

    logging.info(f'Starting the fakeprint computation with config:\n{args}')

    # Prepare output data dir
    exp_paths = FullAiExperimentPaths()
    out_dir = exp_paths.data_dir(args.db, args.codec, args.chunk_dur, mkdir=True, clean=True)
    logging.info(f'Output data will be exported as parquet part files in {out_dir}')

    # Get list of files to process
    if args.db == 'fma':
        database_df = db_loader.fma()
        database_df = database_df[database_df['codec'] == args.codec].reset_index(drop=True)
        assert len(database_df) > 0, 'no files found for db={args.db} and codec={args.codec}'

    else:
        raise ValueError(f"Unknown database {args.db}")

    database_df['chunk_dur'] = args.chunk_dur
    database_df['chunk_ov'] = 0.5  # Fix chunk overlap to 50%

    # Split into batches and run parallel processing
    database_df['batch_id'] =  database_df.index.values % args.n_batches
    batch_ids = database_df["batch_id"].unique()

    multiprocess_run(
        func=compute_fakeprint_batch,
        worker_args=[(database_df[database_df['batch_id'] == batch_id], int(batch_id), out_dir)
                      for batch_id in batch_ids],
        n_workers=args.n_workers or int(os.cpu_count() / N_MAX_THREADS)
    )
