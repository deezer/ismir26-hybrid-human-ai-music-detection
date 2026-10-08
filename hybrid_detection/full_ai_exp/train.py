
import os
import gc
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

from paths import FullAiExperimentPaths
from hybrid_detection.core.eval import tpr_fpr
from hybrid_detection.utils.argparse_type import float_or_none


logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", help="training database", choices=['fma'], type=str, default='fma')
    parser.add_argument("--codec", help="neural codec", choices=['encodec24'], type=str, default='encodec24')
    parser.add_argument("--chunk_dur", help="chunk duration (in sec)", type=float_or_none, default=None)
    parser.add_argument("--train_ratio", help="train proportion", type=float, default=0.7)
    parser.add_argument("--random_seed", help="for reproducibility", type=int, default=42)
    args = parser.parse_args()

    logging.info(f'Starting the training of the full AI music detector with config:\n{args}')

    exp_paths = FullAiExperimentPaths()

    ai_data_dir = exp_paths.data_dir(args.db, args.codec, args.chunk_dur)
    real_data_dir = exp_paths.data_dir(args.db, 'real', args.chunk_dur)

    out_dir = exp_paths.model_dir(args.db, args.codec, args.chunk_dur, mkdir=True)
    out_model_path = os.path.join(out_dir, 'model.pkl')
    out_predictions_path = os.path.join(out_dir, 'train_test_predictions.parquet')

    # Load dataset and split train/test
    logging.info(f'Loading fakeprint data from {real_data_dir} and {ai_data_dir}')

    ai_df = pd.read_parquet(ai_data_dir)
    ai_df['label'] = 1
    real_df = pd.read_parquet(real_data_dir)
    real_df['label'] = 0

    full_data_df = pd.concat([ai_df, real_df], ignore_index=True)

    # Split into train / test
    split_df = full_data_df[['fname']] \
        .drop_duplicates() \
        .sort_values(by=['fname']) \
        .sample(frac=args.train_ratio, replace=False, random_state=args.random_seed)
    split_df['train_set'] = True

    full_data_df = full_data_df \
        .merge(split_df, how='left', on='fname') \
        .fillna({'train_set': False})

    stats_df = full_data_df \
        .groupby(['train_set', 'label', 'chunk_dur'], dropna=False) \
        .agg({'fname': 'nunique', 'chunk_id': len}) \
        .rename(columns={'fname': 'n_unique_files', 'chunk_id': 'n_chunks'})

    logging.info(f"\n{stats_df}")

    # Train model
    logging.info(f'Training logistic regression model and saving to {out_model_path}')

    X = np.vstack(full_data_df[full_data_df['train_set']]['fakeprint'].to_list())
    y = np.array(full_data_df[full_data_df['train_set']]['label'].to_list())

    regressor = LogisticRegression(class_weight="balanced", random_state=args.random_seed)
    regressor.fit(X, y)

    joblib.dump(regressor, out_model_path)

    # Free some memory
    X, y = None, None
    gc.collect()

    # Run predictions and export
    logging.info(f'Running predictions and saving to {out_predictions_path}')

    X = np.vstack(full_data_df['fakeprint'].to_list())
    ai_score = regressor.predict_proba(X)[:, 1]

    out_data_df = full_data_df.drop(columns=['fakeprint'])
    out_data_df['full_ai_score'] = ai_score

    out_data_df.to_parquet(out_predictions_path, index=False)

    # Display eval
    eval_df = out_data_df \
        .groupby(['train_set', 'chunk_dur'], dropna=False) \
        .apply(lambda x: tpr_fpr(x['label'], x['full_ai_score'], as_pandas=True))

    logging.info(f"\n{eval_df}")
