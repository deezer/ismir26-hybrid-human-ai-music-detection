

for codec in "encodec24" "same-l"; do

  # Reproduce experiments from Section 5. DETECTION OF AI-GENERATED STEMS: A NAIVE APPROACH
  # Apply the global binary detector directly on ht-demucs separated stems from MUSDB18-HQ hybrid mixtures

  uv run python ../full_ai_exp/predict.py --db "musdb_demucs_sources" --db_train "fma" --codec "$codec" --chunk_dur "None" --n_workers 16 --n_batches 100


  # Reproduce experiments from Section 6. PROPOSED APPROACH
  # Train stem-specific detectors taking as input the local binary detector scores & the target stem local SNR
  # Two scenarios: SNR from 'oracle' or 'demucs' sources

  for chunk_dur in 5 2; do
    for db in "musdb_oracle" "musdb_demucs"; do
      uv run python ./compute_training_data.py --db "$db" --codec "$codec" --chunk_dur "$chunk_dur" --n_workers 16 --n_batches 100
    done
  done

  for stem in "vocals" "accomp"; do
    for chunk_dur in 5 2; do
      for db in "musdb_oracle" "musdb_demucs"; do
        for rs in 12 31 42 54 76 98 47 77 88 666; do
          uv run python ./train.py --stem "$stem" --db "$db" --codec "$codec" --chunk_dur "$chunk_dur" --random_seed "$rs"
        done
      done
    done
  done

  # Reproduce the qualitative analysis on posterior probas in the simple case of a single SNR band

  for stem in "vocals" "accomp"; do
    for chunk_dur in 5 2; do
      for db in "musdb_oracle" "musdb_demucs"; do
        uv run python ./train.py --stem "$stem" --db "$db" --codec "$codec" --chunk_dur "$chunk_dur" --random_seed 42 --train_ratio 1 --single_snr_band
      done
    done
  done

done