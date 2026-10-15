
# Reproduce experiments from Section 4. FULLY GENERATED AI MUSIC DETECTION

# Training global & local binary detectors on FMA-medium dataset

for chunk_dur in "None" 10 5 2 1; do

  for codec in "real" "encodec24" "same-l"; do
    uv run python ./compute_training_data.py --db "fma" --codec "$codec" --chunk_dur "$chunk_dur" --n_workers 16 --n_batches 100
  done

  for codec in "encodec24" "same-l"; do
    uv run python ./train.py --db "fma" --codec "$codec" --chunk_dur "$chunk_dur"
  done

done

# Applying the global binary detector to hybrid mixtures from MUSDB18-HQ dataset

for codec in "encodec24" "same-l"; do
  uv run python ./predict.py --db "musdb_mixtures" --db_train "fma" --codec "$codec" --chunk_dur "None" --n_workers 16 --n_batches 100
done