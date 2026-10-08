# Detection of AI-generated stems within hybrid human-AI music

Code repository to reproduce the experiments presented in our [ISMIR 2026 paper](https://arxiv.org/abs/2607.26874), 
F. Rigaud, G. Meseguer-Brocal, B. Martin, R. Hennequin

> This paper presents, to the best of our knowledge, the first study on detecting human-AI hybrid music tracks created by mixing human-produced and AI-generated stems. Building on recent work showing that AI music detectors can identify decoder-related artifacts in fully generated music, we investigate whether such artifacts remain detectable at the stem level after mixing. Using MUSDB18-HQ database in a two-stem vocals + accompaniment setting, we simulate hybrid mixtures by autoencoding individual stems with a neural codec. We compare two strategies combining AI-generated mix detection and source separation. A naive sequential pipeline, where source separation is followed by detection on separated sources, confirms that artifacts associated with an AI-generated stem are not reliably recovered by generic source separation systems. We therefore propose a parallel architecture in which source separation is only used to estimate source-relative energy within the mixture. We then train simple stem-specific binary classifiers that take as input the generated mix prediction together with the relative energy of the target stem on short audio chunks. Averaging chunk-level predictions yields encouraging track-level results, highlighting the potential of such approaches for detecting AI-generated stems in hybrid music.


### Quick install notes

- All required python packages are listed in `requirements.txt`. Install them in your preferred virtual environment or environment manager.
- `demucs` music source separation is based on the Docker image built from [docker-facebook-demucs](https://github.com/xserrat/docker-facebook-demucs) and requires CUDA version `12.6.2` or later.

### Repository organisation

- `databases/`: Scripts used to generate the datasets required to reproduce all experiments. The results can be analyzed using `./databases_stats.ipynb`.
- `core/`: Classes used to compute input representations for the models.
- `full_ai_exp/`: Scripts used to train fully AI-generated music detectors, both global detectors that produce track-level predictions and local detectors that produce chunk-level predictions. The results can be analyzed using `./fully_ai_detection.ipynb`.
- `stem_ai_exp/`: Scripts used to train AI-generated stem detectors, reusing the fully AI-generated music detectors trained in `./full_ai_exp/`. The results can be analyzed using `./stem_ai_detection.ipynb`.
- `utils/`: Utility functions, including tools for multiprocessing batches of files.

### Databases preparation

Before running the experiments, prepare the audio datasets required for:

- Training the fully AI-generated music detectors from [FMA-medium](https://github.com/mdeff/fma) dataset + [encodec](https://huggingface.co/docs/transformers/model_doc/encodec) auto-encoding.
- Training the AI-generated stems detectors from [MUSDB18-HQ](https://zenodo.org/records/3338373) dataset + [encodec](https://huggingface.co/docs/transformers/model_doc/encodec) auto-encoding + [demucs](https://github.com/facebookresearch/demucs) source separation.

Set the required data paths in `databases/config.py`:
- `FMA_REAL_DIR`: path to your copy of FMA-medium dataset
- `FMA_ENCODEC_DIR`: path where the autoencoded version of the dataset will be stored
- `MUSDB_4STEMS_DIR`: path to your copy of musdb_hq (4 stems) dataset
- `MUSDB_2STEMS_DIR`: path where the 2-stem datasets (real stems, autoencoded stems, hybrid mixtures and demucs separation) will be stored
- `DEMUCS_REPO`: the path to your clone of docker-facebook-demucs repo

Then run `databases/fma_medium.py` and `databases/musdb_hq.py`.

The notebook `databases_stats.ipynb` can be used to display statistics about the generated datasets.

### Fully generated AI music detection: Training & evaluating

First, set `EXP_ROOT_DIR` variable in `config.py`. 
This directory is used to store all training data, models, and predictions generated during the experiments.

Then, in `full_ai_exp/` execute `./run_exp.sh` in order to reproduce all the experiments presented in Section 4 of the paper:
- Training global & local binary detectors on FMA-medium dataset.
- Applying the global binary detector to hybrid mixtures from MUSDB18-HQ dataset.

The notebook `full_ai_detection.ipynb` can be used to analyze the evaluation results and reproduce the corresponding tables and figures from the paper.

### AI-generated stem detection: Training & evaluation

In `stem_ai_exp/` execute `./run_exp.sh` in order to reproduce all the experiments presented in Sections 5 & 6 of the paper.
- Naive baseline: Applying the global binary detector directly to the separated vocals and accompaniment stems from MUSDB18-HQ hybrid dataset.
- Proposed approach: Training stem-specific detectors for vocals and accompaniment. These detectors take as input the local binary detection scores for the mixture and the local SNR of the target stem, using either the `oracle` or `demucs` sources.

Then, the notebook `stem_ai_detection.ipynb` can be used to analyze the evaluation results and reproduce the corresponding tables and figures from the paper.
