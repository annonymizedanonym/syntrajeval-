This repository accompanies the submitted paper and provides:

- Preprocessing and temporal aggregation for Geolife trajectories (weekly/daily/hourly):
  - You first need to download the Geolife dataset: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/
  - Then set the path in the Data/prep_geolife/prep_geolife.py script
- Training for three generative models: LSTM‑TrajGAN, ExGAN, TrajGDM
- Utility evaluation via the syntraj_utileval package:
  - Notebooks/caseAB.ipynb (to reproduce tables 2,3,9 and 10)
  - Notebooks/maup_sensitivity_analysis.ipynb (to reproduce table 7)
- Privacy evaluation on LSTM-TrajGAN and exGAN via two notebooks (eval_privacy/).
  - MIATrajGAN.ipynb executes our MIA attack on the LSTM-TrajGAN model
  - MIAexGAN.ipynb executes our MIA attack on the exGAN model

This repository provides 3 environements for reproducibility:

- the original environment to train the LSTM-TrajGAN and its variant exGAN
- the environment to train the TrajGDM model with variable length support
- the environment to conduct the utility evaluation of the synthetic trajectory datasets

Quickstart:

# 1) Create a conda environment for training models or to conduct a utility evaluation

cd models\lstm-trajgan
conda activate venv_LSTM-TrajGAN_original  
conda env create -f environement.yml

# 2) Prepare Geolife (weekly shown; also supports daily/hourly) (use the env_UtilEval environement)

cd Data/prep_geolife  
python prep_geolife.py --dataset_temporal_aggregation weekly

# 3) Split the data and train models (verify path to Data)

cd ../../models/lstm-trajgan  
conda env create -f environement.yml  
conda activate venv_LSTM-TrajGAN_original  
python split_data.py  
python train.py

cd ../exGAN  
conda env create -f environement.yml
conda activate venv_LSTM-TrajGAN_original
python split_data.py
python train.py

cd ../TrajGDM
conda env create -f environement.yml
conda activate venv_TrajGDM
python process_data.py\\
python runners/trainer.py

# 4) Install and run utility evaluation

cd ../../eval_utility
conda env create -f environement.yml
conda activate venv_UtilEval
pip install . # or: pip install -e .\\
jupyter lab # open notebooks under eval_utility/Notebook

# 5) run privacy evaluation Notebooks
