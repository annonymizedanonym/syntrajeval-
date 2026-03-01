This repository accompanies the submitted paper and provides:

- Preprocessing and temporal aggregation for Geolife trajectories (weekly/daily/hourly):
  - You first need to download the Geolife dataset: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/
  - Then set the path in the Data/prep_geolife/prep_geolife.py script
- Training for three generative models: LSTM‑TrajGAN, ExGAN, TrajGDM
- Utility evaluation via the syntraj_utileval package:
  - Notebooks/caseAB.ipynb (to reproduce tables 2,3,9 and 10)
  - Notebooks/maup_sensitivity_analysis.ipynb (to reproduce table 7)
- Privacy evaluation on LSTM-TrajGAN and exGAN via two notebooks (eval_privacy/).
  - MIATrajGAN.ipynb executes our MIA attack on the LSTM-TrajGAN model (make a folder named "params" inside the path you set for saving modes )
  - MIAexGAN.ipynb executes our MIA attack on the exGAN model

This repository provides 3 environements for reproducibility:

- the original environment to train the LSTM-TrajGAN and its variant exGAN
- the environment to train the TrajGDM model with variable length support
- the environment to conduct the utility evaluation of the synthetic trajectory datasets

Quickstart:

# 1) Create a conda environment for training models or to conduct a utility evaluation

cd models\lstm-trajgan<br>
conda activate venv_LSTM-TrajGAN_original<br>  
conda env create -f environement.yml<br>

# 2) Prepare Geolife (weekly shown; also supports daily/hourly) (use the env_UtilEval environement)

cd Data/prep_geolife<br>
python prep_geolife.py --dataset_temporal_aggregation weekly<br>

# 3) Split the data and train models (verify path to Data)

cd ../../models/lstm-trajgan<br>
conda env create -f environement.yml<br>
conda activate venv_LSTM-TrajGAN_original<br>
python split_data.py<br>
python train.py<br>

cd ../exGAN<br>
conda env create -f environement.yml<br>
conda activate venv_LSTM-TrajGAN_original<br>
python split_data.py<br>
python train.py<br>

cd ../TrajGDM<br>
conda env create -f environement.yml<br>
conda activate venv_TrajGDM<br>
python process_data.py<br>
python runners/trainer.py<br>

# 4) Install and run utility evaluation

cd ../../eval_utility<br>
conda env create -f environement.yml<br>
conda activate venv_UtilEval<br>
pip install . # or: pip install -e .<br>
jupyter lab # open notebooks under eval_utility/Notebook<br>

# 5) run privacy evaluation Notebooks
