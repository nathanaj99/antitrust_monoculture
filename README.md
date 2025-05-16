# "Homogeneous Algorithms Can Reduce Competition in Personalized Pricing"

## Python Packages
`requirements.txt` outlines all the python packages required.

## Scripts

First, in order to run the ``empirical*`` scripts, run ``1-download_acs.py``, which retrieves ACSIncome data. 

All other scripts are standalone and can be run independently, with the exception that `empirical_data_sharing_train.py` should be run before `empirical_data_sharing_analysis.py`. We provide the intermediary results, so the running the former script is not strictly necessary. Scripts starting with `theory` are simulated results, while scripts that start with `empirical` are stylized experiments using ACSIncome data. All files produce figures that are automatically saved in the `figs/` directory.

## Data
All data are taken directly from [folktables](https://github.com/socialfoundations/folktables), see ``1-download_acs.py``. 