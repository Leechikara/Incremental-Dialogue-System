# Incremental-Dialogue-System
Code and data for acl2019 paper "Incremental Learning from Scratch for Task-Oriented Dialogue Systems".

If you use any source codes or datasets included in this toolkit in your
work, please cite the following paper. The bibtex are listed below:

    @article{wang2019incremental,
      title={Incremental Learning from Scratch for Task-Oriented Dialogue Systems},
      author={Wang, Weikang and Zhang, Jiajun and Li, Qian and Hwang, Mei-Yuh and Zong, Chengqing and Li, Zhifei},
      journal={arXiv preprint arXiv:1906.04991},
      year={2019}
    }

### Requirements
     python 3.6
     pytorch >= 1.0.0
     numpy
     scipy
     sklearn

## Datasets
Our data is available at https://drive.google.com/file/d/1KYZNxzcU5kximq1-_IDC_4NCFabz40jN/view?usp=sharing.
Download and unzip at ./data.
- ./data/preprocessed contains the preprocessed five sub-dataset.
- ./data/original contains the original dialogue episodes before entity replacing.
- ./data/script annotates the dialogue scenarios in each episode.

## Run Models
python run.py

## Change Configurations
Change the parameters in RunConfig at config.py.