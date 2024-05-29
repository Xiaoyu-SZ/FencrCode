# FencrCode
Welcome to the GitHub repository for FENCR. This repository contains all the necessary code and instructions to run and evaluate FENCR.

## Installation
To run FENCR, you'll need to have Python 3.7+ installed. Follow these steps to set up the environment:

1. Clone the repository: `git clone git@github.com:Xiaoyu-SZ/FencrCode.git`
2. Navigate to the repository directory: `cd FENCR`
3. Install the required packages: `pip install -r requirements.txt`

## Running FENCR
To run FENCR, you'll need to have a dataset prepared in the correct format. The code for preprocessing datasets are in the `preprocess/` folder. We also provide preprocessed data to download.
Follow these steps to run FENCR:

1. **Download the preprocessed dataset** (https://drive.google.com/drive/folders/1ITEC4ZC2UtCt1g_oEC9rh_BGKFoSM_wW?usp=sharing) and place it in the `dataset/` folder.
2. You can run the model as follows:

```bash
'python main.py --model_name FENCR --val_metrics ndcg@10 --test_metrics ndcg@5.10.20,hit@10,recall@10.20,precision@10 --eval_batch_size 8 --latent_dim 1 --bucket_size 0 --dataset taobao-1-1 --l2 1e-06 --lr 0.001 --es_patience 20 --output_strategy adaptive_sigmoid_ui --r_logic 1e-06 --loss_sum 1 --adaptive_loss 1 --layers [16] --batch_size 128 --test_sample_n 1000 --val_sample_n 1000'

'python main.py --model_name FENCR --val_metrics ndcg@10 --test_metrics ndcg@5.10.20,hit@10,recall@10.20,precision@10 --eval_batch_size 8 --latent_dim 1 --bucket_size 0 --dataset recsys2017-1-1 --l2 1e-06 --lr 0.001 --es_patience 20 --output_strategy adaptive_sigmoid_ui --loss_sum 1 --adaptive_loss 1 --layers [16] --batch_size 128 --test_sample_n 1000 --val_sample_n 1000 --r_logic 1e-08 --random_seed 1949'
```