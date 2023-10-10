# SpikeLog
SpikeLog: Log-based anomaly detection via Potential-assisted Spiking Neuron Network

# 1. Data Preparation
## 1.1 Download preprocessed datasets
We refer to the hub: https://github.com/LogIntelligence/LogADEmpirical/tree/dev#1-data-preparation.  

Raw and preprocessed datasets (including parsed logs and their embeddings) are available at https://zenodo.org/record/8115559.

Download the data to <a>./dataset/ </a>. 
## 1.2  Prepare your own data set
Run following script to generate `embedding.json` (Default BGL)
```python
python generate_template_embedding.py 
```
# BGL_2k Demo
```Shell
python -u main_run.py --folder=bgl/ --log_file=BGL_2k.log --dataset_name=bgl --model_name=prolog --window_type=sliding  \
--semantics  --input_size=1 --embedding_dim=300 \
--data_dir=./dataset/ \
--sample=sliding_window --is_logkey --train_size=0.7 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=10 \
--n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=64  --history_size=100 --lr=5e-4 \
--session_level=entry --window_size=100 --step_size=100 --output_dir=experimental_results/bgl/ \
--is_process
```



