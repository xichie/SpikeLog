import sys
sys.path.append('./')
import numpy as np
import gc

from logadempirical.logdeep.tools.predict import Predicter
from logadempirical.logdeep.tools.train import Trainer
from logadempirical.logdeep.dataset.vocab import Vocab
import os
import pickle
 
def run_prolog(options):
    if not os.path.exists(options["vocab_path"]):
        with open(options["train_vocab"], 'rb') as f:
            data = pickle.load(f)
        logs = []
        for x in data:
            try:
                l = max(x['Label'])
            except:
                l = x['Label']
            if l == 0:
                logs.append(x['EventId'])
        vocab = Vocab(logs, os.path.join(options['data_dir'], "embeddings.json"), "loganomaly")
        print("vocab size", len(vocab))
        print("save vocab in", options["vocab_path"])
        vocab.save_vocab(options["vocab_path"])
    
    Trainer(options).start_train()
    Predicter(options).predict_pairwised()