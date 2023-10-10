import pickle
import numpy as np
import os
from sklearn.utils import shuffle
import gc
from logadempirical.logdeep.dataset.sample import sliding_window, load_features


vocab_path = './dataset/bgl/vocab.pkl'
data_dir = './dataset/bgl/'
output_dir = './dataset/bgl/'
is_predict_logkey = False
history_size = 100
emb_dir = data_dir
semantics = True
train_ratio = 1
embeddings = 'embeddings.json'
# embeddings = 'neural'
input_size = 300
valid_ratio = 0.1

def save_vocab():
    from logadempirical.logdeep.dataset.vocab import Vocab
    if not os.path.exists(vocab_path):
        with open(data_dir + 'train_fixed_100.pkl', 'rb') as f:
            data = pickle.load(f)
        logs = []
        for x in data:
            try:
                l = max(x['Label'])
            except:
                l = x['Label']
            if l == 0:
                logs.append(x['EventId'])
        vocab = Vocab(logs, os.path.join('data_dir', "embeddings.json"), "deeplog")
        print("vocab size", len(vocab))
        print("save vocab in", vocab_path)
        vocab.save_vocab(vocab_path)

def load_train_data():
    print("Loading vocab")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print('Vocab length: ', len(vocab))
    print("Loading train dataset\n")
    data = load_features(output_dir + "train_fixed_100.pkl", only_normal=is_predict_logkey)

    train_logs, train_labels = sliding_window(data,
                                                vocab=vocab,
                                                window_size=history_size,
                                                data_dir=emb_dir,
                                                is_predict_logkey=is_predict_logkey,
                                                semantics=semantics,
                                                sample_ratio=train_ratio,
                                                e_name=embeddings,
                                                in_size=input_size
                                                )
    print('Shuffling...')
    train_logs, train_labels = shuffle(train_logs, train_labels, random_state=22)
    train_data = np.array(train_logs)
    n_val = int(len(train_logs) * valid_ratio)
    val_logs, val_labels = train_logs[-n_val:], train_labels[-n_val:]
    
    val_data = np.array(val_logs)
    
    return train_data, np.array(train_labels), val_data, np.array(val_labels)

def load_test_data():
    print("Loading vocab")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print('Vocab length: ', len(vocab))
    print("Loading test dataset\n")
    data = load_features(output_dir + "test_fixed_100.pkl", only_normal=is_predict_logkey)

    test_logs, test_labels = sliding_window(data,
                                            vocab=vocab,
                                            window_size=history_size,
                                            data_dir=emb_dir,
                                            is_predict_logkey=is_predict_logkey,
                                            semantics=semantics,
                                            e_name=embeddings,
                                            in_size=input_size
                                            )
    test_data = np.array(test_logs)
    return test_data, np.array(test_labels)

def contamination_data(X_train, y_train, contamination_ratio=0.01, abnormal_ratio=0.3):
    '''
        无标签数据中存在一定比例的异常(异常污染)
        contamination_ratio: 污染比例 (0.01, 0.05)
        abnormal_ratio: 训练集中保留异常数据占原始异常数据的比例
    '''
    unlabeld_X = X_train[y_train==0]
    unlabeld_y = y_train[y_train==0]
    ano_X = X_train[y_train==1]
    ano_y = y_train[y_train==1]
    orig_ano_num = len(ano_y)
    
    # 注入一定比例的异常到无标签数据中
    total_unlabeled_num = len(unlabeld_y)
    fake_num = int(total_unlabeled_num * contamination_ratio / (1 - contamination_ratio)) 
    fake_X = ano_X[:fake_num]
    # ano_X = ano_X[fake_num:]
    # ano_y = ano_y[fake_num:]
    fake_y = np.zeros((len(fake_X),))
    unlabeld_X = np.concatenate((unlabeld_X, fake_X), axis=0)
    unlabeld_y = np.concatenate((unlabeld_y, fake_y), axis=0)
    print('Anomaly sequence in unlabeled set: ', fake_num)
    
    # 保留多少比例的异常在训练集中
    ano_num = int(orig_ano_num * abnormal_ratio) 
    ano_X = ano_X[:ano_num]
    ano_y = ano_y[:ano_num]
    X_train = np.concatenate((unlabeld_X, ano_X), axis=0)
    y_train = np.concatenate((unlabeld_y, ano_y), axis=0)
    
    X_train, y_train = shuffle(X_train, y_train, random_state=22)
    print('Anomaly sequence remaing in training set: {}/{}'.format(ano_num, orig_ano_num))
    print('Contamination sequence: {}/{} '.format(fake_num, len(unlabeld_X)))

    del unlabeld_X
    del unlabeld_y
    del ano_X
    del ano_y
    del fake_X
    del fake_y
    gc.collect()
    return X_train, y_train

def mislabeled_data(X_train, y_train, mis_ratio=0.1):
    '''
        将一定比例X_train中的正常数据错误标记为异常
        @params:
        mis_ratio: 错误标记的比例
    '''
    abn_idx = y_train == 1
    nor_idx = y_train == 0
    
    mis_num = int(mis_ratio * np.sum(abn_idx) / (1 - mis_ratio))
    
    nor_data = X_train[nor_idx]
    abn_data = X_train[abn_idx]
    
    abn_data = np.concatenate((nor_data[:mis_num], abn_data), axis=0)
    nor_data = nor_data[mis_num:]
    
    X_train = np.concatenate((abn_data, nor_data), axis=0)
    y_train = np.concatenate((np.ones(len(abn_data)), np.zeros(len(nor_data))), axis=0)
    print('Missing labeled sequence:{}/{} '.format(mis_num, len(abn_data)))

    del nor_data
    del abn_data
    gc.collect()
    X_train, y_train = shuffle(X_train, y_train, random_state=22)
    return X_train, y_train
    
    


if __name__ == '__main__':
    # save_vocab()
    # train_data, train_labels, val_data, val_labels = load_train_data()
    # print(train_data.shape)
    # print(train_labels.shape)
    test_data, test_labels = load_test_data()
    print(test_data.shape)
