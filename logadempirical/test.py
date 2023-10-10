import sys
sys.path.append('./')

from deepod.models.prenet import PReNet
from deepod.utils.utility import cal_metrics
# from load_data import load_train_data, load_test_data, contamination_data, mislabeled_data
import torch
import pickle
from logadempirical.logdeep.dataset.sample import sliding_window, load_features
from logadempirical.logdeep.dataset.log import pairwise_log_dataset, pairwise_log_dataset_for_test
from logadempirical.logdeep.models.spiknet_bgl import DualInputNet_Spike
from torch.utils.data import DataLoader
import gc
import numpy as np
import random
import snntorch as snn

def con_data(ratio=0.1):
    data_path = './experimental_results/demo/random/bgl/100/train_c0.1.pkl'
    min_len = 0
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    logs = [] 
    # i = 0
    # j = 0
    # for seq in data:
    #     if len(seq['EventId']) < min_len:
    #         continue
    #     if not isinstance(seq['Label'], int):
    #         label = max(seq['Label'].tolist())
    #     else:
    #         label = seq['Label']
    #     print(label)
    #     if label  == 1:
    #         if random.random() < ratio:
    #             seq['Label'] = 0
    #             j += 1
    #         # print(seq['Label'])
    #         i += 1
    k = 0
    for seq in data:
        if len(seq['EventId']) < min_len:
            continue
        if not isinstance(seq['Label'], int):
            label = max(seq['Label'].tolist())  
            if len(seq['Label']) == 3:
                print(seq)
                print(type(seq))
                print(seq['Label'])
                break
        else:
            print('......////////')
            label = seq['Label']
            print(seq['Label'])
            if seq['Label'] == 1:
                print(seq)
            else:
                print(seq['Label'])
            k += 1
            
       
        
    # print('Total {}/{} anomlous sequence append to normal sequence'.format(j, i))
    # print(k)

def run(abnormal_ratio, contamination_ratio, mis_ratio, is_inference=True):
    hidden_dims = '128,100'
    batch_size = 1024
    epochs = 20 
    iter_num = -1 #
    lr = 1e-4
    saved_name = 'abn={}_con={}_mis={}'.format(abnormal_ratio, contamination_ratio, mis_ratio)
    if is_inference:
        epochs = -1
    # Processing data
    X_train, y_train, X_val, y_val = load_train_data()
    if contamination_ratio > 0:
        X_train, y_train = contamination_data(X_train, y_train, contamination_ratio=contamination_ratio, abnormal_ratio=abnormal_ratio)
    if mis_ratio > 0:
        X_train, y_train = mislabeled_data(X_train, y_train, mis_ratio=mis_ratio)
    return X_train, y_train

def analysis_data():
    path = 'experimental_results/demo/random/bgl/100/train.pkl'
    with open(path, 'rb') as f:
        data_iter = pickle.load(f) 
    total = len(data_iter)  
    print('Total sequence: {}'.format(total))
    total_anomaly = 0
    for seq in data_iter:
        if not isinstance(seq['Label'], int):
            label = max(seq['Label'].tolist())
        else:
            label = seq['Label']
        if label == 1:
            total_anomaly += 1
    print('Anomaly sequence: {}'.format(total_anomaly))

def test_dataloader():
    vocab_path = './experimental_results/demo/random/spirit/100/prolog_vocab.pkl'
    emb_dir = './dataset/parsed_dataset/dataset/spirit'
    is_predict_logkey = False
    semantics = True
    train_ratio = 1.0
    embeddings = 'embeddings.json'
    embedding_dim = 300
    data = load_features("./experimental_results/demo/random/spirit/100/train.pkl", only_normal=is_predict_logkey)
    with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
    train_logs, train_labels = sliding_window(data,
                                            vocab=vocab,
                                            window_size=100,
                                            data_dir=emb_dir,
                                            is_predict_logkey=is_predict_logkey,
                                            semantics=semantics,
                                            sample_ratio=train_ratio,
                                            e_name=embeddings,
                                            in_size=embedding_dim 
                                            )
    # train_logs = train_logs[:200000]
    # train_labels = train_labels[:200000]
    # n_val = int(len(train_logs) * 0.01)
    # val_logs, val_labels = train_logs[-n_val:], train_labels[-n_val:]
    del data
    gc.collect()
    
    train_dataset = pairwise_log_dataset(logs=train_logs,
                                labels=train_labels)
    
    # valid_dataset = pairwise_log_dataset_for_test(val_logs, val_labels, train_logs, train_labels)
    
    train_loader = DataLoader(train_dataset,
                            batch_size=1024,
                            shuffle=True,
                            pin_memory=True)
    # valid_loader = DataLoader(valid_dataset,
    #                             batch_size=8,
    #                             shuffle=False,
    #                             pin_memory=True)

    for i, (inputs, labels) in enumerate(train_loader):
        # x1, x2_a_list, x2_u_list = inputs

        # semantic_feature1 = x1['features'][2]
        # semantic_feature2 = x2_a_list['features'][2]
        # semantic_feature3 = x2_u_list['features'][2]
        # print(semantic_feature1.size())
        # print(semantic_feature2.size())
        # print(semantic_feature3.size())
        # break
        uu = torch.sum(labels == 0)
        au = torch.sum(labels == 4)
        aa = torch.sum(labels == 8)
        print('uu:aa:au = {}:{}:{}'.format(uu, au, aa))
        x1, x2 = inputs
        print(x1['features'][2].size())
        print(x2['features'][2].size())
        if i == 5:
            break
        
def view_vocab(event_id):
    vocab_path = './experimental_results/demo/random/bgl/100/loganomaly_vocab.pkl'
    with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)

    return vocab.itos[event_id-1]

def get_data_loader():
    vocab_path = './experimental_results/demo/random/bgl/100/prolog_vocab.pkl'
    emb_dir = './dataset/parsed_dataset/dataset/bgl/'
    is_predict_logkey = False
    semantics = True
    sample_ratio = 1.0
    embeddings = 'embeddings.json'
    embedding_dim = 300
    data = load_features("./experimental_results/demo/random/bgl/100/test.pkl", only_normal=is_predict_logkey)
    with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
    logs, labels = sliding_window(data,
                                vocab=vocab,
                                window_size=100,
                                data_dir=emb_dir,
                                is_predict_logkey=is_predict_logkey,
                                semantics=semantics,
                                sample_ratio=sample_ratio,
                                e_name=embeddings,
                                in_size=embedding_dim 
                                )

    del data
    gc.collect()
    
    logs_test = []
    for i in range(len(labels)):
        features = [torch.tensor(logs[i][0][0], dtype=torch.long)]
        for j in range(1, len(logs[i][0])):
            features.append(torch.tensor(logs[i][0][j], dtype=torch.float))
        logs_test.append({
            "features": features,
            "idx": logs[i][1]
        })
    
    return logs_test, np.array(labels)
     
@torch.no_grad()  
def show_results():
    model_path = 'experimental_results/demo/random/bgl/100/prolog/prolog_90.pth'
    model = DualInputNet_Spike(num_inputs=300, num_hidden=128, num_out=32).to('cuda')
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()
    logs, labels = get_data_loader()    
    
    anom_seq = []
    norm_seq = []
    anom_event_seq = []
    norm_event_seq = []
    for i, log in enumerate(logs):
        if labels[i] == 1:
            anom_seq.append(log['features'][2])
            anom_event_seq.append(log['features'][0])
        else:
            norm_seq.append(log['features'][2])
            norm_event_seq.append(log['features'][0])
    anom_seq = torch.stack(anom_seq)
    norm_seq = torch.stack(norm_seq)
    
    x1 = anom_seq.cuda()
    x2 = norm_seq.cuda()
    print(x1.size())
    spk_rec1_anom, spk_rec2_anom, spk_rec3_anom, mem_rec1_anom, mem_rec2_anom, mem_rec3_anom  = model.spike_rnn.forward_pass(x1) # (bs, time_step, embedding_dims)
    spk_rec1_norm, spk_rec2_norm, spk_rec3_norm, mem_rec1_norm, mem_rec2_norm, mem_rec3_norm = model.spike_rnn.forward_pass(x2)
    
    mem_rec1_anom_seq = mem_rec1_anom.cpu().numpy()
    mem_rec2_anom_seq = mem_rec2_anom.cpu().numpy()
    mem_rec3_anom_seq = mem_rec3_anom.cpu().numpy()
    spk_rec1_anom_seq = spk_rec1_anom.cpu().numpy()
    spk_rec2_anom_seq = spk_rec2_anom.cpu().numpy()
    spk_rec3_anom_seq = spk_rec3_anom.cpu().numpy()
    
    mem_rec1_norm_seq = mem_rec1_norm.cpu().numpy()
    mem_rec2_norm_seq = mem_rec2_norm.cpu().numpy()
    mem_rec3_norm_seq = mem_rec3_norm.cpu().numpy()
    spk_rec1_norm_seq = spk_rec1_norm.cpu().numpy()
    spk_rec2_norm_seq = spk_rec2_norm.cpu().numpy()
    spk_rec3_norm_seq = spk_rec3_norm.cpu().numpy()

    np.save('./intepret/mem_rec1_anom_seq.npz', mem_rec1_anom_seq)
    np.save('./intepret/mem_rec2_anom_seq.npz', mem_rec2_anom_seq)
    np.save('./intepret/mem_rec3_anom_seq.npz', mem_rec3_anom_seq)
    np.save('./intepret/spk_rec1_anom_seq.npz', spk_rec1_anom_seq)
    np.save('./intepret/spk_rec2_anom_seq.npz', spk_rec2_anom_seq)
    np.save('./intepret/spk_rec3_anom_seq.npz', spk_rec3_anom_seq)
    
    np.save('./intepret/mem_rec1_norm_seq.npz', mem_rec1_norm_seq)
    np.save('./intepret/mem_rec2_norm_seq.npz', mem_rec2_norm_seq)
    np.save('./intepret/mem_rec3_norm_seq.npz', mem_rec3_norm_seq)
    np.save('./intepret/spk_rec1_norm_seq.npz', spk_rec1_norm_seq)
    np.save('./intepret/spk_rec2_norm_seq.npz', spk_rec2_norm_seq)
    np.save('./intepret/spk_rec3_norm_seq.npz', spk_rec3_norm_seq)
    
    
    # # print(anom_event_seq[100])
    # print(model.spike_rnn.li_out.threshold)
    
    # neuro_idx = 34
    # import snntorch.spikeplot as splt
    # import matplotlib.pyplot as plt
    # # Generate Plots
    # fig, ax = plt.subplots(4, figsize=(8,7), sharex=True,
    #                   gridspec_kw = {'height_ratios': [0.4, 1, 1, 0.4]})
    # # Plot input current
    # splt.raster(spk_rec1_norm_seq[:], ax[0], s=400, c="black", marker="|")
    # ax[0].set_ylabel("Input Spikes")
    # ax[0].set_title("Synaptic Conductance-based Neuron Model With Input Spikes")
    # ax[0].set_yticks([])
    
    # # Plot membrane potential
    # ax[1].plot(mem_rec1_norm_seq[:])
    # ax[1].set_ylim([0, 1.5])
    # ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
    # ax[1].axhline(y=float(model.spike_rnn.lif1.threshold[neuro_idx]), alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    # plt.xlabel("Time step")
    
    # # Plot membrane potential
    # ax[2].plot(mem_rec2_norm_seq[:])
    # ax[2].set_ylim([0, 1.5])
    # ax[2].set_ylabel("Membrane Potential ($U_{mem}$)")
    # ax[2].axhline(y=float(model.spike_rnn.li_out.threshold), alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    # plt.xlabel("Time step")

    # # Plot output spike using spikeplot
    # splt.raster(spk_rec2_norm_seq[:], ax[3], s=400, c="black", marker="|")
    # plt.ylabel("Output spikes")
    # ax[3].set_yticks([])
    # plt.savefig('./images/{}_norm.jpg'.format(neuro_idx), bbox_inches='tight')
    
    # show_mem(mem_rec1_anom_seq[:, :9], spk_rec1_anom_seq[:, :9])
    
    
    # anom_idx = 0 # 异常序列索引
    # nom_idx = 0 # 正常序列索引
    # for idx in range(100):
    #     # idx = 2 # 第几个神经元
    #     show_spike(spk_rec1_anom[anom_idx], idx, label='anom')
    #     show_spike(spk_rec1_norm[nom_idx], idx, label='norm')
        
def show_mem(mem_rec, spk_rec):
    import snntorch.spikeplot as splt
    import matplotlib.pyplot as plt
    
    splt.traces(mem_rec, spk_rec) 
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.savefig('./images/mem/0.jpg', bbox_inches='tight') 
   
def show_spike(spk_rec, idx, label='anom'):
    import snntorch.spikeplot as splt
    import matplotlib.pyplot as plt
    fig = plt.figure(facecolor="w", figsize=(8, 1))
    ax = fig.add_subplot(111)
    
    splt.raster(spk_rec[:, idx].unsqueeze(1), ax, s=100, c="black", marker="|")
    plt.title("Neuron {}".format(idx))
    plt.xlabel("Time step")
    plt.yticks([])
    plt.savefig('./images/{}/raster_{}.jpg'.format(label, idx), bbox_inches='tight')

def lif_model():
    inputs = torch.randn(5, 2, 5)
    
    lif = snn.RLeaky(beta=0.5, threshold=5, reset_mechanism='none',linear_features=5)
    spike, mem = lif.init_rleaky()
    
    spike, mem = lif(inputs, spike, mem)    
    print(spike[:])

def compute_spike_rate():
    model_path = 'experimental_results/demo/random/bgl/100/prolog/prolog_90.pth'
    model = DualInputNet_Spike(num_inputs=300, num_hidden=128, num_out=32).to('cuda')
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()
    logs, labels = get_data_loader()    
    
    anom_seq = []
    norm_seq = []
    anom_event_seq = []
    norm_event_seq = []
    for i, log in enumerate(logs):
        if labels[i] == 1:
            anom_seq.append(log['features'][2])
            anom_event_seq.append(log['features'][0])
        else:
            norm_seq.append(log['features'][2])
            norm_event_seq.append(log['features'][0])
    anom_seq = torch.stack(anom_seq)
    norm_seq = torch.stack(norm_seq)
    
    x1 = anom_seq.cuda()
    x2 = norm_seq.cuda()
    print(x1.size())
    spk_rec1_anom, spk_rec2_anom, spk_rec3_anom, mem_rec1_anom, mem_rec2_anom, mem_rec3_anom  = model.spike_rnn.forward_pass(x1) # (bs, time_step, embedding_dims)
    spk_rec1_norm, spk_rec2_norm, spk_rec3_norm, mem_rec1_norm, mem_rec2_norm, mem_rec3_norm = model.spike_rnn.forward_pass(x2)
    # (bs,seq_len, 32)
    total_neuro_1 = (spk_rec1_norm.size()[0] + spk_rec1_anom.size()[0]) * spk_rec1_anom.size()[1] * spk_rec1_anom.size()[2]
    total_neuro_2 = (spk_rec2_norm.size()[0] + spk_rec2_anom.size()[0]) * spk_rec2_anom.size()[1] * spk_rec2_anom.size()[2]
    total_neuro_3 = (spk_rec3_norm.size()[0] + spk_rec3_anom.size()[0]) * spk_rec3_anom.size()[1] * spk_rec3_anom.size()[2]

    spike_rate_1 = (spk_rec1_anom.sum() + spk_rec1_norm.sum()) / total_neuro_1 / 2
    spike_rate_2 = (spk_rec2_anom.sum() + spk_rec2_norm.sum()) / total_neuro_2 / 2
    spike_rate_3 = (spk_rec3_anom.sum() + spk_rec3_norm.sum()) / total_neuro_3 / 2
    print('spike_rate_1:', spike_rate_1)
    print('spike_rate_2:', spike_rate_2)
    print('spike_rate_3:', spike_rate_3)


    
if __name__ == '__main__':
    pass
    # X_train, y_train = run(0.1, 0.01, 0.1)
    # analysis_data()
    # test_dataloader()
    # lif_model()
    # con_data()
    # show_results()
    # explore_interpret()
    # 第100个异常序列
    # anom_event_seq = np.array([109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109,
    #                 109, 109, 109, 109, 109, 109, 109, 109, 109, 109,  76, 130, 130, 130,
    #                 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130,  17,
    #                     17,  17,  17,  17,  17,  30,  30,  30,  30,  30,  30,  30,  30,  30,
    #                     30,  30,  41,  41,  41,  41,  41,  41,  41,  41,  30,  30,  30,  30,
    #                     30,  30,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,
    #                     17,  17,  17,  17,  17,  17,  17,  17,  17, 152, 152, 152, 152, 152,
    #                 286, 286])
    # active_neuros = np.array([4, 5, 6, 9, 11, 13, 14, 24, 33, 34, 43, 45, 46, 48, 50, 52, 59, 63, 65, 67, 74, 76, 79, 81, 83, 84, 94, 96, 97, 99])
    # anom_events = anom_event_seq[active_neuros]
    # print(anom_events)
    # print(view_vocab(76))
    # compute_spike_rate()
