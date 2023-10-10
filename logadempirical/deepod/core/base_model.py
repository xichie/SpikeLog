# -*- coding: utf-8 -*-
"""
Base class for deep Anomaly detection models
some functions are adapted from the pyod library
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

import numpy as np
import torch
import random
import time
from abc import ABCMeta, abstractmethod
from scipy.stats import binom
from deepod.utils.utility import get_sub_seqs
from deepod.core.base_networks import sequential_net_name
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm


class BaseDeepAD(metaclass=ABCMeta):
    """
    Abstract class for deep outlier detection models

    Parameters
    ----------

    data_type: str, optional (default='tabular')
        Data type

    network: str, optional (default='MLP')
        network structure for different data structures

    epochs: int, optional (default=100)
        Number of training epochs

    batch_size: int, optional (default=64)
        Number of samples in a mini-batch

    lr: float, optional (default=1e-3)
        Learning rate

    n_ensemble: int or str, optional (default=1)
        Number of ensemble size

    seq_len: int, optional (default=100)
        Size of window used to create subsequences from the data
        deprecated when handling tabular data (network=='MLP')

    stride: int, optional (default=1)
        number of time points the window will move between two subsequences
        deprecated when handling tabular data (network=='MLP')

    epoch_steps: int, optional (default=-1)
        Maximum steps in an epoch
            - If -1, all the batches will be processed

    prt_steps: int, optional (default=10)
        Number of epoch intervals per printing

    device: str, optional (default='cuda')
        torch device,

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    verbose: int, optional (default=1)
        Verbosity mode

    random_state: int, optional (default=42)
        the seed used by the random

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.

    """
    def __init__(self, model_name, data_type='ts', network='LSTM',
                 epochs=100, batch_size=64, lr=1e-3,
                 n_ensemble=1, seq_len=100, stride=1,
                 epoch_steps=-1, prt_steps=10,
                 device='cuda', contamination=0.1,
                 verbose=1, random_state=42):
        self.model_name = model_name
        
        self.data_type = data_type
        self.network = network

        if data_type == 'ts':
            assert self.network in sequential_net_name, \
                'Assigned network cannot handle time-series data'

        self.seq_len = seq_len
        self.stride = stride

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.device = device
        self.contamination = contamination

        self.epoch_steps = epoch_steps
        self.prt_steps = prt_steps
        self.verbose = verbose

        self.n_features = -1
        self.n_samples = -1
        self.criterion = None
        self.net = None

        self.n_ensemble = n_ensemble

        self.train_loader = None
        self.test_loader = None

        self.epoch_time = None

        self.train_data = None
        self.train_label = None

        self.decision_scores_ = None
        self.labels_ = None
        self.threshold_ = None
        self.optimizer = None
        self.net = None
        self.random_state = random_state
        self.set_seed(random_state)
        return

    def fit(self, X, y=None, saved_name=0):
        """
        Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples, )
            Not used in unsupervised methods, present for API consistency by convention.
            used in (semi-/weakly-) supervised methods

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.saved_name = saved_name
        if self.epochs < 0:
            print('Model was loaded....')
            X_seqs = X
            y_seqs = y
            self.train_data = X_seqs
            self.train_label = y_seqs
            self.n_samples, self.n_features = X_seqs.shape[0], X_seqs.shape[2]
            self.criterion = torch.nn.L1Loss(reduction='mean')
            return self
        if self.data_type == 'ts':
            # X_seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)
            # y_seqs = get_sub_seqs(y, seq_len=self.seq_len, stride=self.stride) if y is not None else None
            X_seqs = X
            y_seqs = y
            self.train_data = X_seqs
            self.train_label = y_seqs
            self.n_samples, self.n_features = X_seqs.shape[0], X_seqs.shape[2]
        else:
            self.train_data = X
            self.train_label = y
            self.n_samples, self.n_features = X.shape

        if self.verbose >= 1:
            print('Start Training...')

        if self.n_ensemble == 'auto':
            self.n_ensemble = int(np.floor(100 / (np.log(self.n_samples) + self.n_features)) + 1)
        if self.verbose >= 1:
            print(f'ensemble size: {self.n_ensemble}')

        for _ in range(self.n_ensemble):
            self.train_loader, self.net, self.criterion = self.training_prepare(self.train_data,
                                                                                y=self.train_label)
            self._training()

        if self.verbose >= 1:
            print('Start Inference on the training data...')

        # self.decision_scores_ = self.decision_function(X)
        # self.labels_ = self._process_decision_scores()

        return self

    def decision_function(self, X):
        """Predict raw anomaly scores of X using the fitted detector.

        The anomaly score of an input sample is computed based on the fitted
        detector. For consistency, outliers are assigned with
        higher anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """

        testing_n_samples = X.shape[0]

        if self.data_type == 'ts':
            # X = get_sub_seqs(X, seq_len=self.seq_len, stride=1)
            X = X

        s_final = np.zeros(testing_n_samples)
        for _ in range(self.n_ensemble):
            self.test_loader = self.inference_prepare(X)

            z, scores = self._inference()
            z, scores = self.decision_function_update(z, scores)

            # if self.data_type == 'ts':
            #     padding = np.zeros(self.seq_len-1)
            #     scores = np.hstack((padding, scores))
            s_final += scores

        return s_final

    def predict(self, X, return_confidence=False):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        return_confidence : boolean, optional(default=False)
            If True, also return the confidence of prediction.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        confidence : numpy array of shape (n_samples,).
            Only if return_confidence is set to True.
        """

        pred_score = self.decision_function(X)
        prediction = (pred_score > self.threshold_).astype('int').ravel()

        if return_confidence:
            confidence = self._predict_confidence(pred_score)
            return prediction, confidence

        return prediction

    def _predict_confidence(self, test_scores):
        """Predict the model's confidence in making the same prediction
        under slightly different training sets.
        See :cite:`perini2020quantifying`.

        Parameters
        -------
        test_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.

        Returns
        -------
        confidence : numpy array of shape (n_samples,)
            For each observation, tells how consistently the model would
            make the same prediction if the training set was perturbed.
            Return a probability, ranging in [0,1].

        """
        n = len(self.decision_scores_)

        count_instances = np.vectorize(lambda x: np.count_nonzero(self.decision_scores_ <= x))
        n_instances = count_instances(test_scores)

        # Derive the outlier probability using Bayesian approach
        posterior_prob = np.vectorize(lambda x: (1 + x) / (2 + n))(n_instances)

        # Transform the outlier probability into a confidence value
        confidence = np.vectorize(
            lambda p: 1 - binom.cdf(n - int(n*self.contamination), n, p)
        )(posterior_prob)
        prediction = (test_scores > self.threshold_).astype('int').ravel()
        np.place(confidence, prediction==0, 1-confidence[prediction == 0])
        return confidence

    def _process_decision_scores(self):
        """Internal function to calculate key attributes:

        - threshold_: used to decide the binary label
        - labels_: binary labels of training data

        Returns
        -------
        self
        """

        self.threshold_ = np.percentile(self.decision_scores_, 100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype('int').ravel()

        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)

        return self

    def _training(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                        lr=self.lr,
                                        weight_decay=1e-5)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)

        self.net.train()
        for i in range(self.epochs):
            t1 = time.time()
            total_loss = 0
            cnt = 0
            for batch_x in self.train_loader:
                loss = self.training_forward(batch_x, self.net, self.criterion)
                self.net.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                cnt += 1

                # terminate this epoch when reaching assigned maximum steps per epoch
                if cnt > self.epoch_steps != -1:
                    break
            
            t = time.time() - t1
            if self.verbose >= 1 and (i == 0 or (i+1) % self.prt_steps == 0):
                print(f'epoch{i+1}, '
                      f'training loss: {total_loss/cnt:.6f}, '
                      f'time: {t:.1f}s')

            if i == 0:
                self.epoch_time = t

            self.epoch_update(i)
            
        self.lr_scheduler.step()
        return

    def _inference(self):
        self.net.eval()
        with torch.no_grad():
            z_lst = []
            score_lst = []

            if self.verbose >= 2:
                _iter_ = tqdm(self.test_loader, desc='testing: ')
            else:
                _iter_ = self.test_loader

            for batch_x in _iter_:
                batch_z, s = self.inference_forward(batch_x, self.net, self.criterion)
                z_lst.append(batch_z)
                score_lst.append(s)

        z = torch.cat(z_lst).data.cpu().numpy()
        scores = torch.cat(score_lst).data.cpu().numpy()

        return z, scores

    @abstractmethod
    def training_forward(self, batch_x, net, criterion):
        """define forward step in training"""
        pass

    @abstractmethod
    def inference_forward(self, batch_x, net, criterion):
        """define forward step in inference"""
        pass

    @abstractmethod
    def training_prepare(self, X, y):
        """define train_loader, net, and criterion"""
        pass

    @abstractmethod
    def inference_prepare(self, X):
        """define test_loader"""
        pass

    def epoch_update(self, epoch):
        """for any updating operation after each training epoch"""
        # state = {'net':self.net.state_dict(), 'optimizer': self.optimizer.state_dict()} 
        if (epoch+1) % 100 == 0 or epoch == self.epochs-1:
            torch.save(self.net, './dataset/bgl/results/2k_active/{}_{}.pth'.format(self.saved_name, epoch+1))
        return

    def decision_function_update(self, z, scores):
        """for any updating operation after decision function"""
        return z, scores

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True