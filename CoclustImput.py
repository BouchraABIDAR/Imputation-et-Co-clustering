## ABIDAR Bouchra , Mustapha BOUSSEBAINE, Abderahme LARBI
# Projet : Données manquantes : Imputation et Co-clustering
# 2019/2020
# Encadré par : M. Mohamed NADIF

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import random
import scipy
import scipy.io
from itertools import groupby
import io
import random
import scipy.io as sio
import scipy.io as io
from sklearn.metrics import (adjusted_rand_score as ari,normalized_mutual_info_score as nmi)
from scipy import sparse 
from sklearn.metrics import confusion_matrix
from coclust.coclustering import CoclustInfo
from coclust.evaluation.external import accuracy as ACCURACY
from coclust.visualization import plot_delta_kl, plot_convergence
from coclust.coclustering import CoclustInfo 
from coclust.coclustering import CoclustSpecMod
from coclust.coclustering import CoclustMod
from coclust.clustering import SphericalKmeans
from coclust.coclust import *
from random import randrange
from sklearn.utils import check_random_state, check_array
from scipy.sparse.sputils import isdense
from sklearn.metrics import (adjusted_rand_score as ari,normalized_mutual_info_score as nmi)
from coclust.evaluation.external import accuracy as ACCURACY
import Cluster_Ensembles as CE
from sklearn.metrics import mean_squared_error

def random_init(n_clusters, n_cols, random_state=None):
    random_state = check_random_state(random_state)
    W_a = random_state.randint(n_clusters, size=n_cols)
    W = np.zeros((n_cols, n_clusters))
    W[np.arange(n_cols), W_a] = 1
    return W


def select_block(X,Z, W, z, w):
    block = X[np.equal(Z,z)==True]
    col_remov =[]
    for loc, i in enumerate(W):
        if i != w :
            col_remov.append(loc)
    block = np.delete(block, col_remov, axis=1)
    return block

def imput_block(X, Z, W, X_trace):
    X_res = X_trace
    na_values_cord = np.argwhere(np.isnan(X_res))
    na_values_block = [(Z[i],W[j]) for i,j in na_values_cord]
    for loc,cord in enumerate(na_values_cord):
        z_ = na_values_block[loc][0]
        w_ = na_values_block[loc][1]
        block = select_block(X,Z, W, z_, w_)
        mean_bloc = np.mean(block)
        X_res[cord[0]][cord[1]] = mean_bloc
    return X_res

def imput_block_mod(B, Z, W, X_trace):
    B_res = B
    na_values_cord = np.argwhere(np.isnan(X_trace))
    na_values_block = [(Z[i],W[j]) for i,j in na_values_cord]
    for loc,cord in enumerate(na_values_cord):
        z_ = na_values_block[loc][0]
        w_ = na_values_block[loc][1]
        block = select_block(B,Z, W, z_, w_)
        mean_bloc = np.mean(block)
        B_res[cord[0]][cord[1]] = mean_bloc
    return B_res

def data_AFC_for_initialisation(data):
    dataExp= pd.read_csv(data)
    return dataExp.to_numpy()


def Visualisation_Matrix(Model, Matrix_Docs_Terms, percentage): 

    reorganisation_indice_rows=np.argsort(Model.row_labels_)
    reorganisation_indice_cols=np.argsort(Model.column_labels_)
    
    Matrix=Matrix_Docs_Terms[reorganisation_indice_rows,:]
    Matrix=Matrix[:,reorganisation_indice_cols]
    
    plt.spy(Matrix,markersize=0.5,color="red",aspect='auto')
    fig =plt.gcf()
    fig.set_size_inches(6, 6)
    plt.title("Percentage des valeurs manquantes est {}".format(percentage))
    plt.show()
    

def metriques(model,y_true,y_pred):
    pred1 = model.row_labels_
    nmi_ = nmi(y_true, pred1)
    ari_ = ari(y_true, pred1)
    accuracy = ACCURACY(y_true, pred1)
    print("NMI: {}\nARI: {} ".format(nmi_, ari_))
    print("ACCURACY: %s" % accuracy)
    return nmi_, ari_, accuracy


def random_imput(X):
    X_res = X
    na_values_cord = np.argwhere(np.isnan(X_res))
    for cord in na_values_cord:
        X_res[cord[0]][cord[1]] = random.random()
    return X_res
    

def genrer_valeur_null(x, taux):
    x = x.astype(float)
    N = x.shape[0] * x.shape[1]
    n_retire = int(N * taux)
    for i in range(n_retire):
        a = random.randint(0,x.shape[0]-1)
        b = random.randint(0,x.shape[1]-1)
        x[a][b] = np.nan
    return x

class CoclustInfoImput(CoclustInfo):
    
    def __init__(self, n_row_clusters=2, n_col_clusters=2, init=None,
                 max_iter=20, n_init=1, tol=1e-9, random_state=None):
        super().__init__(n_row_clusters, n_col_clusters, init, max_iter, n_init, tol, random_state)
        #Imputer = Imput_method
        
        
    def fit(self, X, y=None):
        """Perform co-clustering.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """
        random_state = check_random_state(self.random_state)

        check_array(X, accept_sparse=True, dtype="numeric", order=None,
                    copy=False, force_all_finite='allow-nan', ensure_2d=True,
                    allow_nd=False, ensure_min_samples=self.n_row_clusters,
                    ensure_min_features=self.n_col_clusters,
                    warn_on_dtype=False, estimator=None)

        #check_positive(X)

        X = X.astype(float)

        criterion = self.criterion
        criterions = self.criterions
        row_labels_ = self.row_labels_
        column_labels_ = self.column_labels_
        delta_kl_ = self.delta_kl_

        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for seed in seeds:
            self._fit_single(X, seed, y)
            if np.isnan(self.criterion):
                raise ValueError("matrix may contain negative or "
                                 "unexpected NaN values")
            # remember attributes corresponding to the best criterion
            if (self.criterion > criterion):
                criterion = self.criterion
                criterions = self.criterions
                row_labels_ = self.row_labels_
                column_labels_ = self.column_labels_
                delta_kl_ = self.delta_kl_

        # update attributes
        self.criterion = criterion
        self.criterions = criterions
        self.row_labels_ = row_labels_
        self.column_labels_ = column_labels_
        self.delta_kl_ = delta_kl_

        return self
    
    
    def random_init(self, n_clusters, n_cols, random_state=None):

        random_state = check_random_state(random_state)
        W_a = random_state.randint(n_clusters, size=n_cols)
        W = np.zeros((n_cols, n_clusters))
        W[np.arange(n_cols), W_a] = 1
        
        return W
    
    
    def _fit_single(self, X, random_state, y=None):
        """Perform one run of co-clustering.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """
        
        X_trace = X ## X_trace garde la trace des valeurs null dans X avant l'imputation.
        
        K = self.n_row_clusters
        L = self.n_col_clusters
        
        if self.init is None:
            W = self.random_init(L, X.shape[1], random_state)
        else:
            W = np.matrix(self.init, dtype=float)
        
        X = sp.csr_matrix(X)
        
        N = float(X.sum()) # Dans le cas ou la matrice contient que des 1 et des 0, N = le nombre de 1 donc le nombre de données non null.
        X = X.multiply(1. / N) # Normalisation

        Z = sp.lil_matrix(self.random_init(K, X.shape[0], self.random_state)) # K: Nombre de lignes
    
        W = sp.csr_matrix(W) 
        
        # Imputation pour l'initialisation
        # vu que c une phase d'initialisation pourquoi faire un random si on peut utiliser le KNN imputeur qui pourra accelerer la convergences.
        X = random_imput(X.toarray())
        X = sp.csr_matrix(X)

        # Initial delta
        p_il = X * W # columns
        # p_il = p_il     # matrix m,l ; column l' contains the p_il'
        p_kj = X.T * Z  # matrix j,k

        p_kd = p_kj.sum(axis=0)  # array containing the p_k.
        p_dl = p_il.sum(axis=0)  # array containing the p_.l

        # p_k. p_.l ; transpose because p_kd is "horizontal"
        p_kd_times_p_dl = p_kd.T * p_dl
        min_p_kd_times_p_dl = np.nanmin(
            p_kd_times_p_dl[
                np.nonzero(p_kd_times_p_dl)])
        p_kd_times_p_dl[p_kd_times_p_dl == 0.] = min_p_kd_times_p_dl * 0.01
        p_kd_times_p_dl_inv = 1. / p_kd_times_p_dl

        p_kl = (Z.T * X) * W
        delta_kl = p_kl.multiply(p_kd_times_p_dl_inv)

        change = True
        news = []

        n_iters = self.max_iter
        pkl_mi_previous = float(-np.inf)

        # Loop
        while change and n_iters > 0:
            change = False
            ## X' = X
            ## Imputation(X)
            # Update Z
            p_il = X * W  # matrix m,l ; column l' contains the p_il'
            if not isdense(delta_kl):
                delta_kl = delta_kl.todense()
            
            delta_kl[delta_kl == 0.] = 0.0001  # to prevent log(0)
            log_delta_kl = np.log(delta_kl.T)
            log_delta_kl = sp.lil_matrix(log_delta_kl)
            # p_il * (d_kl)T ; we examine each cluster
            Z1 = p_il * log_delta_kl
            Z1 = Z1.toarray()
            Z = np.zeros_like(Z1)
            # Z[(line index 1...), (max col index for 1...)]
            Z[np.arange(len(Z1)), Z1.argmax(1)] = 1
            Z = sp.lil_matrix(Z)
            
            # Update delta
            # matrice d, k ; column k' contains the p_jk'
            p_kj = X.T * Z
            # p_il unchanged
            p_dl = p_il.sum(axis=0)  # array l containing the  p_.l
            p_kd = p_kj.sum(axis=0)  # array k containing the p_k.

            # p_k. p_.l ; transpose because p_kd is "horizontal"
            p_kd_times_p_dl = p_kd.T * p_dl
            min_p_kd_times_p_dl = np.nanmin(
                p_kd_times_p_dl[
                    np.nonzero(p_kd_times_p_dl)])
            p_kd_times_p_dl[p_kd_times_p_dl == 0.] = min_p_kd_times_p_dl * 0.01
            p_kd_times_p_dl_inv = 1. / p_kd_times_p_dl
            p_kl = (Z.T * X) * W
            delta_kl = p_kl.multiply(p_kd_times_p_dl_inv)
            
            #Imputation partie 2
            X = imput_block(X.toarray(), Z.toarray().argmax(axis=1)
                            , W.toarray().argmax(axis=1)
                            , X_trace)
            X = sp.csr_matrix(X)

            #Update W
            p_kj = X.T * Z  # matrice m,l ; column l' contains the p_il'
            if not isdense(delta_kl):
                delta_kl = delta_kl.todense()
            delta_kl[delta_kl == 0.] = 0.0001  # to prevent log(0)
            log_delta_kl = np.log(delta_kl)
            log_delta_kl = sp.lil_matrix(log_delta_kl)
            W1 = p_kj * log_delta_kl  # p_kj * d_kl ; we examine each cluster
            W1 = W1.toarray()
            W = np.zeros_like(W1)
            W[np.arange(len(W1)), W1.argmax(1)] = 1
            W = sp.lil_matrix(W)

            # Update delta
            p_il = X * W     # matrix d,k ; column k' contains the p_jk'
            # p_kj unchanged
            p_dl = p_il.sum(axis=0)  # array l containing the p_.l
            p_kd = p_kj.sum(axis=0)  # array k containing the p_k.

            # p_k. p_.l ; transpose because p_kd is "horizontal"
            p_kd_times_p_dl = p_kd.T * p_dl
            min_p_kd_times_p_dl = np.nanmin(
                p_kd_times_p_dl[
                    np.nonzero(p_kd_times_p_dl)])
            p_kd_times_p_dl[p_kd_times_p_dl == 0.] = min_p_kd_times_p_dl * 0.01
            p_kd_times_p_dl_inv = 1. / p_kd_times_p_dl
            p_kl = (Z.T * X) * W

            delta_kl = p_kl.multiply(p_kd_times_p_dl_inv)
            
            
            #Imputation partie 4
            X = imput_block(X.toarray(), Z.toarray().argmax(axis=1)
                            , W.toarray().argmax(axis=1)
                            , X_trace)
            X = sp.csr_matrix(X)
            # to prevent log(0) when computing criterion
            if not isdense(delta_kl):
                delta_kl = delta_kl.todense()
            delta_kl[delta_kl == 0.] = 0.0001


            # Criterion
            pkl_mi = sp.lil_matrix(p_kl).multiply(
                sp.lil_matrix(np.log(delta_kl)))
            pkl_mi = pkl_mi.sum()

            if np.abs(pkl_mi - pkl_mi_previous) > self.tol:
                pkl_mi_previous = pkl_mi
                change = True
                news.append(pkl_mi)
                n_iters -= 1

        self.criterions = news
        self.criterion = pkl_mi
        self.row_labels_ = Z.toarray().argmax(axis=1).tolist()
        self.column_labels_ = W.toarray().argmax(axis=1).tolist()
        self.delta_kl_ = delta_kl
        self.X = X
        self.Z = Z
        self.W = W
    

class CoclustModImput(CoclustMod):

    def __init__(self, n_clusters=2, init=None, max_iter=20, n_init=1,
                 tol=1e-9, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state

        self.row_labels_ = None
        self.column_labels_ = None
        self.modularity = -np.inf
        self.modularities = []
    

    def fit(self, X, y=None):
        """Perform co-clustering by direct maximization of graph modularity.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """

        random_state = check_random_state(self.random_state)
        
        check_array(X, accept_sparse=True, dtype="numeric", order=None,
                    copy=False, force_all_finite='allow-nan', ensure_2d=True,
                    allow_nd=False, ensure_min_samples=self.n_clusters,
                    ensure_min_features=self.n_clusters,
                    warn_on_dtype=False, estimator=None)
        

#         if type(X) == np.ndarray:
#             X = np.matrix(X)

        X = np.array(X)

        X = X.astype(float)

        modularity = self.modularity
        modularities = []
        row_labels_ = None
        column_labels_ = None

        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for seed in seeds:
            self._fit_single(X, seed, y)
            if np.isnan(self.modularity):
                raise ValueError("matrix may contain unexpected NaN values")
            # remember attributes corresponding to the best modularity
            if (self.modularity > modularity):
                modularity = self.modularity
                modularities = self.modularities
                row_labels_ = self.row_labels_
                column_labels_ = self.column_labels_

        # update attributes
        self.modularity = modularity
        self.modularities = modularities
        self.row_labels_ = row_labels_
        self.column_labels_ = column_labels_

        return self

    def _fit_single(self, X, random_state, y=None):
        """Perform one run of co-clustering by direct maximization of graph
        modularity.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """
        X_trace = X ## trace des valeurs manquantes

        if self.init is None:
            W = random_init(self.n_clusters, X.shape[1], random_state)
        else:
            W = np.matrix(self.init, dtype=float)

        Z = np.zeros((X.shape[0], self.n_clusters))
        
        X = sp.csr_matrix(X)

        ## Imputation random
        X = random_imput(X.toarray())
        X = sp.csr_matrix(X)

        # Compute the modularity matrix
        row_sums = np.matrix(X.sum(axis=1))
        col_sums = np.matrix(X.sum(axis=0))
        N = float(X.sum())
        indep = (row_sums.dot(col_sums)) / N

        # B is a numpy matrix
        B = X - indep

        self.modularities = []

        # Loop
        m_begin = float("-inf")
        change = True
        iteration = 0
        while change:
            change = False

            # Reassign rows 
            BW = B.dot(W)
            for idx, k in enumerate(np.argmax(BW, axis=1)):
                Z[idx, :] = 0
                Z[idx, k] = 1
            
            ## Imputation block
            B = imput_block_mod(np.array(B), Z.argmax(axis=1)
                            , W.argmax(axis=1)
                            , X_trace)
            B = np.matrix(B)
            

            # Reassign columns
            BtZ = (B.T).dot(Z)
            for idx, k in enumerate(np.argmax(BtZ, axis=1)):
                W[idx, :] = 0
                W[idx, k] = 1
                
            ##Imputation block
            B = imput_block_mod(np.array(B), Z.argmax(axis=1)
                            , W.argmax(axis=1)
                            , X_trace)
            B = np.matrix(B)

            k_times_k = (Z.T).dot(BW)
            m_end = np.trace(k_times_k)
            iteration += 1
            if (np.abs(m_end - m_begin) > self.tol and
                    iteration < self.max_iter):
                self.modularities.append(m_end/N)
                m_begin = m_end
                change = True

        self.row_labels_ = np.argmax(Z, axis=1).tolist()
        self.column_labels_ = np.argmax(W, axis=1).tolist()
        self.btz = BtZ
        self.bw = BW
        self.modularity = m_end / N
        self.nb_iterations = iteration
        self.X=B

    def get_assignment_matrix(self, kind, i):
        """Returns the indices of 'best' i cols of an assignment matrix
        (row or column).

        Parameters
        ----------
        kind : string
             Assignment matrix to be used: rows or cols

        Returns
        -------
        numpy array or scipy sparse matrix
            Matrix containing the i 'best' columns of a row or column
            assignment matrix
        """
        if kind == "rows":
            s_bw = np.argsort(self.bw)
            return s_bw[:, -1:-(i+1):-1]
        if kind == "cols":
            s_btz = np.argsort(self.btz)
            return s_btz[:, -1:-(i+1):-1]


def rmes_data_original_imputer(data_original, data_imputer, data_Nan):
    indice_Nan_Values = np.argwhere(np.isnan(data_Nan))
    list_imputer =[]
    list_original= []
    for i in range(0,indice_Nan_Values.shape[0]):  
        list_imputer.append(data_imputer[indice_Nan_Values[i][0]][indice_Nan_Values[i][1]])
        list_original.append(data_original.toarray()[indice_Nan_Values[i][0]][indice_Nan_Values[i][1]])
    
    return mean_squared_error(list_original, list_imputer)

def data_missing_percent(dtm, ratio, name):
    data_missing = genrer_valeur_null(dtm.toarray(), ratio)
    pd.DataFrame(data_missing).to_csv("data_imputation/DataValeurManquante_{}_{}percent.csv".format(name,int(ratio*100)),header=True, index=False)
    print(" {}% is Done ".format(int(ratio*100)))


def models_comparaison(data,ratio,algo,y_true,n_row_clusters,n_col_clusters):
    model = algo(n_row_clusters = 4, n_col_clusters = 4,n_init = 1, random_state = 0)
    model.fit(data)
    metriques(model,y_true,model.row_labels_)
    Visualisation_Matrix(model,data, ratio)
    #confusion_matrix(y_true, model.row_labels_)

def random_imput(X):
    X_res = X
    na_values_cord = np.argwhere(np.isnan(X_res))
    for cord in na_values_cord:
        X_res[cord[0]][cord[1]] = random.random()
    return X_res
    

def genrer_valeur_null(x, taux):
    x = x.astype(float)
    N = x.shape[0] * x.shape[1]
    n_retire = int(N * taux)
    for i in range(n_retire):
        a = random.randint(0,x.shape[0]-1)
        b = random.randint(0,x.shape[1]-1)
        x[a][b] = np.nan
    return x

def models_comparaison_Mod(data,ratio, algo, y_true, n_clusters):
    model = algo(n_clusters = n_clusters, n_init = 1, random_state = 0)
    model.fit(data)
    metriques(model,y_true,model.row_labels_)
    Visualisation_Matrix(model,data, ratio)
    #confusion_matrix(y_true, model.row_labels_)

