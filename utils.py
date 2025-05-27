#! /usr/bin/env python
# -*- coding: utf-8 -*-

#############################
# utils files.
#############################
from numpy import *
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
import pickle
import math
from utils import *
import os
from sklearn.cluster import DBSCAN
import random
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from typing import List, Tuple
from numpy import arange, argsort, argwhere, empty, full, inf, intersect1d, max, ndarray, sort, sum, zeros
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
import umap

def load_dataset(filepath):
    """
        Return:
            dataset: dict
    """
    with open(filepath, 'rb') as fr:
        dataset = pickle.load(fr)
    return dataset
# 按行的方式计算两个坐标点之间的距离
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def results(data_path):
    parameters=[]# number of centers in clients, number of centers in servers, k in SNN
    datapkl = load_dataset(data_path)
    #order = datapkl['order']
    data = list(datapkl['full_data'])
    # if len(data)>10000:
    #     u=1
    #     reducer = umap.UMAP(random_state=42)
    #     data = reducer.fit_transform(data)

    corepoints = [] # save local kmeans centers.
    for i_client in range(datapkl['num_clusters']):
        # add noise to local data (Differential privacy)# 对中心点进行差分隐私保护,即为加上噪声。

        lodata = datapkl["client_" + str(i_client)]
        # if u==1:
        #     reducer = umap.UMAP(random_state=42)
        #     lodata = reducer.fit_transform(lodata)
        # noise = np.random.laplace(0, 1 / 20, lodata.shape[0] * lodata.shape[1])
        # noise = noise.reshape(lodata.shape[0], lodata.shape[1])
        # lodata = lodata + noise
        #计算每个局部数据点的反向k近邻
        nk = min([len(lodata) // 2, 200])

        # print('客户端kmeans中的k值',len(lodata)-5)
        cluster = KMeans(nk).fit(lodata)
        # print('center',cluster.cluster_centers_)
        corepoints.append(cluster.cluster_centers_)


    # server: process the information from clients
    serverdata = np.concatenate(corepoints, axis=0)
    #serverdata =np.array(corepoints)
    label = datapkl['true_label']
    parameters.append(nk)
    cnum=len(set(label))+1
    parameters.append(cnum)
    #print('snndpc中的参数k',k )
    #centroid= SNN(k, cnum, serverdata)
    k=5
    parameters.append(k)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(serverdata)
    distances, indices = nbrs.kneighbors(serverdata)
    rnn = []  # 每个数据点的反向最近邻
    for i in range(serverdata.shape[0]):
        templist = []
        for j in range(len(indices)):
            if i in indices[j][1:]:
                templist.append(j)
        rnn.append(templist)
    # 计算每个数据点的局部密度locald
    locald = []
    distances = cdist(serverdata, serverdata)
    for i in range(len(rnn)):
        a = len(rnn[i]) ** 2
        b = sum(distances[i, rnn[i]])
        if a == 0 and b == 0:
            a = 0
            b = 1
        locald.append(a / b)
    #求每个数据点的MNN
    mnn=[]
    for i in range(serverdata.shape[0]):
        templist=[]
        for j in range(len(indices)):
            if i!=j and i in indices[j] and j in indices[i]:
                templist.append(j)
        mnn.append(templist)
    #再求局部密度
    finallocald=[]
    for i in range(len(mnn)):
        a=len(mnn[i])
        locald=np.array(locald)
        b=sum(locald[mnn[i]])
        if a == 0 and b == 0:
            a = 0
            b = 1
        finallocald.append(a/b)

    sorted_list = sorted(locald, reverse=True)
    new_indices = [sorted_list.index(x) for x in locald]
    selected_index = new_indices[:cnum]
    # 选择前n个最大的局部密度作为代表点，上传到服务器端。
    finalcenter=[]
    for i in selected_index:
        finalcenter.append(serverdata[i])

    #finalcenter = cluster2.cluster_centers_
    # 根据finalcenter分配所有客户端的数据点。
    idx = []
    for i in data:
        simi = []
        for j in finalcenter:
            simi.append(np.linalg.norm(i - j))
        idx.append(simi.index(min(simi)) + 1)
    arr = np.array(idx)
    ari = round(adjusted_rand_score(label, arr),4 )
    nmi = round(normalized_mutual_info_score(label, arr),4)
    ami=  round(adjusted_mutual_info_score(label, arr),4)
    return ari, nmi,ami,parameters