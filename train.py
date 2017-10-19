# -*- coding: utf-8 -*-

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from model import Word2Vec, SGNS
from sklearn.decomposition import PCA

class PermutedCorpus(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords)

def train(use_gpu=False):
    num_epochs = 2
    batch_size = 256
    every = 10
    vocab = pickle.load(open('./stat/vocab_set.dat', 'rb'))
    V = len(vocab)
    word2vec = Word2Vec(V=V, use_gpu=use_gpu)
    perm_dict = pickle.load(open('./stat/permutation_dict.dat', 'rb'))

    start = time.time()
    for l in perm_dict:
        print("training sets with size {}...".format(l))
        sgns = SGNS(V=V, embedding=word2vec, batch_size=batch_size, window_size=l, n_negatives=5)
        optimizer = SGD(sgns.parameters(), 5e-1)
        dataset = PermutedCorpus(perm_dict[l])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        total_batches = len(dataset) // batch_size
        for epoch in range(1, num_epochs + 1):
            for batch, (iword, owords) in enumerate(dataloader):
                if len(iword) != batch_size:
                    continue
                loss = sgns(iword, owords)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if not batch % every:
                    print("\t[e{}][b{}/{}] loss: {:7.4f}\r".format(epoch, batch, total_batches, loss.data[0]))
    end = time.time()
    print("training done in {:.4f} seconds".format(end - start))  # It takes about 3.5 minutes with GPU, loss less than 7.5
    idx2vec = word2vec.forward([idx for idx in range(V + 1)])
    if use_gpu:
        idx2vec = idx2vec.cpu()
    pickle.dump(idx2vec.data.numpy(), open('./stat/idx2vec_{}epochs.dat'.format(num_epochs), 'wb'))

def graph_pca(idx2vec, idxtoing, searchword=""):
    X = idx2vec
    X = PCA(n_components=2).fit_transform(np.array(X))
    idxlist = []

    x_coord = []
    y_coord = []
    for idx, ing in idxtoing.items():
        if searchword in ing.lower():
            idxlist.append(idx)
            x_coord.append(X[idx+1, 0])
            y_coord.append(X[idx+1, 1])
    # x_coord = X[:, 0]
    # y_coord = X[:, 1]

    plt.scatter(x_coord, y_coord, alpha=1)
    for i in range(len(idxlist)):
        label = idxtoing[idxlist[i]]
        plt.annotate(label, (x_coord[i], y_coord[i]), size=7)
    plt.show()
    # plt.savefig('pca_ingvector.png')

def find_n_most_similar(idx2vec, w, n):
    cos_sim = list(map(lambda x: np.dot(w, x)/(np.linalg.norm(w) * np.linalg.norm(x)), idx2vec))
    return list(map(lambda x: cos_sim.index(x), sorted(cos_sim)[:n]))

if __name__ == '__main__':
    train()
    idx2vec = pickle.load(open('./stat/idx2vec_2epochs.dat', 'rb'))
    ingtoinx = pickle.load(open('./stat/ingredient_to_index_dict.dat', 'rb'))
    idxtoing = pickle.load(open('./stat/index_to_ingredient_dict.dat', 'rb'))
    graph_pca(idx2vec, idxtoing)

    # ai = ingtoinx["chopped cilantro"] + 1
    # xi = ingtoinx["cilantro"] + 1
    # yi = ingtoinx["onion"] + 1
    # w = idx2vec[ai] - idx2vec[xi] + idx2vec[yi]
    # wilist = find_n_most_similar(idx2vec, w, 10)
    # for i, wi in enumerate(wilist):
    #     print("{}. {} (index: {})".format(i+1, idxtoing[wi-1], wi-1))

    print("done")