import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.manifold import TSNE
import pickle
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import pyplot as plt


def tSNE_plot(x, label, names=None, seed=0):
    type_list = list(set(label))
    n_types = len(type_list)

    model = TSNE(n_components=2, method='barnes_hut', random_state=seed)
    tsne = model.fit_transform(x)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    cmap = plt.get_cmap(name='rainbow')
    color = cmap(np.linspace(0.01, 0.99, n_types))
    for i in range(n_types):
        lab = [j for j, k in enumerate(label) if k == type_list[i]]
        if names == None:
            ax.scatter(tsne[lab, 0], tsne[lab, 1], label='cluster-' + str(i), edgecolors='k', color=color[i])
        else:
            ax.scatter(tsne[lab, 0], tsne[lab, 1], label=names[i], edgecolors='k', color=color[i])
    ax.set_xlabel("tSNE-1")
    ax.set_ylabel("tSNE-2")
    ax.legend(loc="best")

def mark_plot(dns, genes, tsne, marker, celltype, ax=None, mexp=None, log=True):
    if mexp is None:
        mexp = dns[:, genes.index(marker)]
    if log:
        mexp = np.log(mexp + 1)
    if ax is None:
        fig, ax = plt.subplot()
    mexp = (mexp - np.mean(mexp)) / np.std(mexp)
    mexp = mexp.tolist()
    ax.scatter(tsne[:, 0], tsne[:, 1], c=mexp, cmap=cm.coolwarm, vmin=min(mexp),
               vmax=max(mexp), edgecolors='face', label=marker)
    ax.set_title(celltype)
    ax.legend(loc="upper left")


def mark_tsne_by_cluster(dns, genes, tsne, marker, celltype, cluster, label, ax):
    lab = [j for j, k in enumerate(label) if cluster.count(k) == 1]
    mexp = dns[:, genes.index(marker)]
    mexp = np.log(mexp + 1)
    mexp = (mexp - np.mean(mexp)) / np.std(mexp)
    mexp = mexp[lab].tolist()
    ax.scatter(tsne[lab, 0], tsne[lab, 1], c=mexp, cmap=cm.coolwarm, vmin=min(mexp), vmax=max(mexp), edgecolors='face')
    ax.set_title(celltype)
    ax.legend(loc="upper left")


def mark_hist_by_cluster(dns, genes, label, marker, celltype, cluster, ax):
    lab = [j for j, k in enumerate(label) if cluster.count(k) == 1]
    mexp = dns[:, genes.index(marker)]
    mexp = np.log(mexp + 1)
    mexp = (mexp - np.mean(mexp)) / np.std(mexp)
    others_mexp = np.delete(mexp, lab).tolist()
    mexp = mexp[lab].tolist()
    ax.hist(others_mexp, bins=20, normed=1, facecolor="grey", alpha=0.5, label=celltype)
    ax.hist(mexp, bins=20, normed=1, facecolor="navy", alpha=0.75, label='others')
    ax.set_title(celltype)


def show_removed_cells(dns, genes, label, tsne, marker, celltype, cluster, index, ax):
    lab = [j for j, k in enumerate(label) if cluster.count(k) == 1]
    lab_del = [j for j in range(len(label)) if (cluster.count(label[j]) == 1) & (index[j] == 1)]
    # print "Total number of cells:", len(lab)
    # print "Removed number of cells:", len(lab_del)
    mexp = dns[:, genes.index(marker)]
    mexp = np.log(mexp + 1)
    mexp = (mexp - np.mean(mexp)) / np.std(mexp)
    mexp = mexp[lab].tolist()
    ax.scatter(tsne[lab, 0], tsne[lab, 1], c=mexp, cmap=cm.coolwarm, vmin=min(mexp), vmax=max(mexp),
               label=marker, edgecolors='face')
    ax.scatter(tsne[lab_del, 0], tsne[lab_del, 1], marker="x")
    ax.set_title(celltype)
    ax.legend(loc='upper left')


def celltype_label(marker, celltype, dns, genes, label, tsne, cand_cluster, threshold, ax):
    n_cell = dns.shape[0]

    # Plot 1: histogram
    mark_hist_by_cluster(dns=dns, genes=genes, label=label, marker=marker, celltype=celltype,
                         cluster=cand_cluster, ax=ax[0])
    ax[0].plot([threshold] * 2, [0, 5], linestyle='dashed', color='red')

    # Plot 2: show removed samples
    mexp = dns[:, genes.index(marker)]
    mexp = np.log(mexp + 1)
    mexp = (mexp - np.mean(mexp)) / np.std(mexp)
    mexp = mexp.tolist()

    index = [j for j, k in enumerate(mexp) if k < threshold]
    lab_del = np.repeat(0, len(dns))
    lab_del[index] = 1
    show_removed_cells(dns=dns, genes=genes, label=label, tsne=tsne, marker=marker, celltype=celltype,
                       cluster=cand_cluster, index=lab_del, ax=ax[1])

    # 3: Keep cells in candidate clusters that express marker genes
    lab_celltype = np.repeat(0, n_cell)
    index = [j for j in range(n_cell) if (cand_cluster.count(label[j]) == 1) & (mexp[j] > threshold)]
    lab_celltype[index] = 1

    return lab_celltype