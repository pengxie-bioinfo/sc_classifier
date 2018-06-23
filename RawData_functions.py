import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.manifold import TSNE
import pickle
import re
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from scipy import stats

def sps2dns(sps):
    sps = sparse.coo_matrix((sps[1:, 2], (sps[1:, 0] - 1, sps[1:, 1] - 1)),
                            shape=(sps[0, 0], sps[0, 1]))  # Time-consuming
    dns = sps.todense()
    dns = np.array(dns, dtype='float32')
    return (dns)


def norm_rpm(dns, transpose=True):
    dns = dns / np.sum(dns, axis=0) * 10.0 ** 6  # Caculate RPM
    if transpose:
        dns = np.transpose(dns)
    return (dns)


def load_matrix(sps_file):
    sps = np.loadtxt(sps_file, skiprows=2, dtype=int)
    dns = sps2dns(sps)
    dns = norm_rpm(dns)
    return (dns)


def load_gene(gene_file):
    genes = []
    for line in open(gene_file, 'r'):
        line = line.rstrip()
        line = line.split("\t")[1]
        genes.append(line)
    return (genes)


def load_cluster(cluster_file):
    label = np.loadtxt(cluster_file, skiprows=1, delimiter=",", usecols=(1,), dtype=int)
    return (label)


class scRNAseq:
    """Single-cell RNAseq data including: matrix, gene names and cluster_id"""

    def __init__(self, dns, genes, label):

        self.dns, self.genes, self.label = dns, genes, label

        # Sample labels
        nclust = len(set(label))
        label_mat = np.zeros((len(label), nclust))
        for i in range(len(label)):
            label_mat[i, label[i] - 1] = 1
        self.nclust, self.label_mat = nclust, label_mat

        # RPM distribution
        tp = dns.reshape(-1, 1)
        tp = np.log10(tp + 1)
        hist = plt.hist(tp, bins=50, normed=True)
        plt.xlabel('Log10 RPM')
        plt.ylabel('Probability Density')
        plt.ylim(0, sorted(hist[0])[-2])

        # Select genes
        dns_sub = np.transpose(dns)
        '''
        lab = []
        for i in range(len(genes)):
            for j in range(1, nclust+1):
                tp = [k for k,l in enumerate(label) if l==j]
                tp = dns[tp, i]
                if(np.mean(tp)>500):
                    lab.append(i)
                    break
        '''
        lab = (np.mean(dns_sub, axis=1) > 100)
        dns_sub = dns_sub[lab, :]
        dns_sub = np.transpose(dns_sub)
        genes_sub = [genes[j] for j, k in enumerate(lab) if k]
        print("Selected expressed genes", dns_sub.shape)
        self.dns_sub = dns_sub

    def tSNE(self, plot_name, xlim=[-20, 20], random_state=0):
        model = TSNE(n_components=2, method='barnes_hut', random_state=random_state)
        self.tsne = model.fit_transform(self.dns_sub[:, :])

        mpl.rcParams.update({'font.size': 13})
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        cmap = plt.get_cmap(name='rainbow')
        color = cmap(np.linspace(0.01, 0.99, self.nclust))
        plt.subplot(111)
        for i in range(1, self.nclust + 1):
            lab = [j for j, k in enumerate(self.label) if k == i]
            plt.scatter(self.tsne[lab, 0], self.tsne[lab, 1], label='cluster-' + str(i), edgecolors='k',
                        color=color[i - 1])
        plt.xlim(xlim)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(loc="best")
        plt.savefig(plot_name)
        return (self.tsne)

def load_pickle(file_path, select_label=[0], subsampling=0, avoid_sample=[]):
    dns, genes, label = pickle.load(open(file_path, "rb"))
    cell_id = range(len(dns))
    # 1. Avoid using user-specified samples: useful for setting train/test data
    cell_id = list(set(cell_id) - set(avoid_sample))
    dns = dns[cell_id, :]
    label = [label[i] for i in cell_id]
    # 2. Select user-specified cell types/labels
    if select_label.count(0) == 0:
        lab = [i for i,j in enumerate(label) if select_label.count(j)==1]
        dns = dns[lab, :]
        label = [label[i] for i in lab]
        cell_id = [cell_id[i] for i in lab]
    # 3. Sub-sampling
    if subsampling != 0:
        if len(dns) > subsampling:
            lab = np.random.choice(range(len(dns)), size=subsampling, replace=False)
            dns = dns[lab,:]
            label = [label[i] for i in lab]
            cell_id = [cell_id[i] for i in lab]
    return([dns, genes, label, cell_id])

def load_pickle_control_each_type(file_path, select_label=[0], n_each_type=0, avoid_sample=[]):
    dns, genes, label = pickle.load(open(file_path, "rb"))
    cell_id = range(len(dns))
    # 1. Avoid using user-specified samples: useful for setting train/test data
    cell_id = list(set(cell_id) - set(avoid_sample))
    dns = dns[cell_id, :]
    label = [label[i] for i in cell_id]
    # 2. Select user-specified cell types/labels
    if select_label.count(0) == 0:   # select_label = [0] is defined as selecting all labels
        lab = [i for i,j in enumerate(label) if select_label.count(j)==1]
        dns = dns[lab, :]
        label = [label[i] for i in lab]
        cell_id = [cell_id[i] for i in lab]
    # 3. Sub-sampling
    label_array = np.array(label)
    if n_each_type != 0:
        lab = []
        for i in select_label:
            tp = np.arange(len(dns))
            tp = tp[label_array == i]
            if len(tp) > n_each_type:
                lab = lab + np.random.choice(tp, size=n_each_type, replace=False).tolist()
            else:
                lab = lab + tp.tolist()
        dns = dns[lab,:]
        label = [label[i] for i in lab]
        cell_id = [cell_id[i] for i in lab]
    return([dns, genes, label, cell_id])

# def select_genes(x, x_genes, ref_genes):
#     if len(x_genes) == len(ref_genes):
#         if np.sum(np.array(x_genes) == np.array(ref_genes)) == len(x_genes):
#             return(x, x_genes)
#     y = np.zeros((len(x), len(ref_genes)))
#     for i in range(len(ref_genes)):
#         if x_genes.count(ref_genes[i]) > 0:
#             tp = x[:,x_genes.index(ref_genes[i])]
#             y[:,i] = tp
#     return(y, ref_genes)
def select_genes(x, x_genes, ref_genes):
    if len(x_genes) == len(ref_genes):
        if np.sum(np.array(x_genes) == np.array(ref_genes)) == len(x_genes):
            print('Gene IDs identical to reference.')
            return(x, x_genes)
    y = np.zeros((len(x), len(ref_genes)))
    ct = 0
    for i in range(len(ref_genes)):
        if x_genes.count(ref_genes[i]) > 0:
            tp = x[:,x_genes.index(ref_genes[i])]
            y[:,i] = tp
        else:
            ct = ct+1
    print('Number of NA genes: ', ct)
    return(y, ref_genes)

def remove_mt_rp(x, genes):
    '''
    # Get rid of mitochondrial and ribosomal genes
    :param x:
    :param genes:
    :return:
    '''
    lab = []
    for i in range(len(genes)):
        mt = re.compile('mt', re.IGNORECASE)
        rpl = re.compile('rpl', re.IGNORECASE)
        rps = re.compile('rps', re.IGNORECASE)
        if ((re.match(mt, genes[i]) == None)  # remove mitochondrial genes
                & (re.match(rpl, genes[i]) == None)
                & (re.match(rps, genes[i]) == None)
            ):
            lab.append(i)
    x = x[:, lab]
    genes = [genes[i] for i in lab]
    return(x, genes)

def pos_specific_genes(x, y, genes):
    # Find specific genes for positive samples
    t1 = x[y[:,1]==1,:]
    t2 = x[y[:,1]==0,:]
    ks = []
    # fc = []
    for i in range(len(genes)):
        tp1 = t1[:,i]
        tp2 = t2[:,i]
        if np.mean(tp1) > 0:
            ks.append(stats.ks_2samp(tp1, tp2)[1])
            # fc.append((np.mean(tp1)+10**(-5))/(np.mean(tp2)+10))
        else:
            ks.append(1)
            # fc.append(1)
    deg = pd.DataFrame({'exp_Pos':np.mean(t1,axis=0), 'exp_Neg':np.mean(t2, axis=0),
                        # 'FC':fc,
                        'P-value':ks}, index=genes)
    deg = deg.reindex_axis(['exp_Pos', 'exp_Neg',
                            # 'FC',
                            'P-value'], axis=1)
    deg = deg.sort_values('P-value', ascending=True)
    return(deg)

def label_2_matrix(label, lab_list=None):
    # convert labels to matrix format
    if lab_list==None:
        lab_list = list(set(label))
    else:
        assert (set(label).issubset(lab_list)),'the specified lab_list does not include all the labels'
    label = np.array(label)
    y = np.zeros((len(label), len(lab_list)))
    for i in range(len(label)):
        y[i, lab_list.index(label[i])] = 1
    return(y)

def input_formatting(input_file, output_file):
    df = pd.read_csv(input_file, header=0, index_col=0)
    print(df.shape)
    data = [np.transpose(np.array(df)), df.index.tolist()]
    print("Number of cells:\t", data[0].shape[0])
    print("Number of genes:\t", data[0].shape[1])
    pickle.dump(data, open(output_file, 'wb'))
    return