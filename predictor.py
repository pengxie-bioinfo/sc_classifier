import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import sparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from NN_utils import nn_pred
from RawData_functions import label_2_matrix, remove_mt_rp, input_formatting, select_genes
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Directory of input file', default=None)
    parser.add_argument('--input_is_csv', type=str, help='If set to True, the input will be formatted before running', default=False)
    parser.add_argument('--output', type=str, help='Directory of output file', default=None)
    parser.add_argument('--predictor', type=str, help='Which predictor to use, e.g. T_cell', default=None)
    args = parser.parse_args()
    input = args.input
    output = args.output
    model_tag = args.predictor  # Tag of the pre-trained model

    ref_genes, _, _, _, _ = pickle.load(open('data/Input_parameter_'+model_tag+".pickle", "rb"))

    ###################################################
    # Load testing data
    ###################################################
    # Format conversion
    if args.input_is_csv:
        input_formatting(input, input+'.pickle')
        data = pickle.load(open(input+'.pickle', 'rb'))
        data[0], _ = select_genes(data[0], data[1], ref_genes)
    else:
        data = pickle.load(open(input, 'rb'))
    # Add dummy parameters
    if sparse.issparse(data[0]):
        data[0] = data[0].toarray()
    _, _, _, _, w_output = pickle.load(open('data/Input_parameter_'+model_tag+".pickle", "rb"))
    lab = [0] * len(data[0])
    lab[0] = 1
    lab_mat = label_2_matrix(lab_list=range(w_output), label=lab)
    data = data + [lab_mat] + [list(range(len(lab)))]
    # data[0], data[1] = remove_mt_rp(data[0], data[1])
    # data[0] = preprocessing.scale(data[0], axis=1)

    ###################################################
    # Make predictions
    ###################################################

    pred_prob, pred_lab = nn_pred(model_tag, data)
    if pred_prob.shape[1] == 2:  # Binary prediction
        print('Percentage of predicted %s:\t%f' % (model_tag, np.sum(pred_lab==1)/len(pred_lab)))
        df = pd.DataFrame({'Pred_prob': pred_prob[:, 1], 'Pred_lab': pred_lab})
    else:
        row_max = np.max(pred_prob, axis=1).reshape(-1,1)
        print('Percentage of prediction:\n', np.mean((pred_prob == row_max)*1, axis=0))
        df = pd.DataFrame(np.append(pred_prob, pred_lab.reshape(-1,1), axis=1))
    df.to_csv(output)
