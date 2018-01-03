import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from NN_utils import nn_pred
from RawData_functions import label_2_matrix, remove_mt_rp, input_formatting
plt.style.use('bmh')
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 18

if __name__ == '__main__':
    model_tag = 'T_cell'  # Tag of the pre-trained model
    test_tag = 'test_data'    # Tag of testing data
    print('Predicting %s in %s.' % (model_tag, test_tag))

    ###################################################
    # Load testing data
    ###################################################
    # Format conversion
    input_formatting('data/'+test_tag+'.csv', 'data/'+test_tag+'.pickle')
    data = pickle.load(open('data/' + test_tag + ".pickle", 'rb'))
    # Formatting testing labels
#    true_lab = [135, 163, 215, 233, 406]
    true_lab = [31, 8, 127, 7, 125, 46, 30, 123, 15, 90, 128, 135, 71, 4, 5, 38, 55,88]
    lab = [0] * len(data[0])
    for i in true_lab:
        lab[i] = 1
    lab_mat = label_2_matrix(lab)
    data = data + [lab_mat] + [list(range(len(lab)))]
    # Normalizing testing data
    data[0], data[1] = remove_mt_rp(data[0], data[1])
    data[0] = preprocessing.scale(data[0], axis=1)

    ###################################################
    # Make predictions
    ###################################################

    pred_prob, pred_lab = nn_pred(model_tag, data)
    df = pd.DataFrame({'Pred_prob':pred_prob[:,1], 'Pred_lab':pred_lab})
    df.to_csv('./Prediction_'+model_tag+'_'+test_tag+'.csv')
    print(pred_lab[true_lab])
    # Show results
    show_genes = ['CD3D', 'CD3E', 'CD3G', 'CD27', 'CD2', 'IL7R',
                  'CD48', 'CD52', 'CD53',  # General immune markers
                  'EPCAM', 'PROM1', 'KRT19',  # PDAC signature genes
                  'COL5A1'  # CAF (cancer-associated fibroblast)
                  ]

    tp = pd.DataFrame(data[0], columns=data[1])
    tp['Pred_prob'] = pred_prob[:, 1]
    tp['Manual'] = lab_mat[:, 1]

    tp = tp[['Manual'] + ['Pred_prob'] + show_genes]
    tp = tp.sort_values('Pred_prob', ascending=False)
    # tp = tp / tp.max(axis=0)
    tp = tp / np.percentile(np.array(tp), 100, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.imshow(np.array(tp.iloc[:, :]), origin='upper', cmap='Greens', vmax=1.1, aspect=tp.shape[1] / tp.shape[0] * 1.25)
    ax.set_xticks(range(tp.shape[1]))
    ax.set_yticks([])
    ticks = ax.set_xticklabels(tp.columns.tolist(), rotation=45, size=14)
    plt.show()



