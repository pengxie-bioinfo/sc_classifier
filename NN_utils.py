import numpy as np
import tensorflow as tf
import pickle
import math
from RawData_functions import select_genes

'''
1. Functions for building a network
'''
def shift_scale(train_data):
    train_x, train_g, train_y, sample_info = train_data
    input_shift = np.mean(train_x, axis=0).reshape(1,-1)
    input_scale = np.std(train_x, axis=0).reshape(1,-1)
    return(input_shift, input_scale)


def weight_variables(shape, given_initial=None):
    if given_initial == None:
        initial = tf.truncated_normal(shape, stddev=0.1)
    else:
        initial = given_initial
    return tf.Variable(initial, name='W')


def bias_variables(shape, given_initial=None):
    if given_initial == None:
        initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
    else:
        initial = given_initial
    return tf.Variable(initial, name='b')


# this is a simpler version of Tensorflow's 'official' version. See:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102
def batch_norm_wrapper(inputs, is_training=True, decay=0.99):
    epsilon = 0.001
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale, epsilon)


def build_graph(w_input, w_hidden=[200, 100], w_output=2, given_initial=None):
    '''
    :param n_feature: in the cell-typing case, this is the number of genes at input layer
    :param n_output:
    :return:
    '''
    # Placeholders
    x_shift = tf.placeholder(tf.float32, [None, w_input], name='x_shift')
    x_scale = tf.placeholder(tf.float32, [None, w_input], name='x_scale')
    xs = tf.placeholder(tf.float32, [None, w_input], name='xs')
    ys = tf.placeholder(tf.float32, [None, w_output], name='ys')
    class_factor = tf.placeholder(tf.float32, [None, w_output], name='class_factor')  # A factor for balancing class training samples
    kp = tf.placeholder(tf.float32, len(w_hidden), name='kp')
    param = []


    # Input ---> hidden layers
    current_input_train = xs
    current_input_test = xs
    for layer_i, n_output in enumerate(w_hidden):
        # 1. Initialize parameters
        n_input = int(current_input_train.get_shape()[1])
        if given_initial == None:
            W_init, b_init = None, None
        else:
            W_init, b_init = given_initial[layer_i]
        W = weight_variables([n_input, n_output], W_init)
        b = bias_variables([n_output], b_init)
        param.append([W, b])

        # 2. Forward propagation
        # For training purpose, we perform batch normalization
        z_train = tf.matmul(current_input_train, W) + b
        bn = batch_norm_wrapper(z_train)
        h_fc = tf.nn.relu(bn)
        h_fc_drop = tf.nn.dropout(h_fc, kp[layer_i])
        current_input_train = h_fc_drop
        # For testing purpose, we do not perform batch normalization. Important!
        z_test = tf.matmul(current_input_test, W) + b
        current_input_test = tf.nn.relu(z_test)

    # Hidden layers ---> Output
    W = weight_variables([w_hidden[-1], w_output])
    b = bias_variables([w_output])
    h_fc_train = tf.nn.relu(tf.matmul(current_input_train, W) + b, name='Final_activation_train')
    h_fc_test = tf.nn.relu(tf.matmul(current_input_test, W) + b, name='Final_activation_test')
    prediction_train = tf.nn.softmax(h_fc_train, name='Prediction_train')
    prediction_test = tf.nn.softmax(h_fc_test, name='Prediction_test')

    ########################################################################
    #   OP's for training
    ########################################################################
    with tf.name_scope('Loss'):  # Cross-entropy loss
        loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction_train+1e-20) * class_factor,
                                             reduction_indices=[1]), name='loss')
        tf.summary.scalar('Loss/', loss)
    with tf.name_scope('Train'):
        train_step = tf.train.AdamOptimizer(5e-4, name='AdamOptimizer').minimize(loss, name='min_loss')

    ########################################################################
    #   OP's for testing
    ########################################################################
    y_pred = tf.argmax(prediction_test, 1)
    y_true = tf.argmax(ys, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32), name='Accuracy')
    return (xs, ys), class_factor, x_shift, x_scale, kp, train_step, loss, prediction_test, accuracy, tf.train.Saver()


'''
3. Predict testing data
'''

def nn_pred(model_tag, test_data):
    # Load model
    global train_gene, input_shift, input_scale, w_hidden, w_output
    train_gene, input_shift, input_scale, w_hidden, w_output = pickle.load(open('data/Input_parameter_'+model_tag+".pickle", "rb"))
    tf.reset_default_graph()
    global saver, graph, predition, xs, ys, x_shift, x_scale, kp
    saver = tf.train.import_meta_graph('./data/nn-save/' + model_tag + '/model.meta')
    graph = tf.get_default_graph()
    prediction = graph.get_tensor_by_name("Prediction_test:0")
    xs = graph.get_tensor_by_name("xs:0")
    ys = graph.get_tensor_by_name("ys:0")
    x_shift = graph.get_tensor_by_name("x_shift:0")
    x_scale = graph.get_tensor_by_name("x_scale:0")
    kp = graph.get_tensor_by_name("kp:0")
    # Restore values of graph
    global sess
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './data/nn-save/' + model_tag + '/model')
    # Testing data
    test_x, _, test_y, _ = test_data
    # Make predictions
    keep_1 = [1]*len(w_hidden)
    feed = {xs: test_x,
            ys: test_y,
            x_shift: input_shift,
            x_scale: input_scale,
            kp:keep_1}
    preds = sess.run(prediction, feed_dict=feed)
    pred_label = np.argmax(preds, axis=1)
    return [preds, pred_label]

'''
4. Functions for training a network
'''
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (number of examples, input size)
    Y -- true "label"  of shape (number of examples, number of cell types)
    mini_batch_size - size of the mini-batches, integer
    seed -- so we can permutate mini-batch assignments.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

'''
5. Functions for testing model performance
'''

def tf_confusion_metrics(model, actual_classes, session, feed_dict, pos_cutoff=0.5):
    actuals = tf.argmax(actual_classes, 1)
    # predictions = tf.argmax(model, 1)
    predictions = tf.greater(model[:, 1], pos_cutoff)
    predictions = tf.cast(predictions, dtype=tf.int32)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    tp, tn, fp, fn = \
        session.run(
            [tp_op, tn_op, fp_op, fn_op],
            feed_dict={}
        )

    tpr = float(tp) / (float(tp) + float(fn) + 10 ** (-10))
    fpr = float(fp) / (float(fp) + float(tn) + 10 ** (-10))

    accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn) + 10 ** (-10))

    recall = tpr
    precision = float(tp) / (float(tp) + float(fp) + 10 ** (-10))

    f1_score = (2 * (precision * recall)) / (precision + recall + 10 ** (-10))

    print('Precision = ', precision)
    print('Recall = ', recall)
    print('F1 Score = ', f1_score)
    print('Accuracy = ', accuracy)

    return ([[tp, fp], [fn, tn]])

