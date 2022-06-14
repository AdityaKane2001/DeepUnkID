import pandas as pd
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras.backend import set_session
import numpy as np
import random as rn
import random
import glob
import os

SEED = 20190222
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)

def set_allow_growth(device="1"):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.visible_device_list=device
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

def load_data(dataset):
    texts = []
    labels = []
    partition_to_n_row = {}
    for partition in ['train', 'valid', 'test']:
        with open("data/" + dataset + "/" + partition + ".seq.in") as fp:
            lines = fp.read().splitlines()
            texts.extend(lines)
            partition_to_n_row[partition] = len(lines)
        with open("data/" + dataset + "/" + partition + ".label") as fp:
            labels.extend(fp.read().splitlines())

    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df, partition_to_n_row


def load_20ng():
    ALL_SUBSETS = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
    random.seed(42)
    random.shuffle(ALL_SUBSETS)

    def readfile(filepath):
        with open(filepath, 'rb') as file:
            data = str(file.read()).replace('\\n', ' ').replace('\\r', ' ')
        return data

    def get_all_20ng(basepath):
        X = []
        y = []
        for subset_idx, subset in enumerate(ALL_SUBSETS):
            subset_globstr = os.path.join(basepath, subset, "*")
            all_files = glob.glob(subset_globstr)
            for filepath in all_files:
                X.append(readfile(filepath))
                y.append(subset)
        return X, y

    X, y = get_all_20ng("/home/user/aditya_ws/DeepUnkID/20ng")
    df = pd.DataFrame([X, y]).T

    df.columns = ['text', 'label']
    df = df.sample(frac=1).reset_index(drop=True)
    print(df)
    total = len(y)
    train, valid, test = int(0.6 * total), int(0.1 * total), int(total - 0.7 * total)
    partition_to_n_row = {"train" : train, "valid": valid, "test": test}
    return df, partition_to_n_row

    


def get_score(cm):
    fs = []
    n_class = cm.shape[0]
    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum()!=0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum()!=0 else 0
        f = 2*r*p/(r+p) if (r+p)!=0 else 0
        fs.append(f*100)

    f = np.mean(fs).round(2)
    f_seen = np.mean(fs[:-1]).round(2)
    f_unseen = round(fs[-1], 2)
    print("Overall(macro): ", f)
    print("Seen(macro): ", f_seen)
    print("=====> Uneen(Experiment) <=====: ", f_unseen)
    
    return f, f_seen, f_unseen

def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', figsize=(12,10),
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Compute confusion matrix
    np.set_printoptions(precision=2)
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('img/mat.png')
