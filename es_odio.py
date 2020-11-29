import os
import sys

import numpy as np
from sklearn import svm, metrics
import pandas as pd

import utils

def data_vector_to_mean(x):
    # count_vector = np.count_nonzero(np.any(x!=np.zeros(x.shape[2]), axis=2), axis=1)
    sum_data = np.sum(x, axis=1)
    # ret_data = np.transpose(np.divide( np.transpose(sum_data), count_vector ))
    return sum_data

def svm_mean_run(Xtr, Xte, Ytr, Yte,  hyperparams):
    # Process data
    X_train = data_vector_to_mean(Xtr)
    X_test = data_vector_to_mean(Xte)

    # SVM setup
    # hyperparams = {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'}
    svm_clf = svm.SVC( C=hyperparams['C'], gamma=hyperparams['gamma'], kernel=hyperparams['kernel'])

    # Train svm
    print(X_train.shape)
    print(X_test.shape)
    svm_clf.fit(X_train, Ytr)
    y_pred = svm_clf.predict(X_test)

    # Append to statistics
    print_stat(Yte, y_pred)

    return svm_clf

def print_stat(x, y):
    print("Accuracy {}".format(metrics.accuracy_score(x, y)))
    print("Precision {}".format(metrics.precision_score(x, y)))
    print("Recall {}".format(metrics.recall_score(x, y)))
    print("F1 {}".format(metrics.f1_score(x, y)))


if __name__ == "__main__":
    data = utils.load_and_join_datasets("data")
    procesing_pipe = [
        utils.remove_urls,
        utils.new_line_to_space,
        utils.remove_special_chars,
        # utils.remove_non_ascii_chars,
        utils.strip_spaces,

        utils.tokenize_split,
        # utils.tokenize_nltk,

        utils.remove_small_words,
        utils.remove_stopwords,

        # utils.stemming_Porter,
        # utils.stemming_Snowball,
        # utils.spacy_lemma,
    ]
    print("Aplicando preprocesamiento a datos de entrenamiento")
    utils.data_apply(data, procesing_pipe)

    # === Split data ===
    X_train, y_train, X_test, y_test = utils.split_datasets(data)

    embedding = utils.file_embedding()

    # === Using embeddings ===
    X_tr = utils.intersect_embedding_data(embedding, X_train)
    X_te = utils.intersect_embedding_data(embedding, X_test)

    # == SVM ==
    hyperparams = {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'}
    svm = svm_mean_run(X_tr, X_te, y_train, y_test,  hyperparams)

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            full_path = os.path.abspath(arg)
            path, fl_name = os.path.split(full_path)

            new_data = utils.simple_data_load(full_path)
            utils.data_apply(new_data, procesing_pipe)
            X = utils.intersect_embedding_data(embedding, new_data)
            X = data_vector_to_mean(X)
            print(X.shape)
            y = svm.predict(X)

            target_filename = "{}/{}.out".format(path,fl_name[:-4])
            print(target_filename)
            with open(target_filename, 'w') as fl:
                for y_pred in y:
                    fl.write(str(y_pred)+'\n')


