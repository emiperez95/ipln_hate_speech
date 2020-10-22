import numpy as np
import os
import pandas as pd
import random
import re
from sklearn import svm, metrics
import sys

TUNE_HYPERPARAMS = False

def read_examples(file_path, name, word_embeddings, drop_unknown_words=False):
    """Reads examples of `name` dataset and their classification.
    """

    print(f'Reading {name} set')

    tweets = pd.read_csv(file_path,"\t")
    X = np.array([take_average(tweet_embedding(tweet, word_embeddings, drop_unknown_words)) for tweet in tweets.text])
    y = tweets.odio.to_numpy()

    print(f'Done reading {name} set')

    return X, y


def tweet_embedding(tweet, embeddings, drop_unknown_words):
    """ Returns an np.array where each entry is a 300-dim vector with the
    embedding of a word. If the word doesn't have an embedding, it's ommitted.
    """

    tw_words = re.sub(r'(\W)', r' \1 ', tweet).split()

    if drop_unknown_words:
        tw_words = [word for word in tw_words if (word != '') and (word in embeddings.index)] # Discard empty strings and unknown words
    else:
        tw_words = [word for word in tw_words if (word != '')] # Discard empty strings

    tw_embedding = np.empty([len(tw_words), 300])

    for i, word in enumerate(tw_words):
        if word in embeddings.index:
            word_embedding = embeddings.loc[word]
        else:
            word_embedding = pd.Series(np.random.uniform(-1, 1, size=(300,)))
        tw_embedding[i] = word_embedding.to_numpy()

    return tw_embedding


def take_average(tweet_embedding):
    """ Takes the embedding of a tweet (the embedding of each word)
    and return s the average of each of the 300 dimensions
    """

    return np.average(tweet_embedding, axis=0)


def write_output_file(results, test_name):
    """ Outputs the results of `test_name` to a file of the same name
    but .out extension.
    """

    out_file = test_name.replace('.csv', '.out')
    print(f'Writing to {out_file}')
    with open(out_file, 'w+') as out_file:
        out_file.write("\n".join(list(map(str, results))))
    print('Done writing')


def tune_hyperparameters(X_val, y_val):
    """Returns the SVM hyperparams that best fit the given dataset
    """
    from sklearn.model_selection import GridSearchCV

    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3)
    grid.fit(X_val, y_val)

    print(grid.best_params_)
    print(grid.best_estimator_)

    return grid.best_params_


def main():
    data_path = sys.argv[1]
    test_files = sys.argv[2:]
    word_embeddings = pd.read_csv(
        data_path + 'fasttext.es.300.txt',
        delim_whitespace=True,
        header=None,
        quoting=3,
        index_col=0,
        keep_default_na=False
    )

    hyperparams = {'C': 100,
                  'gamma': 0.1,
                  'kernel': 'rbf'}

    if TUNE_HYPERPARAMS:
        X_val, y_val = read_examples(os.path.join(data_path, 'val.csv'), 'validation', word_embeddings, drop_unknown_words=True)
        hyperparams = tune_hyperparameters(X_val, y_val)

    X_train, y_train = read_examples(os.path.join(data_path, 'train.csv'), 'training', word_embeddings, drop_unknown_words=True)

    # Create a svm Classifier
    clf = svm.SVC(
        C=hyperparams['C'],
        gamma=hyperparams['gamma'],
        kernel=hyperparams['kernel']
    )

    # Train the model using the training set
    print("Training SVM model..")
    X_train = np.nan_to_num(X_train)
    clf.fit(X_train, y_train)

    print("Done training.")

    for test_path in test_files:
        X_test, y_test = read_examples(test_path, test_path, word_embeddings)

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)
        print(y_pred)

        # Write results into a file
        write_output_file(y_pred, test_path)

        # Model Accuracy: how often is the classifier correct?
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

        # Model Precision: how many selected items are relevant?
        print("Precision:", metrics.precision_score(y_test, y_pred))

        # Model Recall: how many relevant items are selected?
        print("Recall:", metrics.recall_score(y_test, y_pred))

        # F1 score: weighted average of the precision and recall
        print("F1:", metrics.f1_score(y_test, y_pred))

if __name__ == "__main__":
    main()
