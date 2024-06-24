import matplotlib.pyplot as plt
import numpy as np


def plot_classifier_score(
        pred_labels_train,
        truth_labels_train,
        pred_labels_test,
        truth_labels_test
):

    pred_labels_train_sig = pred_labels_train[truth_labels_train == 1]
    pred_labels_train_bkg = pred_labels_train[truth_labels_train == 0]

    pred_labels_test_sig = pred_labels_test[truth_labels_test == 1]
    pred_labels_test_bkg = pred_labels_test[truth_labels_test == 0]

    plt.hist(
        pred_labels_train_sig,
        color='r',
        alpha=0.5,
        range=(0, 1),
        bins=30,
        histtype='stepfilled',
        density=True,
        label='S (train)',
    )
    plt.hist(
        pred_labels_train_bkg,
        color='b',
        alpha=0.5,
        range=(0, 1),
        bins=30,
        histtype='stepfilled',
        density=True,
        label='B (train)',
    )

    hist, bins = np.histogram(
        pred_labels_test_sig, bins=30, range=(0, 1), density=True
    )
    scale = len(pred_labels_test_sig) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

    hist, bins = np.histogram(
        pred_labels_test_bkg, bins=30, range=(0, 1), density=True
    )
    scale = len(pred_labels_test_bkg) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.title('Nueral Network Score Distribution')
    plt.xlabel('NN Score')
    plt.ylabel('Normalized Events')
    plt.legend(loc='best')
    plt.show()
