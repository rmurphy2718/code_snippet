import numpy as np


def majority_vote_func(x):
    unique, counts = np.unique(ar=x, return_counts=True)
    return unique[np.argmax(counts)]


def majority_vote_accuracy(act_output, full_model_output):
    output = np.apply_along_axis(majority_vote_func, 1, np.round(full_model_output))
    correct = np.equal(act_output, output[:, None]).sum()
    return (1.0 * correct) / (len(act_output))


def rounded_accuracy(act_output, output):
    correct = np.equal(act_output, np.round(output[:, None])).sum()
    return (1.0 * correct) / (len(act_output))
