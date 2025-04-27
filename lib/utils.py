import numpy as np

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cal_acc(y_pred, y_true, threshold=0.7):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    max_probs = np.max(y_pred, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    correct = ((y_pred_labels == y_true) & (max_probs >= threshold)).sum()
    incorrect = (max_probs < threshold).sum()

    err_ind = np.where ((y_pred_labels != y_true))[0]#np.where((max_probs < threshold))[0]
    conf_acm = list()
    if len(err_ind) > 0:
        for ii in err_ind:
            conf_acm.append(max_probs[ii])

    total = correct + incorrect

    accuracy = correct / total
    return accuracy, {'err_ind': err_ind, 'conf_acm':conf_acm}

def cal_acc_org(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    correct = (y_pred == y_true).sum()
    total = len(y_true)
    accuracy = correct / total
    return accuracy

from pathlib import Path
import glob
import re


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path
