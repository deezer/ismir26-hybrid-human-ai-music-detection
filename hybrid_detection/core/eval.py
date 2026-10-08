
import pandas as pd
from sklearn.metrics import confusion_matrix


def tpr_fpr(y_true, y_score, thr=0.5, as_pandas=False):
    tn, fp, fn, tp = confusion_matrix(y_true, (y_score >= thr).astype(int)).ravel()
    tpr = tp / (tp + fn) if (tp + fn > 0) else 0
    fpr = fp / (tn + fp) if (tn + fp > 0) else 0

    if as_pandas:
        return pd.Series([tpr, fpr], index=['tpr', 'fpr'])
    else:
        return [tpr, fpr]
