
import pylab as pl
from scikits.learn.metrics import auc

def plot_roc(fpr, tpr):
    """
    Plot the ROC curve.
       fpr, tpr, thresholds = roc_curve(y[half:], probas_[:,1])
        plot_roc(fpr, tpr)
    Code from http://scikit-learn.sourceforge.net/auto_examples/plot_roc.html
    """
    # Plot ROC curve
    pl.figure(-1)
    pl.clf()
    roc_auc = auc(fpr, tpr)
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0,1.0])
    pl.ylim([0.0,1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()

def plot_precision_recall(precision, recall):
    """
    Plot the ROC curve.
       precision, recall, thresholds = precision_recall_curve(y[half:], probas_[:,1])
       plot_precision_recall(precision, recall)
    Code from http://scikit-learn.sourceforge.net/auto_examples/plot_precision_recall.html
    """
    pl.figure(-1)
    pl.clf()
    area = auc(recall, precision)
    pl.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % area)
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0,1.05])
    pl.xlim([0.0,1.0])
    pl.title('Precision-Recall example: AUC=%0.2f' % area)
    pl.legend(loc="lower left")
    pl.show()
