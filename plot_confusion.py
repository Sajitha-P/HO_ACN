
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")


def plot_confusion_matrix(cm, classes,
                          normalize=False,title='',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontweight='bold',y=1.01,fontsize=12)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0,fontname = "Times New Roman",fontsize=12)
    plt.yticks(tick_marks, classes,fontname = "Times New Roman",fontsize=12)
    plt.ylabel('True Label',fontname = "Times New Roman",fontweight='bold',fontsize=12)
    plt.xlabel('Predicted Label',fontname = "Times New Roman",fontweight='bold',fontsize=12)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
               
      horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

