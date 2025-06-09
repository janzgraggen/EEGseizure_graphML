# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-05-13 -*-
# -*- Last revision: 2025-06-09 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Functions to plot-*-

# Import libraries
import numpy as np
import mlflow
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_cm_matrix(cm_array: np.ndarray, set: str, file_pth, epoch):
    """plot confusion matrix and log it as an image in mlflow

    Args:
        cm_array (np.ndarray): The confusion matrix to plot.
        set (str): The set name, e.g., 'train' or 'val'.
        file_pth (_type_): Path to save the confusion matrix image temporarily.
        epoch (_type_): The epoch number, used for naming the saved image.
    """
    filename_tmp = os.path.join(file_pth, "conf_matrix.jpg")
    filename_artifact = os.path.join("cfmx", "{}_cfmx_{}.jpg".format(set, epoch))

    cm_plot_train = ConfusionMatrixDisplay(
        confusion_matrix=cm_array,
        display_labels=[0, 1],
    )

    cm_plot_train.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(filename_tmp)
    plt.close()
    mlflow.log_image(Image.open(filename_tmp), artifact_file=filename_artifact)
