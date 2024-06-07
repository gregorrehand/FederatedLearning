import os
import sys

import torch
from model import load_parameters
import torch.nn.functional as F

from data import load_data
from fedn.utils.helpers.helpers import save_metrics

import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def validate(in_model_path, out_json_path, data_path=None):
    """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    # Load data
    x_train, y_train = load_data(data_path)
    x_test, y_test = load_data(data_path, is_train=False)

    # Load model
    model = load_parameters(in_model_path)
    model.eval()

    # Evaluate
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        train_out = model(x_train)
        training_loss = criterion(train_out, y_train)
        training_accuracy = torch.sum(torch.argmax(train_out, dim=1) == y_train) / len(train_out)
        test_out = model(x_test)
        test_loss = criterion(test_out, y_test)
        test_accuracy = torch.sum(torch.argmax(test_out, dim=1) == y_test) / len(test_out)

        y_test_np = y_test.numpy()
        y_pred_np = torch.argmax(test_out, dim=1).numpy()
        y_score_np = F.softmax(test_out, dim=1).numpy()
        
        kappa = cohen_kappa(y_test_np, y_pred_np)
        f1 = f1_score(y_test_np, y_pred_np)
        roc = roc_auc_score(y_test_np, y_score_np)

    # JSON schema
    report = {
        "training_loss": training_loss.item(),
        "training_accuracy": training_accuracy.item(),
        "test_loss": test_loss.item(),
        "test_accuracy": test_accuracy.item(),
        "kappa": kappa.item(),
        "f1": f1.item(),
        "roc": roc.item(),

    }

    # Save JSON
    save_metrics(report, out_json_path)

def cohen_kappa(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Length of true and predicted labels must be the same."
    
    n = len(y_true)
    observed_agreement = np.sum(np.array(y_true) == np.array(y_pred)) / n
    
    labels = np.unique(np.concatenate((y_true, y_pred)))
    expected_agreement = 0.0
    
    for label in labels:
        p_y_true = np.sum(np.array(y_true) == label) / n
        p_y_pred = np.sum(np.array(y_pred) == label) / n
        expected_agreement += p_y_true * p_y_pred
    
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return kappa

def f1_score(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Length of true and predicted labels must be the same."
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = np.unique(y_true)

    f1_scores = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return np.mean(f1_scores)

def roc_auc_score(y_true, y_score):
    assert len(y_true) == len(y_score), "Length of true labels and scores must be the same."
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    roc_aucs = []
    for c in range(10):
        y_true_c = (y_true == c).astype(int)
        y_score_c = y_score[:, c]
        
        # Sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score_c, kind="mergesort")[::-1]
        y_score_c = y_score_c[desc_score_indices]
        y_true_c = y_true_c[desc_score_indices]
        
        # Compute true positive and false positive rates
        tps = np.cumsum(y_true_c)
        fps = np.cumsum(1 - y_true_c)
        
        tpr = tps / tps[-1] if tps[-1] != 0 else np.zeros_like(tps)
        fpr = fps / fps[-1] if fps[-1] != 0 else np.zeros_like(fps)
        
        # Append initial point (0,0)
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])
        
        # Calculate AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr)
        roc_aucs.append(auc)
    
    return np.mean(roc_aucs) 


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
