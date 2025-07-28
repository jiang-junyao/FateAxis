# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 17:37:11 2025

@author: peiweike
"""
from FateAxis.tool import dl_trainer
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import FateAxis.model.clf as clf
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import FateAxis.tool.extractor as ext
from torch.utils.data import Dataset, DataLoader
epsilon = 1e-6


def get_model_weights(fsn):

    model_losses = torch.tensor([fsn.model_loss[model_name] for model_name in fsn.model_loss.keys()])
    weights = 1 / (model_losses + epsilon)
    normalized_weights = weights / weights.sum() 

    return normalized_weights.numpy()



def get_model_predictions(fsn,pred_data,batch_size=10,model_acc_cutoff=0.95):
    
    test_loader = DataLoader(pred_data, batch_size=batch_size, 
                                     shuffle=False)
    model_probabilities = []
    model_labels = []

    for model_name in fsn.saved_models.keys():
        if fsn.model_acc[model_name] > model_acc_cutoff:
            model = fsn.saved_models[model_name]
            ## for torch model
            if model_name.startswith('pytorch'):  # 如果是 PyTorch 模型
                Trainer = dl_trainer.torch_trainer(model,
                                                   device=fsn.device,
                                                   just_train=False)
                Trainer.load_model(model)
                labels, probabilities = Trainer.predict(test_loader)
                model_labels.append(labels)
                model_probabilities.append(probabilities)
                
            ## for sklearn model
            else:  
    
                probabilities = model.predict_proba(pred_data)
                labels = np.argmax(probabilities, axis=1)
                model_labels.append(labels)
                model_probabilities.append(probabilities)
    
    if len(model_labels)>0:
        return model_labels, model_probabilities
    
    else:
        raise ValueError("no model pass the cutoff, please re-start the acc_cut")


def generate_fuzzy_labels(probabilities, method='fixed', threshold=0.1, 
                          alpha=0.5, relative_threshold=0.8, beta=0.5):
    """
    Generalized fuzzy label generation function, supporting multiple methods.
    
    :param probabilities: Prediction probabilities for each sample, shape = (num_samples, num_classes)
    :param method: Method for generating fuzzy labels, supports 'fixed', 'std', 'relative', 'entropy'
    :param threshold: Fixed threshold, applicable to the 'fixed' method

    :param alpha: Control parameter for standard deviation adjustment, applicable to the 'std' method

    :param relative_threshold: Relative proportion threshold, applicable to the 'relative' method

    :param beta: Entropy control parameter, applicable to the 'entropy' method

    :return: List of fuzzy labels

    """
    fuzzy_labels = []
    
    for prob in probabilities:
        # Sort labels and their corresponding probabilities

        sorted_indices = np.argsort(prob)[::-1]  # Sort label indices in descending order of probabilities

        sorted_prob = prob[sorted_indices]      # Corresponding probability values

        # Initialize the fuzzy label set

        label_set = [str(sorted_indices[0])]
        
        # Calculate the threshold based on the specified method

        if method == 'fixed':
            # Fixed threshold

            threshold_value = threshold

        elif method == 'std':
            # Dynamically adjust based on standard deviation

            threshold_value = alpha * np.std(prob)
        elif method == 'relative':
            # Based on proportional difference

            max_prob = sorted_prob[0]
            threshold_value = max_prob * relative_threshold

        elif method == 'entropy':
            # Dynamically adjust based on entropy

            entropy = -np.sum(prob * np.log(prob + 1e-12))  # Avoid log(0)
            threshold_value = beta * entropy

        else:
            raise ValueError(f"Unsupported method: {method}. Use 'fixed', 'std', 'relative', or 'entropy'.")
        
        # Determine the fuzzy labels

        for i in range(1, len(sorted_prob)):
            if method == 'relative':
                # For proportional difference, directly compare probability values

                if sorted_prob[i] >= threshold_value:
                    label_set.append(str(sorted_indices[i]))
                else:
                    break

            else:
                # For other methods, compare probability differences

                if sorted_prob[0] - sorted_prob[i] <= threshold_value:
                    label_set.append(str(sorted_indices[i]))
                else:
                    break

        # Combine fuzzy labels into a single string, e.g., "A+B" or "A+B+C"
        fuzzy_label = "+".join(label_set)
        fuzzy_labels.append(fuzzy_label)
    return fuzzy_labels



def normalize_label(label):

    parts = label.split('+')
    parts.sort()
    return '+'.join(parts)

        
def ensemble_predict_with_fuzzy_labels(fsn, test_loader, method='fixed', 
                                       fixed_threshold=0.1, 
                          alpha=0.5, relative_threshold=0.8, 
                          beta=0.5,model_acc_cutoff = 0.95):

    weights = get_model_weights(fsn)
    _, model_probabilities = get_model_predictions(fsn, test_loader,
                                                   model_acc_cutoff=model_acc_cutoff)
    weighted_probabilities = sum(
        weight * prob for weight, prob in zip(weights, model_probabilities)
    )
    
    ### generate fuzzy labels
    fuzzy_labels_fixed = generate_fuzzy_labels(weighted_probabilities, 
                                               method=method, 
                                               threshold=fixed_threshold,
                                               alpha=alpha,
                                               relative_threshold=relative_threshold,
                                               beta=beta)
    normalized_labels = [normalize_label(lbl) for lbl in fuzzy_labels_fixed]
    
    return weighted_probabilities,normalized_labels

