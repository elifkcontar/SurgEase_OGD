import os
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import ranksums
from math import sqrt
import sklearn
import sklearn.metrics

def true_predictions(output, target):
    correct = 0
    for i in range(len(output)):
        prediction = output[i].argmax(dim=1, keepdim=True)
        correct += prediction.eq(target[i].view_as(prediction)).sum().item()

    return correct

def true_predictions_binary(output, target):
    prediction = output.argmax(dim=1, keepdim=True)
    correct = prediction.eq(target.view_as(prediction)).sum().item()

    return correct

def true_predictions_class(output, target, class_num):
    correct_list = []   #append TP and total number for each class. [TP_0, Total_0, TP_1, Total_1, ..., TP_n, Total_n]
    prediction = output.argmax(dim=1, keepdim=True)
    
    for i in range (class_num):
        target_boolean = target == i

        tmp_output =  prediction[target_boolean]
        tmp_target = target[target_boolean]

        correct = tmp_output.eq(tmp_target.view_as(tmp_output)).sum().item() 
        correct_list.append(correct)
        correct_list.append(len(tmp_target))

    return correct_list

def true_predictions_remission(output, target):
    target_sum = target.sum(dim=0)
    output_reduced = torch.zeros_like(target)

    for i in range(len(output)):
        output_reduced[i] = output[i].argmax(1)
    output_sum = output_reduced.sum(dim=0)

    output_sum_binary = (output_sum > 0).type(torch.uint8)
    target_sum_binary = (target_sum > 0).type(torch.uint8)
    
    acc = output_sum_binary.eq(target_sum_binary.view_as(output_sum_binary)).sum().item()
    return acc

def true_predictions_remission_class(output, target):
    ret_list = []
    target_sum = target.sum(dim=0)
    output_reduced = torch.zeros_like(target)

    for i in range(len(output)):
        output_reduced[i] = output[i].argmax(1)
    output_sum = output_reduced.sum(dim=0)

    output_sum_binary = (output_sum > 0).type(torch.uint8)
    target_sum_binary = (target_sum > 0).type(torch.uint8)
    for i in range(2):
        if(i==1):
            target_boolean = target_sum > 0
        elif(i==0):
            target_boolean = target_sum == 0
        tmp_output =  output_sum_binary[target_boolean]
        tmp_target = target_sum_binary[target_boolean]

        correct = tmp_output.eq(tmp_target.view_as(tmp_output)).sum().item()

        ret_list.append(correct)
        ret_list.append(len(tmp_target))
    
    return ret_list

def true_predictions_UCEIS(output, target):
    target_sum = target.sum(dim=0)
    output_reduced = torch.zeros_like(target)

    for i in range(len(output)):
        output_reduced[i] = output[i].argmax(1)
    output_sum = output_reduced.sum(dim=0)

    return output_sum.eq(target_sum.view_as(output_sum)).sum().item()

def convert_confusion_matrix_form(output, target):
    target_sum = target.sum(dim=0)
    output_reduced = torch.zeros_like(target)

    for i in range(len(output)):
        output_reduced[i] = output[i].argmax(1)
    output_sum = output_reduced.sum(dim=0)

    return output_sum, target_sum

def convert_pred_binary(output, target):
    output_sum = torch.zeros_like(target)

    for i in range(len(output)):
        output_sum[i] = output[i].argmax(1)

    return output_sum, target

def confusion_matrix_severity(output, target, level=0):
    target_sum = target.sum(dim=0)
    output_reduced = torch.zeros_like(target)

    for i in range(len(output)):
        output_reduced[i] = output[i].argmax(1)
    output_sum = output_reduced.sum(dim=0)

    output_sum_binary = (output_sum > level).type(torch.uint8)
    target_sum_binary = (target_sum > level).type(torch.uint8)

    [tn, fp, fn, tp] = sklearn.metrics.confusion_matrix(target_sum_binary.cpu(), output_sum_binary.cpu(), labels=[0, 1]).ravel()
    #prec = tp/(tp+fp)
    #recc = tp/(tp+fn)
    #spec = tn/(tn+fp)
    #f1 = (2*prec*recc)/(prec+recc)

    #return prec, recc, f1, spec
    return tn, fp, fn, tp

def true_predictions_mixup(output, targets_a, targets_b, lam):
    correct = 0
    for i in range(len(output)):
        prediction = output[i].argmax(dim=1, keepdim=True)
        correct += lam * prediction.eq(targets_a[i].view_as(prediction)).sum().item() + (1 - lam) * prediction.eq(
                targets_b[i].view_as(prediction)).sum().item()

    return correct


def true_predictions_UCEIS_mixup(output, targets_a, targets_b, lam):
    target_sum_a = targets_a.sum(dim=0)
    target_sum_b = targets_b.sum(dim=0)
    output_reduced = torch.zeros_like(targets_a)

    for i in range(len(output)):
        output_reduced[i] = output[i].argmax(1)
    output_sum = output_reduced.sum(dim=0)

    final_sum = lam * output_sum.eq(target_sum_a.view_as(output_sum)).sum().item() + (1 - lam) * output_sum.eq(
            target_sum_b.view_as(output_sum)).sum().item()

    return final_sum

def plot_confusion_matrix_2(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def get_frame_annotation(video_name, frame_id, annotations_root_path):
    vascular_pattern_scores = ["87000c19-48ca-426e-b50a-c5dc4a7b8253", "610fe22a-2385-43eb-b43e-bfccc8670ba0", "d9aa49f5-babe-4bac-bd7b-691c92588fd0"]                                        
    bleeding_score = ["c0f5dd0d-688b-451f-9c08-378c2d8a5bb8", "b97e8465-70ee-422e-9d99-57f6d8aaf156", "3a0a3cec-a444-46bd-9b0e-8f6d3f09d7cd", "cc34b5ad-8d3b-4fc3-ae5c-7662988de4b5"]                     
    erosions_score = ["50ff75e0-2973-4f21-8c74-eba731250ae3", "ff155879-b965-466b-b457-d22fd9efce14", "ed3c08d1-dbc7-4a84-9e28-dd3a997c1c8c", "67648647-2cc6-407b-8b56-5af866d17788"]
    
    annotation_path = os.path.join(annotations_root_path, video_name, "classifications.json")
    with open(annotation_path) as f:
        class_annotations = json.load(f)
    
    video_score = int(video_name[-5])
    # If overall video has score of 0
    if video_score == 0:
        return [0, 0, 0]
    else:
        sub_scores = [-1, -1, -1]
        frame_class_annotations = class_annotations["frame_labels"].get(str(frame_id), -1)
        
        if frame_class_annotations != -1:
            for annotation in frame_class_annotations:
                # Vascular pattern score
                if annotation[0]["featureHash"] == "0480a66e-9b49-4a8e-94db-4288463d5f74":
                    sub_scores[0] = vascular_pattern_scores.index(annotation[0]["answers"][0]["featureHash"])                                          
                # Bleeding
                elif annotation[0]["featureHash"] == "7938122d-8f6e-44ed-b4aa-1ca3de9adf30":
                    sub_scores[1] = bleeding_score.index(annotation[0]["answers"][0]["featureHash"])                                            
                # Erosion
                elif annotation[0]["featureHash"] == "94691065-ee12-4150-9ad5-3688ef4ecab3":
                    sub_scores[2] = erosions_score.index(annotation[0]["answers"][0]["featureHash"])
                                            
            total_missing = sub_scores.count(-1)
            if total_missing == 1:
                missing_index = sub_scores.index(-1)
                sub_scores[missing_index] = video_score - (sum(sub_scores) + 1)
            
            return sub_scores
        else:
            return sub_scores
    
def get_n_level_bucketing_accuracy(y_true, y_pred, n=1):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    inside_bucket = np.sum(abs(y_true_np - y_pred_np) <= n)

    return inside_bucket / y_true_np.size


def get_all_level_bucketing_accuracy(y_true, y_pred):
    level_accuracies = []
    for i in range(9):
        level_accuracies.append(get_n_level_bucketing_accuracy(y_true, y_pred, i))

    return level_accuracies


def get_stratified_UCEIS_accuracies(y_true, y_pred):
    """
    remission: (0-1)
    mild: (2-4)
    moderate: (5-6)
    severe: (7-8)

    :param y_true: true individual scores
    :param y_pred: predicted individual scores
    :return: stratified accuracy
    """
    stratification_map = {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
        4: 1,
        5: 2,
        6: 2,
        7: 3,
        8: 3
    }

    y_true_stratified = []
    y_pred_stratified = []

    for i in range(len(y_true)):
        y_true_stratified.append(stratification_map[y_true[i]])
        y_pred_stratified.append(stratification_map[y_pred[i]])

    return y_true_stratified, y_pred_stratified

def mixup_data(x, y, device, alpha=1.0, ):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a = y
    y_b = y[:,index]
    return mixed_x, y_a, y_b, lam

def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def convert_multiclass_to_binary_class(multiclass, zero_classes=[0]):
    binary = []
    for item in multiclass:
        if item in zero_classes:
            binary.append(0)
        else:
            binary.append(1)
    return binary

def get_dataset_mean_and_std(torch_dataset):
    R_total = 0
    G_total = 0
    B_total = 0

    total_count = 0
    for image, _ in torch_dataset:
        image = np.asarray(image)
        total_count = total_count + image.shape[0] * image.shape[1]

        R_total = R_total + np.sum(image[:, :, 0])
        G_total = G_total + np.sum(image[:, :, 1])
        B_total = B_total + np.sum(image[:, :, 2])

    R_mean = R_total / total_count
    G_mean = G_total / total_count
    B_mean = B_total / total_count

    R_total = 0
    G_total = 0
    B_total = 0

    total_count = 0
    for image, _ in torch_dataset:
        image = np.asarray(image)
        total_count = total_count + image.shape[0] * image.shape[1]

        R_total = R_total + np.sum((image[:, :, 0] - R_mean) ** 2)
        G_total = G_total + np.sum((image[:, :, 1] - G_mean) ** 2)
        B_total = B_total + np.sum((image[:, :, 2] - B_mean) ** 2)

    R_std = sqrt(R_total / total_count)
    G_std = sqrt(G_total / total_count)
    B_std = sqrt(B_total / total_count)

    return [R_mean / 255, G_mean / 255, B_mean / 255], [R_std / 255, G_std / 255, B_std / 255]