import os
from datetime import datetime
import time
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import pandas as pd
from utils.provider import true_predictions_remission
#from pytorch_model_summary import summary

from datasets.UCEIS_dataset import UCEIS_csv
from models import resnet, densenet, convnext
from utils.losses import multitask_CE, multitask_WCE, multitask_focal
from utils.provider import true_predictions, true_predictions_binary, true_predictions_UCEIS, fix_seed, get_dataset_mean_and_std
from utils.provider import true_predictions_remission_class, true_predictions_class, confusion_matrix_severity,convert_confusion_matrix_form, convert_pred_binary
import wandb
from random import shuffle
from math import ceil
from sklearn.utils import class_weight


def train(model, device, train_loader, criterion, optimizer):
    print('starting train')
    epsilon=1e-10
    model.train()
    training_loss = 0.0
    training_loss_0 = 0.0
    training_loss_1 = 0.0
    training_loss_2 = 0.
    correct_subscores_0 = 0
    correct_subscores_1 = 0
    correct_subscores_2 = 0
    correct_subscores = 0
    correct_UCEIS = 0
    correct_remission = 0
    remission_tuple = [0,0,0,0]
    v_tuple = [0,0,0,0,0,0]
    b_tuple = [0,0,0,0,0,0,0,0]
    e_tuple = [0,0,0,0,0,0,0,0]
    
    print('Train Lengths:')
    print(len(train_loader))
    print(len(train_loader.dataset))

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        target.transpose_(0, 1)
        
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()
        correct_subscores += true_predictions(output, target)
        correct_UCEIS += true_predictions_UCEIS(output, target)

        correct_subscores_0 += true_predictions_binary(output[0], target[0])
        correct_subscores_1 += true_predictions_binary(output[1], target[1])
        correct_subscores_2 += true_predictions_binary(output[2], target[2])
        
        tmp=true_predictions_class(output[0], target[0], 3) #vasc_0_tp,vasc_0_total,vasc_1_tp,vasc_1_total,vasc_2_tp,vasc_2_total
        v_tuple = [a + b for a, b in zip(tmp, v_tuple)]

        tmp=true_predictions_class(output[1], target[1], 4) #bleed_0_tp, bleed_1_tp, bleed_2_tp, bleed_3_tp, bleed_0_total, bleed_1_total, bleed_2_total, bleed_3_total
        b_tuple = [a + b for a, b in zip(tmp, b_tuple)]

        tmp=true_predictions_class(output[2], target[2], 4) #eros_0_tp, eros_1_tp, eros_2_tp, eros_3_tp, eros_0_total, eros_1_total, eros_2_total, eros_3_total
        e_tuple = [a + b for a, b in zip(tmp, e_tuple)]

        correct_remission += true_predictions_remission(output, target)            
        tmp = true_predictions_remission_class(output, target)
        remission_tuple = [a + b for a, b in zip(tmp, remission_tuple)]


        training_loss += loss[0].item()
        training_loss_0 += loss[1].item()
        training_loss_1 += loss[2].item()
        training_loss_2 += loss[3].item()

    training_loss /= len(train_loader)
    training_loss_0 /= len(train_loader)
    training_loss_1 /= len(train_loader)
    training_loss_2 /= len(train_loader)
    vasc_0 = v_tuple[0] / (v_tuple[1]+epsilon)
    vasc_1 = v_tuple[2] / (v_tuple[3]+epsilon)
    vasc_2 = v_tuple[4] / (v_tuple[5]+epsilon)
    bleed_0 = b_tuple[0] / (b_tuple[1]+epsilon)
    bleed_1 = b_tuple[2] / (b_tuple[3]+epsilon)
    bleed_2 = b_tuple[4] / (b_tuple[5]+epsilon)
    bleed_3 = b_tuple[6] / (b_tuple[7]+epsilon)
    eros_0 = e_tuple[0] / (e_tuple[1]+epsilon)
    eros_1 = e_tuple[2] / (e_tuple[3]+epsilon)
    eros_2 = e_tuple[4] / (e_tuple[5]+epsilon)
    eros_3 = e_tuple[6] / (e_tuple[7]+epsilon)
    correct_remission /=len(train_loader.dataset)
    correct_remission_0 = remission_tuple[0] / remission_tuple[1]
    correct_remission_1 = remission_tuple[2] / remission_tuple[3]
    correct_subscores /= (len(train_loader.dataset) * 3)
    correct_UCEIS /= len(train_loader.dataset)

    return (training_loss, training_loss_0, training_loss_1, training_loss_2, correct_subscores, correct_UCEIS, correct_remission,correct_remission_0,correct_remission_1,
                vasc_0, vasc_1, vasc_2, bleed_0, bleed_1, bleed_2, bleed_3, eros_0, eros_1, eros_2, eros_3)

def validation(model, device, val_loader, criterion):
    model.eval()
    epsilon=1e-10
    val_loss = 0.0
    val_loss_0 = 0.0
    val_loss_1 = 0.0
    val_loss_2 = 0.0
    correct_subscores_0 = 0
    correct_subscores_1 = 0
    correct_subscores_2 = 0
    correct_subscores = 0
    correct_UCEIS = 0
    correct_remission = 0
    remission_tuple = [0,0,0,0]
    v_tuple = [0,0,0,0,0,0]
    b_tuple = [0,0,0,0,0,0,0,0]
    e_tuple = [0,0,0,0,0,0,0,0]
    
    print('Validation Lengths:')
    print(len(val_loader))
    print(len(val_loader.dataset))

    with torch.no_grad():
        for data, target, in val_loader:
            data, target = data.to(device), target.to(device)
            target.transpose_(0, 1)

            output = model(data)
            loss = criterion(output, target)
            correct_subscores += true_predictions(output, target)
            correct_UCEIS += true_predictions_UCEIS(output, target)

            correct_subscores_0 += true_predictions_binary(output[0], target[0])
            correct_subscores_1 += true_predictions_binary(output[1], target[1])
            correct_subscores_2 += true_predictions_binary(output[2], target[2])
        
            tmp=true_predictions_class(output[0], target[0], 3) #vasc_0_tp,vasc_0_total,vasc_1_tp,vasc_1_total,vasc_2_tp,vasc_2_total
            v_tuple = [a + b for a, b in zip(tmp, v_tuple)]

            tmp=true_predictions_class(output[1], target[1], 4) #bleed_0_tp, bleed_1_tp, bleed_2_tp, bleed_3_tp, bleed_0_total, bleed_1_total, bleed_2_total, bleed_3_total
            b_tuple = [a + b for a, b in zip(tmp, b_tuple)]

            tmp=true_predictions_class(output[2], target[2], 4) #eros_0_tp, eros_1_tp, eros_2_tp, eros_3_tp, eros_0_total, eros_1_total, eros_2_total, eros_3_total
            e_tuple = [a + b for a, b in zip(tmp, e_tuple)]

            correct_remission += true_predictions_remission(output, target)
            tmp = true_predictions_remission_class(output, target)
            remission_tuple = [a + b for a, b in zip(tmp, remission_tuple)]

            val_loss += loss[0].item()
            val_loss_0 += loss[1].item()
            val_loss_1 += loss[2].item()
            val_loss_2 += loss[3].item()

        val_loss /= len(val_loader)
        val_loss_0 /= len(val_loader)
        val_loss_1 /= len(val_loader)
        val_loss_2 /= len(val_loader)
        correct_subscores_0 /=len(val_loader.dataset)
        correct_subscores_1 /=len(val_loader.dataset)
        correct_subscores_2 /=len(val_loader.dataset)
        vasc_0 = v_tuple[0] / (v_tuple[1]+epsilon)
        vasc_1 = v_tuple[2] / (v_tuple[3]+epsilon)
        vasc_2 = v_tuple[4] / (v_tuple[5]+epsilon)
        bleed_0 = b_tuple[0] / (b_tuple[1]+epsilon)
        bleed_1 = b_tuple[2] / (b_tuple[3]+epsilon)
        bleed_2 = b_tuple[4] / (b_tuple[5]+epsilon)
        bleed_3 = b_tuple[6] / (b_tuple[7]+epsilon)
        eros_0 = e_tuple[0] / (e_tuple[1]+epsilon)
        eros_1 = e_tuple[2] / (e_tuple[3]+epsilon)
        eros_2 = e_tuple[4] / (e_tuple[5]+epsilon)
        eros_3 = e_tuple[6] / (e_tuple[7]+epsilon)
        correct_remission /=len(val_loader.dataset)
        correct_remission_0 = remission_tuple[0] / remission_tuple[1]
        correct_remission_1 = remission_tuple[2] / remission_tuple[3]
        correct_subscores /= (len(val_loader.dataset) * 3)
        correct_UCEIS /= len(val_loader.dataset)

        #return (val_loss, val_loss_0, val_loss_1, val_loss_2, correct_subscores, correct_UCEIS)
        return (val_loss, val_loss_0, val_loss_1, val_loss_2, correct_subscores_0, correct_subscores_1, correct_subscores_2, correct_subscores, correct_UCEIS, correct_remission,correct_remission_0,correct_remission_1,
                vasc_0, vasc_1, vasc_2, bleed_0, bleed_1, bleed_2, bleed_3, eros_0, eros_1, eros_2, eros_3)

def test(model, device, loader):
    tn =0
    fp = 0
    fn = 0
    tp = 0
    tn_severity = 0
    fp_severity = 0
    fn_severity = 0
    tp_severity = 0
    output_list=[]
    target_list=[]
    output_list_=[[],[],[]]
    target_list_=[[],[],[]]

    with torch.no_grad():
        for data, target, in loader:
            data, target = data.to(device), target.to(device)
            target.transpose_(0, 1)

            output = model(data)
            o, t = convert_confusion_matrix_form(output, target)
            output_list.extend(o.cpu().detach().numpy())
            target_list.extend(t.cpu().detach().numpy())
            o, t = convert_pred_binary(output, target)
            output_list_[0].extend(o.cpu().detach().numpy()[0])
            target_list_[0].extend(t.cpu().detach().numpy()[0])
            output_list_[1].extend(o.cpu().detach().numpy()[1])
            target_list_[1].extend(t.cpu().detach().numpy()[1])
            output_list_[2].extend(o.cpu().detach().numpy()[2])
            target_list_[2].extend(t.cpu().detach().numpy()[2])

            scrore_tuple= confusion_matrix_severity(output, target, 0)
            tn += scrore_tuple[0]
            fp += scrore_tuple[1]
            fn += scrore_tuple[2]
            tp += scrore_tuple[3]

            
            scrore_tuple = confusion_matrix_severity(output, target, 3)
            tn_severity += scrore_tuple[0]
            fp_severity += scrore_tuple[1]
            fn_severity += scrore_tuple[2]
            tp_severity += scrore_tuple[3]
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    specificity = tn/(tn+fp)
    f1 = (2*precision*recall)/(precision+recall)

    precision_severity = tp_severity/(tp_severity+fp_severity)
    recall_severity = tp_severity/(tp_severity+fn_severity)
    specificity_severity = tn_severity/(tn_severity+fp_severity)
    f1_severity = (2*precision_severity*recall_severity)/(precision_severity+recall_severity)

    wandb.log({"conf_mat_UCEIS" : wandb.plot.confusion_matrix(probs=None,
            y_true=target_list, preds=output_list,
            class_names=['0','1','2','3','4','5','6','7','8'],title="conf_mat_UCEIS")})
    wandb.log({"conf_mat_Vascularity" : wandb.plot.confusion_matrix(probs=None,
            y_true=target_list_[0], preds=output_list_[0],
            class_names=['0','1','2'],title="conf_mat_Vascularity")})
    wandb.log({"conf_mat_Bleeding" : wandb.plot.confusion_matrix(probs=None,
            y_true=target_list_[1], preds=output_list_[1],
            class_names=['0','1','2','3'],title="conf_mat_Bleeding")})
    wandb.log({"conf_mat_Erosions" : wandb.plot.confusion_matrix(probs=None,
            y_true=target_list_[2], preds=output_list_[2],
            class_names=['0','1','2','3'],title="conf_mat_Erosions")})

    return precision, recall,f1, specificity, precision_severity, recall_severity, f1_severity, specificity_severity 

def get_model(model_name):
    if model_name == "ResNet18":
        model = resnet.resnet18(pretrained=True, num_classes=[3, 4, 4])
    elif model_name == "ResNet34":
        model = resnet.resnet34(pretrained=True, num_classes=[3, 4, 4])
    elif model_name == "ResNet50":
        model = resnet.resnet50(pretrained=True, num_classes=[3, 4, 4])
    elif model_name == "DenseNet201":
        model = densenet.densenet201(pretrained=True, num_classes=[3, 4, 4])
    elif model_name == "DenseNet121":
        model = densenet.densenet121(pretrained=True, num_classes=[3, 4, 4])
    elif model_name == "Convnext_tiny":
        model = convnext.convnext_tiny(pretrained=True, num_classes=[3, 4, 4])
    elif model_name == "vit":
        model = convnext.vit(pretrained=True, num_classes=[3, 4, 4])
    elif model_name == "effnetb0":
        model = convnext.effnet_b0(pretrained=True, num_classes=[3, 4, 4])
    return model

def split(save=False, dir=''):
    frame_root = r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\Datasets\IBD_cropped2'
    all_folders = [f for f in os.listdir(frame_root) if not f.startswith('.')]
    shuffle(all_folders)
    train_set_ratio = 0.7
    val_set_ratio = 0.15
    test_set_ratio = 0.15
    val_set_folder_size = ceil(val_set_ratio * len(all_folders))
    test_set_folder_size = ceil(test_set_ratio * len(all_folders))

    val_folders = all_folders[0:val_set_folder_size]
    test_folders = all_folders[val_set_folder_size:(val_set_folder_size + test_set_folder_size)]
    train_folders = all_folders[(val_set_folder_size + test_set_folder_size):]
    if(save):
        with open(dir+'/train_folders.txt', "w") as f:
            for item in train_folders:f.write("%s\n" % item)

        with open(dir+'/val_folders.txt', "w") as f:
            for item in val_folders:f.write("%s\n" % item)

        with open(dir+'/test_folders.txt', "w") as f:
            for item in test_folders:f.write("%s\n" % item)

    return train_folders, val_folders, test_folders

def get_weights(train_folders, csv_filename):
    weights=[]
    df = pd.read_csv(csv_filename)
    train_df = df.loc[df['video_name'].isin(train_folders)]
    vascular = train_df['vascular_pattern_scores']
    bleeding = train_df['bleeding_score']
    erosions = train_df['erosions_score']
    #Check weights
    checkpoint = check_weights(vascular.tolist(), bleeding.tolist(), erosions.tolist())
    if not(checkpoint):
        return False
    weights.append(class_weight.compute_class_weight(class_weight='balanced',classes=[0, 1, 2],y=vascular).astype(float))
    weights.append(class_weight.compute_class_weight(class_weight='balanced',classes=[0, 1, 2, 3],y=bleeding).astype(float))
    weights.append(class_weight.compute_class_weight(class_weight='balanced',classes=[0, 1, 2, 3],y=erosions).astype(float))

    return weights

def check_weights(vascular, bleeding, erosions):
    uceis = [sum(x) for x in zip(vascular, bleeding, erosions)]
    for i in range(3):
        if not(vascular.count(i)>5):
           return False    
    for i in range(4):
        if not(bleeding.count(i)>5):
            return False
        if not(erosions.count(i)>5):
            return False
    for i in range(9):
        if not(uceis.count(i)>5):
            return False
    return True

def check_splits(train_folders, val_folders, test_folders, csv_filename):
    if not(get_weights(train_folders, csv_filename)):
        return False
    if not(get_weights(val_folders, csv_filename)):
        return False
    if not(get_weights(test_folders, csv_filename)):
        return False
    return True

if __name__ == '__main__':

    dataset_list = ['label_all_cropped.csv']
    loss_list = ['WCE']
    #model_list = ['effnetb0', 'ResNet34']
    model_list = ['ResNet34']
    seed_list=[1]

    for dataset_name in dataset_list:
        for model_name in model_list:
            for loss_name in loss_list:

                for i in range(1):
                    fix_seed(seed_list[i])
                    learning_rate = 0.0001
                    weight_decay = 0.001
                    num_epoch = 50
                    best_acc = 0
                    best_threshold = 0.0001
                    if(model_name=='effnetb0'):
                        batch_size = 4
                    else:
                        batch_size = 8
                    num_worker = 1
                    early_stop_counter = 0
                    scheduler_patience = 5
                    early_stopping_thresh = 10
                    dropout_rate = 0.3
                    enable_wandb = True
                    optimizer_name = "Adam"

                    if enable_wandb:
                        wandb.init(project="surgease_uceis_zero", entity="elifkcontar")

                        wandb.run.name = "train_" + wandb.run.name.split("-")[2]
                        wandb.run.save()

                        config = wandb.config
                        config.model = model_name
                        config.dataset = dataset_name
                        config.lr = learning_rate
                        config.wd = weight_decay
                        config.bs = batch_size
                        config.num_worker = num_worker
                        config.project = "classification"
                        config.optimizer = optimizer_name
                        config.epoch  = num_epoch
                        config.loss= loss_name

                    
                    train_folders, val_folders, test_folders = split(save=True, dir=wandb.run.dir)
                    #train_folders, val_folders, test_folders = split()
                    while not (check_splits(train_folders, val_folders, test_folders, csv_filename=r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\gi\label_all_cropped.csv')):
                        train_folders, val_folders, test_folders = split()
                        print('another split')
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                

                    tmp_dataset = UCEIS_csv(train_folders, csv_filename=r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\gi\label_all_cropped.csv', transform=None)

                    #Sample x50
                    #train_set_mean = [0.3683037974792629, 0.2545500826256079, 0.21100532802110092]
                    #train_set_std = [0.32256124182067264, 0.23441272870730917, 0.21096266280494616]
                    train_set_mean, train_set_std = get_dataset_mean_and_std(tmp_dataset)
                    normalize = transforms.Normalize(mean=train_set_mean,
                                                    std=train_set_std)

                    train_transform = transforms.Compose([transforms.RandomResizedCrop((480, 540), scale=(0.95, 1.05), ratio=(0.95, 1.05)),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomRotation((180)),
                                                        transforms.ToTensor(),
                                                        normalize])
                    val_transform = transforms.Compose([transforms.ToTensor(),
                                                        normalize])

                    train_dataset = UCEIS_csv(train_folders, csv_filename=r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\gi\label_all_cropped.csv', transform=train_transform)
                    val_dataset = UCEIS_csv(val_folders, csv_filename=r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\gi\label_all_cropped.csv', transform=val_transform)

                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                            num_workers=num_worker,
                                                            shuffle=True,
                                                            pin_memory=True)
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker,
                                                            pin_memory=True)
    
                    model = get_model(model_name)
                    model.to(device)

                    if(loss_name=='CE'):
                        criterion = multitask_CE()
                    elif(loss_name=='WCE'):
                        weights = get_weights(train_folders, csv_filename=r'C:\Users\ElifKübraÇontar\Desktop\gi\label_all_cropped.csv')
                        criterion = multitask_WCE(device, weights)
                    elif(loss_name=='Focal'):
                        criterion = multitask_focal()
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=scheduler_patience, threshold=best_threshold,
                                                            verbose=True)


                    last_epoch = 0
                    for epoch in range(num_epoch):

                        last_epoch = epoch

                        start = time.time()
                        train_loss, tl0, tl1, tl2, train_accuracy_subscore, train_accuracy_UCEIS, train_remission,t_rem_0, t_rem_1, tvasc_0, tvasc_1, tvasc_2, tbleed_0, tbleed_1, tbleed_2, tbleed_3, teros_0, teros_1, teros_2, teros_3  = train(model, device, train_loader, criterion, optimizer)
                        elapsed = time.time() - start

                        val_loss, vl0, vl1, vl2, vac_0, vac_1, vac_2, val_accuracy_subscore, val_accuracy_UCEIS, val_remission, v_rem_0, v_rem_1, vasc_0, vasc_1, vasc_2, bleed_0, bleed_1, bleed_2, bleed_3, eros_0, eros_1, eros_2, eros_3  = validation(model, device, val_loader, criterion)
                        #val_loss, vl0, vl1, vl2, val_accuracy_subscore, val_accuracy_UCEIS = validation(model, device, val_loader, criterion)
                        scheduler.step(val_accuracy_subscore)

                        print("epoch: {:3.0f}".format(epoch + 1) + " | time: {:3.0f} s".format(
                                elapsed) + " | Avg batch time: {:4.3f} s".format(
                                elapsed / len(train_loader)) + " | Train acc subscore: {:4.2f}".format(
                                train_accuracy_subscore * 100) + " | Val acc subscore: {:4.2f}".format(
                                val_accuracy_subscore * 100) + " | Train acc UCEIS: {:4.2f}".format(
                                train_accuracy_UCEIS * 100) + " | Val acc UCEIS: {:4.2f}".format(
                                val_accuracy_UCEIS * 100) + " | Train remission: {:6.4f}".format(
                                train_remission * 100) + " | Val remission: {:6.4f}".format(
                                val_remission * 100) + " | Train loss: {:6.4f}".format(
                                train_loss) + " | T loss_0: {:6.4f}".format(
                                tl0) + " | T loss_1: {:6.4f}".format(
                                tl1)+ " | T loss_2: {:6.4f}".format(
                                tl2) + " | V loss_0: {:6.4f}".format(
                                vl0) + " | V loss_1: {:6.4f}".format(
                                vl1) + " | V loss_2: {:6.4f}".format(
                                vl2) + " | Val loss: {:6.4f}".format(
                                val_loss))
                                
                        if enable_wandb:
                            wandb.log(
                                    {"epoch"     : epoch + 1,
                                    "lr"        : optimizer.param_groups[0]['lr'],
                                    'train loss': train_loss,
                                    'val loss'  : val_loss,
                                    'val vascularity acc': vac_0,
                                    'val bleeding acc': vac_1,
                                    'val erosion acc': vac_2,
                                    'train vascularity_0': tvasc_0,
                                    'train vascularity_1': tvasc_1,
                                    'train vascularity_2': tvasc_2,
                                    'train bleeding_0': tbleed_0,
                                    'train bleeding_1': tbleed_1,
                                    'train bleeding_2': tbleed_2,
                                    'train bleeding_3': tbleed_3,
                                    'train erosion_0': teros_0,
                                    'train erosion_1': teros_1,
                                    'train erosion_2': teros_2,
                                    'train erosion_3': teros_3,
                                    'val vascularity_0': vasc_0,
                                    'val vascularity_1': vasc_1,
                                    'val vascularity_2': vasc_2,
                                    'val bleeding_0': bleed_0,
                                    'val bleeding_1': bleed_1,
                                    'val bleeding_2': bleed_2,
                                    'val bleeding_3': bleed_3,
                                    'val erosion_0': eros_0,
                                    'val erosion_1': eros_1,
                                    'val erosion_2': eros_2,
                                    'val erosion_3': eros_3,
                                    'train acc subscore' : train_accuracy_subscore,
                                    'val acc subscore'   : val_accuracy_subscore,
                                    'train acc UCEIS' : train_accuracy_UCEIS,
                                    'val acc UCEIS'   : val_accuracy_UCEIS,
                                    'train [0]vs[1-8]' : train_remission,
                                    'val [0]vs[1-8]'   : val_remission,
                                    'train [0]vs[1-8] class 0' : t_rem_0,
                                    'val [0]vs[1-8] class 0'   : v_rem_0,
                                    'train [0]vs[1-8] class 1' : t_rem_1,
                                    'val [0]vs[1-8] class 1'   : v_rem_1})

                        if (val_accuracy_subscore > best_acc * (1 + best_threshold)):
                            early_stop_counter = 0
                            best_acc = val_accuracy_subscore
                            if enable_wandb:
                                wandb.run.summary["best accuracy"] = best_acc
                            print("overwriting the best model!")
                            torch.save(model.state_dict(), wandb.run.dir+"/weights_" + model_name + '.pth.tar')
                        else:
                            early_stop_counter += 1

                        if early_stop_counter >= early_stopping_thresh:
                            print("Early stopping at: " + str(epoch))
                            break

                    precision, recall,f1, specificity, precision_severity, recall_severity, f1_severity, specificity_severity = test(model, device, val_loader)

                    if enable_wandb:
                            wandb.log(
                                    {"precision"     : precision,
                                    "recall"        : recall,
                                    'f1' : f1,
                                    'specificity'   : specificity,
                                    "precision [0-3]vs[4-8]"     : precision_severity,
                                    "recall [0-3]vs[4-8]"        : recall_severity,
                                    'f1 [0-3]vs[4-8]' : f1_severity,
                                    'specificity [0-3]vs[4-8]'   : specificity_severity})
                    if enable_wandb:
                        wandb.run.finish()
                        
                    print("------ Training finished ------")
                    print("Best validation set Accuracy: " + str(best_acc))