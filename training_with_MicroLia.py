#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:48:27 2023

@author: karen
"""

import time
import os
from pathlib import Path
import numpy as np
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from MicroLIA import training_set
from MicroLIA import ensemble_model as models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

def set_data(dataset_path, app_w = True):
    path_test = [Path(dataset_path, folder) for folder in os.listdir(dataset_path) if "test" in folder][0]
    path_train = [Path(dataset_path, folder) for folder in os.listdir(dataset_path) if "train" in folder][0] #Path(folders_path,f"training_set-{folders_path[:-1]}") # here we put directories with lightcurves associated to a class
    folders = [name for name in os.listdir(path_test) if not '.txt' in name]
    data_x, data_y = training_set.load_all(path_train, apply_weights=app_w)
    return data_x, data_y, path_train, path_test, folders

def set_model(model_name,  data_x, data_y, path_train, path_test, folders, dataset_path):
    '''Set and create the model
    Get predictions in "predictions.txt" with the columns: class, max_prob [prob of each class] file'''
    model_result = Path("Models_results", model_name+"-"+ dataset_path)
    os.mkdir(model_result)
    generated_files = ["all_features.txt", "lightcurves.fits", "MicroLIA_Training_Set.csv"]
    for file in generated_files:
        os.rename(Path("/home/karen", file), Path(model_result, file))
    model = models.Classifier(data_x,data_y, clf = "xgb", optimize=False, impute=True, balance=True)
    model.create()
    predict_array = np.empty((0,3+len(folders)))
    for i, folder in enumerate(folders):
        print(f"Folder: {folder}    {i+1}/{len(folders)}")
        for file in tqdm(os.listdir(Path(path_test,folder))):
            data = np.loadtxt(Path(path_test,folder, file))
            mjd, mag, magerr = data[:,0], data[:,1], data[:,2]
            pred = model.predict(mjd, mag, magerr, convert=True, zp=27.5)
            predict_array = np.append(predict_array, [[folder,pred[np.argmax(pred[:,1])][0],*pred[:,1],file]],axis=0)
    head_folders = " ".join(folders)
    headers=f"class max_prob {head_folders} lc_file".split(" ")
    predict_df = pd.DataFrame(predict_array, columns=headers)
    predict_df.to_csv('predictions.txt', index=False, sep=' ')
    os.rename("predictions.txt", Path(model_result, file))
#     np.savetxt(Path(model_result, "predictions.txt", predict_array, header= f"class max_prob {" ".join(folders)} lc_file", fmt='%s')
    return model, predict_df # class, max_prob [prob of each class] file

def metrics(y_true, y_pred, model_name, dataset_path):
    '''Plot confusion matrix on "cf_matrix.png"
    Write the presicion, recall, accuracy and f1 score in "score_metrics.txt"'''
    model_result = Path("Models_results", model_name+"-"+ dataset_path.split("/")[-1])
    if model_name == "xgb":
        XGB = {}
        for i, folder in enumerate(sorted(folders)):
            XGB[f"{i}.0"]=folder
        y_pred = [XGB[str(y)] for y in y_pred]

    Labels = [name.split("_")[-1] for name in folders]
    
    cf_matrix = confusion_matrix(y_true, y_pred, labels=folders, normalize='true')
    precision = round(precision_score(y_true, y_pred, average='macro'),3)
    recall = round(recall_score(y_true, y_pred, average='macro'),3)
    accuracy = round(accuracy_score(y_true, y_pred),3)
    f1_score = round((2*precision*recall)/(presicion+recall),3)
    print('Precision score [tp/(tp+fp)] = ', precision)
    print('Recall score [tp/(tp+fn)] = ', recall)
    print('Accuracy = ', accuracy)
    print('F1_score [2*presicion*recall/(pres+recall)] = ', f1_score)
    
    fig, ax = plt.subplots(figsize=(5,5))  
    fontsize=10
    sns.set(font_scale=fontsize/10)
    s = sns.heatmap(cf_matrix, annot=True, cmap='viridis', xticklabels=Labels, yticklabels=Labels, ax=ax)  # cmap='OrRd'
    plt.xticks(rotation=90,fontsize=fontsize)
    plt.yticks(rotation=0,fontsize=fontsize) 
    plt.ylabel('True target',fontsize=fontsize) 
    plt.xlabel('Prediction (max_prom)',fontsize=fontsize) 
    plt.title(f'{model_name} {dataset_path.split("/")[0]}\n{precision}/{recall}/{accuracy}/{f1_score}')
    plt.savefig(Path(model_result, "cf_matrix.png"))
    plt.close()
    with open(Path(model_result,"score_metrics.txt"), "w") as f:
        f.write(f'Precision score [tp/(tp+fp)] = {precision}\n' )
        f.write(f'Recall score [tp/(tp+fn)] = {recall}\n' )
        f.write(f'Accuracy = {accuracy}\n' )
        f.write(f'F1_score [2*presicion*recall/(pres+recall)] = {f1_score}\n' )
    return cf_matrix, precision, recall, accuracy, f1_score


def run_model(model_name, dataset_path):
    path_train = [folder for folder in os.listdir(dataset_path) if "train" in folder][0] #Path(folders_path,f"training_set-{folders_path[:-1]}") # here we put directories with lightcurves associated to a class
    app_w = True
    data_x, data_y = training_set.load_all(Path(dataset_path, path_train), apply_weights=app_w)

    model_result = Path("Models_results", model_name+"-"+ dataset_path)
    os.mkdir(model_result)
    generated_files = ["all_features_.txt",
    "lightcurves__.fits", "MicroLIA_Training_Set.csv"]
    for file in generated_files:
        os.rename(Path("..", file), Path(model_result, file))

    data = np.loadtxt(Path(model_result, "all_features_.txt"), dtype=str)

    data_x = data[:,2:].astype('float')
    data_y = data[:,0]

    model = models.Classifier(data_x,data_y, clf = model_name, optimize=False, impute=True, balance=True)
    model.create()

    path_test = [folder for folder in os.listdir(dataset_path) if "test" in folder][0]
    folders = [name for name in os.listdir(Path(dataset_path, path_test)) if not '.txt' in name]

    int_array = np.empty((0,3), int)
    for i, folder in enumerate(folders):
        print(f"Folder: {folder}    {i+1}/{len(folders)}")
        for file in tqdm(os.listdir(Path(dataset_path, path_test,folder))):
            data = np.loadtxt(Path(dataset_path, path_test,folder, file))
            mjd, mag, magerr = data[:,0], data[:,1], data[:,2]
            prediction = model.predict(mjd, mag, magerr, convert=True, zp=27.5)
            pred = []
            for i in range(len(prediction[:,1])):
                pred.append(float(prediction[:,1][i])) 
            int_array = np.append(int_array, [[folder,prediction[np.argmax(pred)][0],file]],axis=0)

    classification_file = f'clasificacion_{model_name}_galactic_elastic.txt'
    np.savetxt(Path(model_result, classification_file), int_array, fmt='%s')

    cm = pd.read_csv(Path(model_result, classification_file),sep=' ')
    cm = cm.values

    y_true = cm[:,0]
    y_pred = cm[:,1]

    if model_name == "xgb":
        XGB = {'0.0':'ELASTICC_TRAIN_EB',
        '1.0':'ELASTICC_TRAIN_Mdwarf-flare',
        '2.0':'ELASTICC_TRAIN_RRL',
        '3.0':'ELASTICC_TRAIN_uLens-Single_PyLIMA'}
        y_pred = [XGB[str(y)] for y in y_pred]

    print(y_true[0],y_pred[0])

    labels = [ 'ELASTICC_TRAIN_uLens-Single_PyLIMA','ELASTICC_TRAIN_RRL','ELASTICC_TRAIN_EB','ELASTICC_TRAIN_Mdwarf-flare']
    Labels = ['uLens','RRL','EB', 'Mdwarf-flare']

    cf_matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    precision = round(precision_score(y_true, y_pred, average='macro'),3)
    recall = round(recall_score(y_true, y_pred, average='macro'),3)
    accuracy = round(accuracy_score(y_true, y_pred),3)
    print('Precision score [tp/(tp+fp)] = ', precision)
    print('Recall score [tp/(tp+fn)] = ', recall)
    print('Accuracy = ', accuracy)

    fig, ax = plt.subplots(figsize=(5,5))  
    fontsize=10
    sns.set(font_scale=fontsize/10)
    s = sns.heatmap(cf_matrix, annot=True, cmap='viridis', xticklabels=Labels, yticklabels=Labels, ax=ax)  # cmap='OrRd'
    plt.xticks(rotation=90,fontsize=fontsize)
    plt.yticks(rotation=0,fontsize=fontsize) 
    plt.ylabel('True target',fontsize=fontsize) 
    plt.xlabel('Prediction',fontsize=fontsize) 
    plt.title(f'{model_name} (ELASTICC)')
    plt.savefig(Path(model_result, "cf_matrix.png"))
    plt.close()
    with open(Path(model_result,"score_metrics.txt"), "w") as f:
        f.write(f'Precision score [tp/(tp+fp)] = {precision}\n' )
        f.write(f'Recall score [tp/(tp+fn)] = {recall}\n' )
        f.write(f'Accuracy = {accuracy}\n' )


runs = []
# dataset_paths = ["2305222212"+ i for i in "ugrizY"]
model_names = ["rf", "nn", "xgb"]
dataset_path = ["Anibal_r", "Valid"]
# for model in models_names:
#     for run in dataset_path:
#         runs.append((model, run))

# dataset_path =     # Path of the folder where are the folders of the test and of the train sets
# model_name =       # "xgb", "nn", "rf"
for dataset in dataset_path: 
    data_x, data_y, path_train, path_test, folders = set_data(dataset,  app_w = True)
    for model_name in model_names:
        model, predict_df = set_model(model_name, data_x, data_y, path_train, path_test, folders, dataset)
        y_true = predict_df["class"]
        y_pred = predict_df["max_prom"]
        cf_matrix, precision, recall, accuracy, f1_score = metrics(y_true, y_pred, model_name, dataset_path)

# for run in runs:
#     print("--------------------------------------------------------------------------------------------")
#     print(run[0])
#     print(run[1])
#     run_model(*run)