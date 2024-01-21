#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:52:20 2023

@author: karen
"""

import warnings

# Filter out the specific warning
warnings.filterwarnings("ignore", message="The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().")
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from copy import copy
from tqdm import tqdm


def plot_lc(MJD, M, Merr, title=None, color = None, alpha = 1):
    bandcolor = {'u':'b', 'g':'c', 'r':'g', 'i':'orange', 'z':'r', 'y':'m'}
    if color != None:
        for band in bandcolor:
            bandcolor[band] = color
    for band in MJD.keys():
        plt.errorbar(MJD[band], M[band], Merr[band], marker="o", linestyle=" ", color = bandcolor[band.lower()], label=band, alpha = alpha)
    plt.gca().invert_yaxis()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel("MJD")
    plt.ylabel("Magnitude")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(title)

def read_event(path, bands = "ugrizY"):
    '''path: dict {band:path}'''
    MJD = {}; M = {}; Merr = {}
    for band in path.keys():
        MJD[band], M[band], Merr[band] = np.loadtxt(path[band]).T
    return MJD, M, Merr

def event_catalog(catalog, bands, i_star, plot=True, color =None): 
    '''DataSet: catalog name or model (folder name)'''
    path = [f"{catalog}_{band}/lc_{catalog}_{band}_{i_star}.txt" for band in bands]
    path = dict(zip(bands, path))
    MJD, M, Merr = read_event(path, bands = bands)
    if plot:
        plot_lc(MJD, M, Merr, title = f"Catalog: {catalog} | Star index: {i_star}", color = color)
    return MJD, M, Merr

def filter_m_sat(MJD, M, Merr, plot = False, catalog = "X"): # Si uso mas de una banda va a tirar error: corregir!
    Msat = {'u':14.7, 'g': 15.7, 'r': 15.8, 'i': 15.8, 'z': 15.3, 'y': 13.9}
    M_copy = copy(M); Merr_copy = copy(Merr); MJD_copy= copy(MJD)
    for band in MJD.keys(): #Msat
        M[band] = [m for m, merr in zip(M_copy[band], Merr_copy[band]) if m>Msat[band.lower()]]
        MJD[band] = [mjd for mjd,m,merr in zip(MJD_copy[band], M_copy[band], Merr_copy[band]) if m>Msat[band.lower()]]
        Merr[band] = [merr for m, merr in zip(M_copy[band], Merr_copy[band]) if m>Msat[band.lower()]]
    del_points = len(M[band])-len(M_copy[band])
    if plot:
        bandcolor = {'u':'b', 'g':'c', 'r':'g', 'i':'orange', 'z':'r', 'y':'m'}
        plot_lc( MJD_copy, M_copy, Merr_copy, alpha = 0.2)
        for msat, band in zip(Msat.values(), MJD.keys()):
            plt.hlines(msat, xmin=60342.38001091364, xmax=63856.0782499267, color = bandcolor[band.lower()], linestyles="-")
        plot_lc(MJD, M, Merr)
    return MJD, M, Merr, del_points

def filter_m_5(MJD, M, Merr, N_points = 10, N_sigma = 1, plot = False, catalog = "X"):
    M5 = {'u':23.78, 'g':24.81, 'r':24.35 , 'i':23.92, 'z': 23.34 , 'y':22.45} # 5-sigma depth is a measure of the faintest object (in terms of magnit
    M_copy = copy(M); Merr_copy = copy(Merr); MJD_copy= copy(MJD)
    for band in MJD.keys(): #M5
        M_copy[band] = [m for m, merr in zip(M[band], Merr[band]) if m+N_sigma*merr<M5[band.lower()]]
        band_stat= True
        if len(M_copy[band])<N_points:
            M[band] = []; MJD[band] = [] ;Merr[band] = []
            band_stat = False
    if plot:
        bandcolor = {'u':'b', 'g':'c', 'r':'g', 'i':'orange', 'z':'r', 'y':'m'}
        plot_lc( MJD_copy, M_copy, Merr_copy, alpha = 0.2)
        for m5, band in zip(M5.values(), MJD.keys()):
            plt.hlines(m5, xmin=60342.38001091364, xmax=63856.0782499267, color = bandcolor[band.lower()], linestyles="--")
        plot_lc(MJD, M, Merr)
    return MJD, M, Merr, band_stat

def filter_peak_uLens(mjd, m, merr, n_sigmas=3, n_pts = 4):
    median = np.median(m)
    i=0
    for mjdi, mi, merri in zip(mjd,m, merr): 
        if mi+n_sigmas*merri < median:
            i +=1
    if i>n_pts:
        return True
    return False

def filter_peak(mjd, m, merr, n_sigmas=3, n_pts = 4):
    median = np.median(m)
    i=0
    for mjdi, mi, merri in zip(mjd,m, merr): 
        if abs(mi-median) > n_sigmas*merri:
            i +=1
    if i>n_pts+1:
        return True
    return False

def chi2(m, merr, cte):
    chi=[((mi-cte)**2)/(merri**2) for mi, merri in zip(m, merr)]
    return np.sum(chi)/len(m)

def apply_filters(dataset, name):
    if f"Filters_{name}" not in os.listdir():
        os.mkdir(f"Filters_{name}")
    if "cte" in name:
        files = sorted(os.listdir(dataset))
        files_id = [int(i[:-4].split("_")[-1]) for i in files]
        files_sort = [None] * len(files_id)
        for i, j in enumerate(files_id):
            files_sort[j] = files[i]
    else:
        files_sort = sorted(os.listdir(dataset))
    with open(f"Filters_{name}/results.txt", "w") as f:
        for lc, i in enumerate(tqdm(files_sort, desc = name)):
            path = dataset+"/"+i
            if name == "cte":
                mjd, m, merr = event_catalog("trilegal2", "i", lc, plot=False)
                mjd = {"i": [ii for ii in mjd["i"] if ii < 61309.07516768833]}
                m= {"i":  [j for ii,j in zip(mjd["i"], m["i"]) if ii < 61309.07516768833]}
                merr= {"i":  [j for ii,j in zip(mjd["i"], merr["i"]) if ii < 61309.07516768833]}
            else:
                mjd, m, merr = read_event({"i":path}, "i")
            mjd, m, merr, del_points = filter_m_sat(mjd, m, merr, catalog = "name")
            mjd, m, merr, stat = filter_m_5(mjd, m, merr, catalog = "name")
            np.savetxt(path, np.array([mjd["i"], m["i"], merr["i"]]).T)
            median = np.median(m["i"])
            end_points = len(m["i"])
            have_peak_ml = filter_peak_uLens(mjd["i"], m["i"], merr["i"], n_pts=4)
            have_peak = filter_peak(mjd["i"], m["i"], merr["i"], n_pts=4)
            plt.figure()
            plt.errorbar(mjd["i"], m["i"], merr["i"], marker="o", linestyle=" ")
            chi_lc = chi2(m["i"], merr["i"], median)
            chi_lim_2 = chi_lc>2
            chi_lim_3 = chi_lc>3
            plt.title(f"{lc} peaks = {stat} | peak = {have_peak} | chi = {chi_lc}")
            plt.gca().invert_yaxis()
            plt.hlines(median,xmin=60342.38001091364, xmax=61309.07516768833)
            plt.savefig(f"Filters_{name}/Star_{lc}")
            plt.close()
            f.write(f"{i}\t{del_points}\t{end_points}\t{stat}\t{have_peak}\t{have_peak_ml}\t{chi_lc}\t{chi_lim_2}\t{chi_lim_3}\n")

Sets_path = {"uLens_train": "2305222212_i/training_set-2305222212/ELASTICC_TRAIN_uLens-Single_PyLIMA", 
              "uLens_test": "2305222212_i/test_set-2305222212/ELASTICC_TRAIN_uLens-Single_PyLIMA", 
              "RRL_train":"2305222212_i/training_set-2305222212/ELASTICC_TRAIN_RRL", 
              "RRL_test":"2305222212_i/test_set-2305222212/ELASTICC_TRAIN_RRL", 
              "EB_train":"2305222212_i/training_set-2305222212/ELASTICC_TRAIN_EB", 
              "EB_test":"2305222212_i/test_set-2305222212/ELASTICC_TRAIN_EB",
              "cte":"trilegal2_i"}

Sets_path = { "cte":"trilegal2_i"}

for Set in Sets_path:
    apply_filters(Sets_path[Set], Set)
    gc.collect()

    
import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
uLens_train = pd.read_csv("Filters_uLens_train/results.txt", sep = "\t", header=None, names = ["path","del_peaks", "end_peaks", "points", "peak", "peak_uLens", "chi", "chi_2", "chi_3"])
uLens_test = pd.read_csv("Filters_uLens_test/results.txt", sep = "\t", header=None,names = ["path","del_peaks", "end_peaks", "points", "peak", "peak_uLens", "chi", "chi_2", "chi_3"])
RRL_train = pd.read_csv("Filters_RRL_train/results.txt", sep = "\t", header=None,names = ["path","del_peaks", "end_peaks", "points", "peak", "peak_uLens", "chi", "chi_2", "chi_3"])
RRL_test = pd.read_csv("Filters_RRL_test/results.txt", sep = "\t", header=None, names = ["path","del_peaks", "end_peaks", "points", "peak", "peak_uLens", "chi", "chi_2", "chi_3"])
EB_train = pd.read_csv("Filters_EB_train/results.txt", sep = "\t", header=None, names = ["path","del_peaks", "end_peaks", "points", "peak", "peak_uLens", "chi", "chi_2", "chi_3"])
EB_test = pd.read_csv("Filters_EB_test/results.txt", sep = "\t", header=None, names = ["path","del_peaks", "end_peaks", "points", "peak", "peak_uLens", "chi", "chi_2", "chi_3"])
ctes = pd.read_csv("Filters_cte/results.txt", sep = "\t", header=None, names = ["path","del_peaks", "end_peaks", "points", "peak", "peak_uLens", "chi", "chi_2", "chi_3"])
Sets = {"uLens_train": uLens_train,"uLens_test":uLens_test, "RRL_train":RRL_train, 
              "RRL_test": RRL_test, "EB_train": EB_train, "EB_test": EB_test, "cte":ctes}

Sets_path = {"uLens_train": "2305222212_i/training_set-2305222212/ELASTICC_TRAIN_uLens-Single_PyLIMA", 
              "uLens_test": "2305222212_i/test_set-2305222212/ELASTICC_TRAIN_uLens-Single_PyLIMA", 
              "RRL_train":"2305222212_i/training_set-2305222212/ELASTICC_TRAIN_RRL", 
              "RRL_test":"2305222212_i/test_set-2305222212/ELASTICC_TRAIN_RRL", 
              "EB_train":"2305222212_i/training_set-2305222212/ELASTICC_TRAIN_EB", 
              "EB_test":"2305222212_i/test_set-2305222212/ELASTICC_TRAIN_EB",
              "cte":"trilegal2_i"}
            
os.mkdir("Valid")
for name, path, df in zip(Sets, Sets_path.values(), Sets.values()):
    valid_path = df[df["points"] == True]["path"]
    os.mkdir("Valid/"+name)
    for lc_path in valid_path:
        try:
            shutil.copy(path+"/"+lc_path, "Valid/"+name+"/"+lc_path)
        except FileNotFoundError:
            print("File not found error: ", path+"/"+lc_path)
            pass



uLens = pd.concat([uLens_train, uLens_test], axis=0)
EB = pd.concat([EB_train, EB_test], axis=0)
RRL = pd.concat([RRL_train, RRL_test], axis=0)

condition = (uLens_test['peaks'] == True) & (uLens_test['peak_uLens'] == True)
uLens_test_ok = uLens_test.index[condition].tolist()




ds = {"uLens": uLens, "RRL":RRL, "EB":EB, "ctes":ctes}

uLens["chi"].values[uLens["chi"].values >5] = 5
uLens.hist(bins=20, figsize=(12, 8))
RRL["chi"].values[RRL["chi"].values >5] = 5
RRL.hist(bins=20, figsize=(12, 8))
EB["chi"].values[EB["chi"].values >5] = 5
EB.hist(bins=20, figsize=(12, 8))
ctes["chi"].values[ctes["chi"].values >5] = 5
ctes.hist(bins=20, figsize=(12, 8))

plt.figure()
plt.title("All")
plt.hist(uLens["chi"], density = True, bins = 100, label = "uLens")
plt.hist(EB["chi"], density = True, bins = 100, label = "EB")
plt.hist(RRL["chi"], density = True, bins = 100, label = "RRL")
plt.hist(ctes["chi"], density = True, label = "ctes")
plt.legend()
plt.xlabel("$\chi^2$")
plt.xlim(0,5)
plt.ylim(0,5)

plt.figure()
plt.title("Only valid")
plt.hist(uLens.groupby(["peaks"]).get_group(True)["chi"], density = True, bins = 100, label = "uLens")
plt.hist(EB.groupby(["peaks"]).get_group(True)["chi"], density = True, bins = 100, label = "EB")
plt.hist(RRL.groupby(["peaks"]).get_group(True)["chi"], density = True, bins = 100, label = "RRL")
plt.hist(ctes.groupby(["peaks"]).get_group(True)["chi"], density = True, label = "ctes")
plt.legend()
plt.xlabel("$\chi^2$")
plt.xlim(0,5)
plt.ylim(0,5)

plt.figure()
plt.title("Valid and With peak")
plt.hist(uLens.groupby(["peaks"]).get_group(True).groupby(["peak"]).get_group(True)["chi"], density = True, bins = 100, label = "uLens")
plt.hist(EB.groupby(["peaks"]).get_group(True).groupby(["peak"]).get_group(True)["chi"], density = True, bins = 100, label = "EB")
plt.hist(RRL.groupby(["peaks"]).get_group(True).groupby(["peak"]).get_group(True)["chi"], density = True, bins = 100, label = "RRL")
plt.hist(ctes.groupby(["peaks"]).get_group(True)["chi"], density = True, label = "ctes")
plt.legend()
plt.xlabel("$\chi^2$")
plt.xlim(0,5)
plt.ylim(0,5)

plt.figure()
plt.title("Valid and With Out peak")
plt.hist(uLens.groupby(["peaks"]).get_group(True).groupby(["peak"]).get_group(False)["chi"], density = True, bins = 100, label = "uLens")
plt.hist(EB.groupby(["peaks"]).get_group(True).groupby(["peak"]).get_group(False)["chi"], density = True, bins = 100, label = "EB")
plt.hist(RRL.groupby(["peaks"]).get_group(True).groupby(["peak"]).get_group(False)["chi"], density = True, bins = 100, label = "RRL")
plt.hist(ctes.groupby(["peaks"]).get_group(True)["chi"], density = True, label = "ctes")
plt.legend()
plt.xlim(0,5)
plt.ylim(0,5)

plt.figure()
plt.title("Valid and With peak uLens")
plt.hist(uLens.groupby(["peaks"]).get_group(True).groupby(["peak_uLens"]).get_group(False)["chi"], density = True, bins = 100, label = "uLens_without_peak")
plt.hist(uLens.groupby(["peaks"]).get_group(True).groupby(["peak_uLens"]).get_group(True)["chi"], density = True, bins = 100, label = "uLens_with_peak")
plt.hist(ctes.groupby(["peaks"]).get_group(True)["chi"], density = True, label = "ctes")
plt.legend()
plt.xlabel("$\chi^2$")
plt.xlim(0,5)
plt.ylim(0,5)


from copy import copy
import matplotlib.pyplot as plt


ds = {"uLens": uLens, "RRL":RRL, "EB":EB}
bool_features = ["peak", "peak_uLens", "chi_2", "chi_3"]    
for name, df in zip(ds, ds.values())        :
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(name + " " + str(len(df[df["points"]==True])), fontsize=16)
    plt.subplots_adjust(hspace=0.45, wspace=0.3)
    for i, feat1 in enumerate(bool_features):
        for j, feat2 in enumerate(bool_features):
            ax = axes[i, j]
            df = df[df["points"]==True]
            df_trues = df[(df[feat1] == True) & (df[feat2] == True)]
            hist, bins, _ = ax.hist(df_trues['chi'], density = True, bins=20)
            sorted_hist = sorted(hist, reverse=True)
            second_max = sorted_hist[1] if len(sorted_hist) > 1 else 0
            ax.set_ylim(0, second_max * 2)
            ax.set_title(len(df_trues["chi"]))  
    for i, feat1 in enumerate(bool_features):
        axes[i, 0].text(-0.35, 0.5, feat1, fontsize=12, rotation=90, transform=axes[i, 0].transAxes, va='center', ha='center')
        axes[0, i].text(0.5, 1.3, feat1, fontsize=12, transform=axes[0, i].transAxes, va='center', ha='center')


