#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:31:30 2023

@author: karen
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from copy import copy
from tqdm import tqdm
import rubin_sim
import rubin_sim.maf as maf # LSST's Metrics Analysis Framework (MAF).
from rubin_sim.phot_utils import signaltonoise
from rubin_sim.phot_utils import PhotometricParameters
from rubin_sim.phot_utils import bandpass_dict
from rubin_sim.data import get_baseline
import astropy.units as u
from astropy.coordinates import SkyCoord
#min(list(dataSlice['observationStartMJD'])) 60342.38001091364

def read_lc(path):
    MJD, M, Merr = np.loadtxt(path).T
    return MJD, M, Merr

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

def plot_constant_lc(M_input, simulated_curves, alpha = 1):
    bandcolor = {'u':'b', 'g':'c', 'r':'g', 'i':'orange', 'z':'r', 'y':'m'}
    for i, sim_lc in enumerate(simulated_curves):
        plt.figure()
        plt.gca().invert_yaxis()
        plot_lc(sim_lc["MJD"],sim_lc["M"],sim_lc["Merr"], alpha = alpha)
        plt.title("Light curve "+str(i))
        for j, band in enumerate(M_input.keys()):
            plt.hlines(M_input.iloc[i][band], xmin=60342.38001091364, xmax=63856.0782499267, color=bandcolor[band.lower()])
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    

def constant_lc_simulation(M_input, ra, dec, opsim, save_path = None):
    photParams = PhotometricParameters(exptime=30,nexp=1, readnoise=None) # Photometry: Exposure seconds and times
    LSST_BandPass = bandpass_dict.BandpassDict.load_total_bandpasses_from_files() # Info about filters
    # default5sigma = {'u':23.78, 'g':24.81, 'r':24.35 , 'i':23.92, 'z': 23.34 , 'y':22.45} # 5-sigma depth is a measure of the faintest object (in terms of magnitude) that can be detected in an image with a 5-sigma confidence. 
    metric = rubin_sim.maf.metrics.PassMetric(cols=['filter', 'observationStartMJD', 'fiveSigmaDepth']) # Metrics interested in
    slicer = maf.slicers.UserPointsSlicer(ra=[ra], dec=[dec]) # # Locations interested in - Slice the sky in the point ra,dec
    sql = ''
    bundleMetrics = maf.MetricBundle(metric, slicer, sql) # Metrics ran per slice_point (dict)
    bundle = maf.MetricBundleGroup([bundleMetrics], opsim, out_dir="temp") #This generate ResultsDb with the metrics and stats
    bundle.run_all(plot_now=False) # write metrics in MetricBundle.metric_values #Try plot now to show metrics results
    dataSlice = bundleMetrics.metric_values[0] 

    m5 = {}; MJD = {}; M = {}; Merr = {}
    for band in M_input.keys():
        m5[band] = dataSlice['fiveSigmaDepth'][np.where(dataSlice['filter'] == band.lower())]
        MJD[band] = dataSlice['observationStartMJD'][np.where(dataSlice['filter'] == band.lower())] #+ 2400000.5
        Merr[band] = [signaltonoise.calc_mag_error_m5(M_input[band],  LSST_BandPass[band.lower()], M5 , photParams)[0] for M5 in m5[band]]
        M[band] = [np.random.normal(M_input[band],magerr) for magerr in Merr[band]]  
        # mags = dataSlice['fiveSigmaDepth'][np.where(dataSlice['filter'] == band)]
        if save_path != None:
            np.savetxt(save_path+'/lightcurve_'+band.lower()+'.txt', np.array([MJD[band], M[band], Merr[band]]).T)
    return MJD, M, Merr




def stars_constant_lc_simulation(ra, dec, opsim, stars_catalog, n_stars="all", out="save"):
    ''' 
    stars_file: Output of trilegal
    n_lc: Amount of light curves to simulate from de stars_file
    out: 
        -"save": Only save light curves in folders
        -"load": Only load light curves in the variable simulated_curves
        -"all": Save and load.
    '''
    catalog_name = stars_catalog.split("/")[-1].split(".")[0]
    M_input = pd.read_csv(stars_catalog, usecols=["u", "g", "r", "i", "z", "Y"], sep="\s+", decimal ='.')
    M_input = M_input.drop([len(M_input['u'])-1])
    print("Catalog: "+ catalog_name)
    for band in M_input.keys():
        name = f"{catalog_name}_{band}"
        if name not in os.listdir():
            os.mkdir(name)
    sim_curves = []          # List of tuples (mjd, mag, magerr, m5) wich are dictonaries
    for i_star in tqdm(range(n_stars)):
        mjd, mag, magerr = constant_lc_simulation(M_input.iloc[i_star], ra, dec, opsim)
        if (out=="save") or (out=="all"):
            for band in M_input.keys():
                MJD = mjd[band]
                M = mag[band]
                Merr = magerr[band]
                np.savetxt(f'{catalog_name}_{band}/lc_{catalog_name}_{band}_{i_star}.txt', np.array([MJD, M, Merr]).T)
            del mjd, mag, magerr
        if (out == "load") or (out == "all"):
            sim_curves.append(dict(zip(["MJD","M", "Merr"], [mjd, mag, magerr])))
    if out != "save":    
        return M_input, sim_curves


def filter_M5(MJD, M, Merr, N_sigma = 1, plot = False):
    M5 = {'u':23.78, 'g':24.81, 'r':24.35 , 'i':23.92, 'z': 23.34 , 'y':22.45} # 5-sigma depth is a measure of the faintest object (in terms of magnit
    M_copy = copy(M); Merr_copy = copy(Merr); MJD_copy= copy(MJD)
    print("Filter M5 - Points removed:")
    print("{:^5} {:^10} {:^10} {:^10}".format('Band', 'Before', 'After', 'Removed'))
    for band in MJD.keys():
        M[band] = [m for m, merr in zip(M_copy[band], Merr_copy[band]) if m+N_sigma*merr<M5[band.lower()]]
        MJD[band] = [mjd for mjd,m,merr in zip(MJD_copy[band], M_copy[band], Merr_copy[band]) if m+N_sigma*merr<M5[band.lower()]]
        Merr[band] = [merr for m, merr in zip(M_copy[band], Merr_copy[band]) if m+N_sigma*merr<M5[band.lower()]]
        print("{:^5} {:^10} {:^10} {:^10}".format(band, len(MJD_copy[band]), len(MJD[band]), len(MJD_copy[band])-len(MJD[band])))
    if plot:
        bandcolor = {'u':'b', 'g':'c', 'r':'g', 'i':'orange', 'z':'r', 'y':'m'}
        plot_lc( MJD_copy, M_copy, Merr_copy, alpha = 0.2)
        for m5, band in zip(M5.values(), MJD.keys()):
            plt.hlines(m5, xmin=60342.38001091364, xmax=63856.0782499267, color = bandcolor[band.lower()], linestyles="--")
        plot_lc(MJD, M, Merr)
    return MJD, M, Merr

def filter_saturation(MJD, M, Merr, N_sigma = 1, plot = False):
    Msat = {'u':14.7, 'g': 15.7, 'r': 15.8, 'i': 15.8, 'z': 15.3, 'y': 13.9}
    M_copy = copy(M); Merr_copy = copy(Merr); MJD_copy= copy(MJD)
    print("Filter Saturation - Points removed:")
    print("{:^5} {:^10} {:^10} {:^10}".format('Band', 'Before', 'After', 'Removed'))
    for band in MJD.keys():
        M[band] = [m for m, merr in zip(M_copy[band], Merr_copy[band]) if m>Msat[band.lower()]]
        MJD[band] = [mjd for mjd,m,merr in zip(MJD_copy[band], M_copy[band], Merr_copy[band]) if m>Msat[band.lower()]]
        Merr[band] = [merr for m, merr in zip(M_copy[band], Merr_copy[band]) if m>Msat[band.lower()]]
        print("{:^5} {:^10} {:^10} {:^10}".format(band, len(MJD_copy[band]), len(MJD_copy[band]), len(MJD_copy[band])-len(MJD[band])))
    if plot:
        bandcolor = {'u':'b', 'g':'c', 'r':'g', 'i':'orange', 'z':'r', 'y':'m'}
        plot_lc( MJD_copy, M_copy, Merr_copy, alpha = 0.2)
        for msat, band in zip(Msat.values(), MJD.keys()):
            plt.hlines(msat, xmin=60342.38001091364, xmax=63856.0782499267, color = bandcolor[band.lower()], linestyles="-.")
        plot_lc(MJD, M, Merr)
    return MJD, M, Merr

def filter_points(MJD, M, Merr, N_points = 10, N_sigma = 1, plot = False, log_file="log_filter_point.txt"):
    M5 = {'u':23.78, 'g':24.81, 'r':24.35 , 'i':23.92, 'z': 23.34 , 'y':22.45} # 5-sigma depth is a measure of the faintest object (in terms of magnit
    Msat = {'u':14.7, 'g': 15.7, 'r': 15.8, 'i': 15.8, 'z': 15.3, 'y': 13.9}
    M_copy = copy(M); Merr_copy = copy(Merr); MJD_copy= copy(MJD)
    with open(log_file, "a") as f:
        print("Filter M5 + Saturation = Points removed:")
        print("{:^5} {:^10} {:^10} {:^10} {:^10} {:^10}".format('Band', 'Input', 'M5_filt', 'Sat_filt', "Output", "Status"))
        f.write("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n".format('Band', 'Input', 'M5_filt', 'Sat_filt', "Output", "Status"))
        for band in MJD.keys(): #M5
            M[band] = [m for m, merr in zip(M_copy[band], Merr_copy[band]) if m+N_sigma*merr<M5[band.lower()]]
            MJD[band] = [mjd for mjd,m,merr in zip(MJD_copy[band], M_copy[band], Merr_copy[band]) if m+N_sigma*merr<M5[band.lower()]]
            Merr[band] = [merr for m, merr in zip(M_copy[band], Merr_copy[band]) if m+N_sigma*merr<M5[band.lower()]]
            M_copy1 = copy(M); Merr_copy1 = copy(Merr); MJD_copy1= copy(MJD)
        for band in MJD.keys(): #Msat
            M[band] = [m for m, merr in zip(M_copy1[band], Merr_copy1[band]) if m>Msat[band.lower()]]
            MJD[band] = [mjd for mjd,m,merr in zip(MJD_copy1[band], M_copy1[band], Merr_copy1[band]) if m>Msat[band.lower()]]
            Merr[band] = [merr for m, merr in zip(M_copy1[band], Merr_copy1[band]) if m>Msat[band.lower()]]
            band_stat= ""
            if len(M[band])<N_points:
                M[band] = []; MJD[band] = [] ;Merr[band] = []
                band_stat = "Removed"
            print("{:^5} {:^10} {:^10} {:^10} {:^10} {:^10}".format(band,len(MJD_copy[band]), len(MJD_copy[band])-len(MJD_copy1[band]), len(MJD_copy1[band])-len(MJD[band]), len(MJD_copy1[band]), band_stat))
            f.write("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n".format(band,len(MJD_copy[band]), len(MJD_copy[band])-len(MJD_copy1[band]), len(MJD_copy1[band])-len(MJD[band]), len(MJD_copy1[band]), band_stat))
    if plot:
        bandcolor = {'u':'b', 'g':'c', 'r':'g', 'i':'orange', 'z':'r', 'y':'m'}
        plot_lc( MJD_copy, M_copy, Merr_copy, alpha = 0.2)
        for m5, band in zip(M5.values(), MJD.keys()):
            plt.hlines(m5, xmin=60342.38001091364, xmax=63856.0782499267, color = bandcolor[band.lower()], linestyles="--")
        for msat, band in zip(Msat.values(), MJD.keys()):
            plt.hlines(msat, xmin=60342.38001091364, xmax=63856.0782499267, color = bandcolor[band.lower()], linestyles="-")
        plot_lc(MJD, M, Merr)
    
    return MJD, M, Merr

# ---- CODE -----

# OpSim-  Observation strategy: mjd, 5sigma
opsim = get_baseline()

# Location of the star
galactic_center = SkyCoord(l=0*u.degree, b=0*u.degree, frame='galactic')
coord = galactic_center.transform_to('icrs')
ra, dec = coord.ra.deg, coord.dec.deg

n_stars = 30

# # Simulation of 6000 constant lc
stars_constant_lc_simulation(ra, dec, opsim, stars_catalog = "trilegal2.dat", n_stars=2, out="save")

# # Constant curve simulation from an trilegal file.
# M_input, sim_curves = stars_constant_lc_simulation(ra, dec, opsim, stars_catalog = "trilegal2.dat", n_stars=10, out="all")
# plot_constant_lc(M_input, sim_curves)

# 

# Unique curve simulation from m input values
# MJD, Mag, Magerr = constant_lc_simulation(M_input.iloc[2], ra, dec, opsim, save_path = None)
# plot_lc(MJD,Mag, Magerr, title="lightcurve 4")

# Loading, filtering and ploting each lc from an already generated dataset of curves
for i_star in range(n_stars):
    if i_star % 5 == 0:
        plot = True
        plt.figure()
    else:
        plot = False
    print("---- Star Index: "+str(i_star))
    MJD, M, Merr = event_catalog("trilegal2", "ugrizY", i_star, plot=False)
    MJD, M, Merr = filter_points(MJD, M, Merr, plot = plot, log_file = "log_trilegal2_filter.txt")
    if plot:
        plt.title(f"Star Index: {i_star} + Filter M5 and saturation")
        plt.gca().invert_yaxis()
        plt.savefig(f"filter_plots/Star_{i_star}.png")
    
    
    
    
# def merge_figures():
#     import matplotlib.pyplot as plt
#     import matplotlib.image as mpimg
    
    
#     num_subplots_per_figure = 12  # Now 3 rows x 4 columns = 12 subplots
#     num_stars = 29  # Number of saved figures
#     num_figures = -(-num_stars // num_subplots_per_figure)  # Equivalent to math.ceil(num_stars / num_subplots_per_figure)
    
#     fig = None
#     for i_star in range(num_stars):
#         if i_star % num_subplots_per_figure == 0:
#             if fig is not None:  # Save or display the previous figure if it exists
#                 plt.tight_layout()
#                 plt.subplots_adjust(hspace=0.003, wspace=0.03)  # Adjust vertical and horizontal space
#                 plt.show()
#             fig, axes = plt.subplots(3, 4, figsize=(25, 13))  # Start a new figure with 3x4 subplots and bigger size
    
#         i = i_star % num_subplots_per_figure
#         row = i // 4
#         col = i % 4
#         ax = axes[row, col]
        
#         # Read in saved plot
#         img = mpimg.imread(f'Filters_results/Figure 2023-08-31 154605 ({i_star}).png')
#         ax.imshow(img)
#         ax.axis('off')  # Hide axes
#         ax.set_title(f"Star Index: {i_star}")
    
#     # Display the last figure
#     if fig is not None:
#         plt.tight_layout()
#         plt.subplots_adjust(hspace=0.003, wspace=0.003)  # Adjust vertical and horizontal space
#         plt.show()



    
# artificial_stars(14, 28)


# Magnitud of each band for a single star from TRILEGAL
# mag_trilegal = [23.280, 20.776, 19.620, 18.729, 18.295, 18.082]
# M_input = dict(zip("ugrizy", mag_trilegal))
# cero_p = [27.615, 27.03, 28.38, 28.16, 27.85,  27.46, 26.68]
# mag_sat = [14.8, 14.7, 15.7, 15.8, 15.8, 15.3, 13.9]

