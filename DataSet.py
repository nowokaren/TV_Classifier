# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:48:41 2023

@author: KarenNowo
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:19:43 2023

@author: KarenNowo
"""


from pathlib import Path
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

class DataSet:
    def __init__(self, DataSet_name):
        self.name = DataSet_name
        self.models = [i for i in os.listdir(Path(DataSet_name)) if not "." in i]
        self.model_format = os.path.commonprefix(self.models)
    
    def size(self, plot = "bar"):
        sizes = {}
        for mod in tqdm(self.models, desc=f"Computing amount of light curves of {len(self.models)} models of {self.name} dataset"):
            sizes[mod]=len(os.listdir(Path(self.name, mod)))
        self.models_size = sizes
        if plot == None:
            pass
        elif plot == "bar":
            fig = plt.figure(figsize=(10, round(len(self.models)/2)))
            plt.title(self.name)
            plt.xlabel("# Light curves")
            plt.ylabel("Model name")
            plt.barh([i.replace(self.model_format, "") for i in self.models], self.models_size.values(), color="LightBlue")
            for index, value in enumerate(self.models_size.values()):
                plt.text(value, index-0.3, str(value))
            return fig



class Model(DataSet):
    def __init__(self, Model_path, DataSet_name, bands = None):
        super().__init__(DataSet_name)  # Call parent class's __init__ method
        self.path = DataSet_name+"/" + Model_path
        self.name = Model_path.replace(self.model_format, "")
        self.lc = os.listdir(self.path)
        self.lc_format = os.path.commonprefix(self.lc)
        if bands!= None:
            for folder in os.listdir():
                if self.name in folder:
                    bands = self.name[-1]
        self.bands = bands

# class Event(Model):

class LightCurve(Model):
    def __init__(self, LC_index, model_instance):
        self.path = []
        self.curve = {}
        self.bands = model_instance.bands
        for band in self.bands:
            self.path.append(model_instance.path + "/" + model_instance.lc[LC_index])
            self.curve[band] = np.loadtxt(self.path).T
        
    
    def plot(self, color = True):
        mjd, mag, mag_err = self.curve
        if not color:
            plt.errorbar(mjd, mag, mag_err)
        else:
            plt.errorbar(mjd, mag, mag_err, color = color)
        plt.xlabel("MJD")
        plt.ylabel("Magnitude")
    
    def likelihood(self):
        # Parámetros del modelo (media y desviación estándar)
        mu = np.mean(self.curve[1])
        sigma = np.std(self.curve[1])
        return np.prod(norm.pdf(self.curve[1], loc=mu, scale=sigma))
    
    def chi2_cte(self, cte="mean"):
        if cte == "mean":
            cte = [np.mean(self.)
        chi_squared = np.sum(((self.magnitude - model_magnitude) / self.magnitude_err) ** 2)
        return chi_squared

    def test_cte(self, n_sig=3, n_pts=4):
        '''
        n_sig: número de sigmas critico
        n_pts: cantidad de puntos que están a n_sig*sigma
        '''
        mean = np.mean(self.curve[1])
        pts = 0
        for mag, mag_err in zip(self.curve[1], self.curve[2]):
            if abs(mag-mean)>n_sig*mag_err:
                pts += 1
        if pts>n_pts+1:
            return True
        else:
            return False
        

        
# # Usage
elasticc = DataSet("ELASTICC")
uLens = Model("ELASTICC_TRAIN_uLens-Single_PyLIMA", elasticc.name)
train = DataSet("training_gal")
a = Model(train.models[2], train.name)

lcc = LightCurve(3, a)
