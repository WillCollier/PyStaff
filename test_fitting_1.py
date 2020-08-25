#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 19:21:53 2020

@author: kbmb68
"""
#! /usr/bin/env python
from __future__ import print_function

##############################################################################
import numpy as np
import emcee
import os, sys
import scipy.interpolate as si 
import lmfit as LM
import matplotlib.pyplot as plt

from pystaff.SpectralFitting import SpectralFit
from pystaff import SpectralFitting_functs as SF
import pandas as pd
from pandas import DataFrame, read_csv

#Likelihood function here. We could put it in the SpectraFitting class, but when 
#working with MPI on a cluster that would mean we'd need to pickle the fit_settings
#dictionary, which massively slows things down
def lnprob(T, theta, var_names, bounds, ret_specs=False):

    #Log prob function. T is an array of values

    
    assert len(T)==len(var_names), 'Error! The number of variables and walker position shapes are different'

    #Prior information comes from the parameter bounds now
    if np.any(T > bounds[:, 1]) or np.any(T < bounds[:, 0]):
        return -np.inf


    #make theta from the emcee walker positions
    for name, val in zip(var_names, T):
        theta[name].value = val

    if ret_specs==False:
        ll=SF.lnlike(theta, fit.fit_settings)
        return ll
    else:
        return SF.lnlike(theta, fit.fit_settings, ret_specs=True)


#Can select either Kroupa or Salpeter to use with the SSP models
element_imf='kroupa'

####################################################
"""
WCOLLIER Additions
"""

def get_galnoise(cut_wav=400):
    from astropy.io import fits
    
    galspec_dat = fits.open('data/galaxy_spectrum.fits')
    galspec = galspec_dat[0].data[cut_wav:]
    
    
    
    neighbour = np.zeros(len(galspec)-1)
    for i in range(len(galspec)-1):

        neighbour[i] += abs(galspec[i+1] - galspec[i])
    galnoise = np.nanmedian(neighbour)

    return galnoise


def get_galdata(cut_wav=400):
    from astropy.io import fits
    galspec_dat = fits.getdata('data/galaxy_spectrum.fits')[cut_wav:]
    wavelength = fits.getdata('data/galaxy_wavelength.fits')[cut_wav:]

    return wavelength, galspec_dat


def get_model_data(cut_wav=400):
    from astropy.io import fits
    from scipy.interpolate import interp1d

    """
    p0.26  13Gyr   No Alpha Enhancement
    """

    # library_selection = 'Mbi1.30Zp0.26T13.0000_iTp0.00_Ep0.00_linear_Sigma_314_z0.066.fits'
    model = fits.open('data/MILES_template.fits')
    mspec0  = model[0].data
    wavelength_MILES0 = (((np.arange(0,np.shape(mspec0)[0])*model[0].header['CDELT1'])+model[0].header['CRVAL1'])/1.066)*1.06625
    wavelength = fits.getdata('data/galaxy_wavelength.fits')[cut_wav:]

                        
    g = interp1d(wavelength_MILES0,mspec0)
    MILES_interped_0 = g(wavelength)
    
    
    
    return wavelength, MILES_interped_0




#%%
if 1==1:
    #Read in the data
    # datafile='data/example_spectrum.txt'
    
    # rshift = 1.0603
    # cut_wav = 400
    
    loops = input("How many loops: ")
    loops = int(loops)
    # lamdas, flux = get_galdata(cut_wav)
    # error_sig = get_galnoise(cut_wav)
    
    
    # lamdas, flux = get_model_data(cut_wav)
    # lamdas /= rshift
    # flux*=3e5
    # flux += np.random.normal(0,error_sig,len(flux))
    # 
    
    # flux = np.zeros(len(500))
    lamdas = np.arange(3800,5000,0.62)
    flux = np.ones(len(lamdas))
    errors = flux.copy() / 100.
    # errors = np.zeros(len(lamdas))+error_sig
    
    # instrumental_resolution = 
    
    
    # lamdas, flux, errors, instrumental_resolution=np.genfromtxt(datafile, unpack=True)
    
    # The instrumental resolution can be included if it's known. We need a value of sigma_inst in km/s for every pixel
    # Otherwise leave it as None
    instrumental_resolution=None
    
    # Sky Spectra
    # Give a list of 1D sky spectra to be scaled and subtracted during the fit
    # Otherwise leave sky as None
    skyspecs=None
    # ######################################################
    
    
    # ######################################################
    # Mask out regions that we don't want to fit, e.g. around telluric residuals, particularly nasty skylines, etc
    # THESE SHOULD BE OBSERVED WAVELENGTHS
    # A few examples of areas I often avoid due to skylines or telluric residuals
    # telluric_lam_1=np.array([[6862, 6952]])
    # bgd_emission=np.array([4430, 4450])/rshift
    bgd_emission=np.array([7155, 7175])
    
    telluric_lam_2=np.array([[7586, 7694]])
    # skylines=np.array([[8819, 8834], [8878.0, 8893], [8911, 8925], [8948, 8961]])
    
    masked_wavelengths=np.vstack([bgd_emission,telluric_lam_2]).reshape(-1, 1, 2)
    # masked_wavelengths=np.vstack([telluric_lam_1, telluric_lam_2, skylines]).reshape(-1, 1, 2)
    
    
    string_masked_wavelengths=["{} to {}".format(pair[0][0], pair[0][1]) for pair in masked_wavelengths]
    
    #Make a mask of pixels we don't want
    pixel_mask=np.ones_like(flux, dtype=bool)
    for array in masked_wavelengths:   
        m=SF.make_mask(lamdas, array)
        pixel_mask= m & pixel_mask
    
    #Now switch the weights of these pixels to 0
    pixel_weights=np.ones_like(flux)
    pixel_weights[~pixel_mask]=0.0
    
    
    #Wavelengths we'll fit between.
    #Split into 4 to make the multiplicative polynomials faster
    # fit_wavelengths=np.array([[4750, 5600], [5600, 6800], [6800, 8000], [8000,  9200]])
    # fit_wavelengths=np.array([[4085, 4400], [4400, 4650], [4650, 4900], [4900,  5180]])
    # fit_wavelengths=np.array([[4250, 4635],[4635, 5180]])/rshift
    fit_wavelengths=np.array([[3850, 4890]])
    
    # fit_wavelengths=np.array([[3800, 4890]])
    
    
    string_fit_wavelengths=["{} to {}".format(pair[0], pair[1]) for pair in fit_wavelengths]
    
    #FWHM.
    #This should be the FWHM in pixels of the instrument used to observe the spectrum.
    FWHM_gal=3.0
    
    base_template_location = '/Users/kbmb68/Documents/CvDmodels/vcj_dir'
    varelem_template_location = '/Users/kbmb68/Documents/CvDmodels/Atlas_dir'
    
    #Now set up the spectral fitting class
    print('Setting up the fit')
    fit0 = SpectralFit(lamdas, flux, errors, pixel_weights, fit_wavelengths, 
                       FWHM_gal, base_template_location=base_template_location, 
                       varelem_template_location=varelem_template_location, 
                       instrumental_resolution=instrumental_resolution, 
                       skyspecs=skyspecs, element_imf=element_imf)
    
    
    fit0.set_up_fit()

    thetas_in = []
    thetas_out, thetas_out1 = [], []

    # loops = input("How many loops: ")

    store_vals = {'sigma_in': np.zeros(loops), 'sigma_fit': np.zeros(loops), 'age1_in': np.zeros(loops),
                  'age1_fit': np.zeros(loops), 'age2_in': np.zeros(loops), 'age2_fit': np.zeros(loops),
                  'Z1_in': np.zeros(loops), 'Z1_fit': np.zeros(loops), 'Z2_in': np.zeros(loops),
                  'Z2_fit': np.zeros(loops), 'ratio_in': np.zeros(loops), 'ratio_fit': np.zeros(loops) }

    frame_storevals = pd.DataFrame(store_vals, columns=['sigma_in', 'sigma_fit', 'age1_in', 'age1_fit',
                                                        'age2_in', 'age2_fit', 'Z1_in', 'Z1_fit',
                                                        'Z2_in', 'Z2_fit', 'ratio_in', 'ratio_fit'])
    # sigma_in = np.zeros(loops)
    # sigma_fit = np.zeros(loops)
    # age1_in = np.zeros(loops)
    # age1_fit = np.zeros(loops)
    # age2_in = np.zeros(loops)
    # age2_fit = np.zeros(loops)
    # Z1_in = np.zeros(loops)
    # Z1_fit = np.zeros(loops)
    # Z2_in = np.zeros(loops)
    # Z2_fit = np.zeros(loops)
    # ratio_in = np.zeros(loops)
    # ratio_fit = np.zeros(loops)
    
    for i in range(loops):
        # loops = 1
        
        theta=LM.Parameters()
        #LOSVD parameters
        theta.add('Vel', value=1600, min=-1000.0, max=10000.0)
        theta.add('sigma', value=300.0 + (np.random.random() - 0.5)*100, min=10.0, max=500.0)

        #Abundance of Na. Treat this separately, since it can vary up to +1.0 dex
        theta.add('Na', value=0.5 + (np.random.random() - 0.5)*0.3, min=-0.45, max=1.0, vary=True)

        #Abundance of elements which can vary positively and negatively
        theta.add('Ca', value=0.0,  min=-0.45, max=0.45, vary=True)
        theta.add('Fe', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('C', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('N', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('Ti', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('Mg', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('Si', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('Ba', value=0.0, min=-0.45, max=0.45, vary=True)

        #Abundance of elements which can only vary above 0.0
        theta.add('as_Fe', value=0.0+ (np.random.random())*0.4, min=0.0, max=0.45, vary=True)
        theta.add('Cr', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Mn', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Ni', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Co', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Eu', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Sr', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('K', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('V', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Cu', value=0.0, min=0.0, max=0.45, vary=False)

        #Emission line kinematics
        #Each line is fixed to the same velocity and sigma
        theta.add('Vel_em', value=1600, min=0.0, max=10000)
        theta.add('sig_em', value=200.0, min=10.0, max=500.0)

        #Emission line strengths
        #These are log flux- they get exponentiated in the likelihood function
        theta.add('Ha', value=1.0, min=-10.0, max=10.0, vary=False)
        theta.add('Hb', value=0.3, min=-10.0, max=10.0, vary=True)
        theta.add('NII', value=0.5, min=-10.0, max=10.0, vary=False)
        theta.add('SII_6716', value=0.5, min=-10.0, max=10.0, vary=False)
        theta.add('SII_6731', value=0.5, min=-10.0, max=10.0, vary=False)
        theta.add('OIII', value=-2.0, min=-10.0, max=10.0, vary=False)
        theta.add('OI', value=-2.0, min=-10.0, max=10.0, vary=False)

        #Base population parameters
        #Age, Metallicity, and the two IMF slopes
        theta.add('age', value=10.0 + (np.random.random() - 0.5)*3 , min=2.0, max=14.0,vary=True)
        theta.add('Z', value=-0.4 + (np.random.random() - 0.5)*0.5, min=-1.0, max=0.2,vary=True)
        theta.add('imf_x1', value=1.3, min=0.5, max=3.5, vary=False)
        theta.add('imf_x2', value=2.35, min=0.5, max=3.5, vary=False)

        #Strengths of skylines
        theta.add('O2_Scale', value=0.0, min=-100000000, max=100000000, vary=False)
        theta.add('sky_Scale', value=0.0, min=-100000000, max=100000000, vary=False)
        theta.add('OH_Scale', value=0.0, min=-100000000, max=100000000, vary=False)
        theta.add('NaD_sky_scale', value=0.0, min=-100000000, max=100000000, vary=False)

        #Option to rescale the error bars up or down
        theta.add('ln_f', value=0.0, min=-5.0, max=5.0, vary=False)



        """
        WOLIER Addition
        """

        # theta.add('Vel_2', value=1600, min=-1000.0, max=10000.0)
        # theta.add('sigma_fit', value=314.0, min=10.0, max=500.0)

        #Abundance of Na. Treat this separately, since it can vary up to +1.0 dex
        theta.add('Na_2', value=0.1+ (np.random.random() - 0.5)*0.4, min=-0.45, max=1.0, vary=True)

        #Abundance of elements which can vary positively and negatively
        theta.add('Ca_2', value=0.0,  min=-0.45, max=0.45, vary=False)
        theta.add('Fe_2', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('C_2', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('N_2', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('Ti_2', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('Mg_2', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('Si_2', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('Ba_2', value=0.0, min=-0.45, max=0.45, vary=False)

        #Abundance of elements which can only vary above 0.0
        theta.add('as_Fe_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Cr_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Mn_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Ni_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Co_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Eu_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Sr_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('K_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('V_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Cu_2', value=0.0, min=0.0, max=0.45, vary=False)

        #Emission line kinematics
        #Each line is fixed to the same velocity and sigma
        theta.add('Vel_em_2', value=1600, min=0.0, max=10000)
        theta.add('sig_em_2', value=200.0, min=10.0, max=500.0)

        #Emission line strengths
        #These are log flux- they get exponentiated in the likelihood function
        theta.add('Ha_2', value=1.0, min=-10.0, max=10.0, vary=True)
        theta.add('Hb_2', value=0.3, min=-10.0, max=10.0, vary=False)
        theta.add('NII_2', value=0.5, min=-10.0, max=10.0, vary=False)
        theta.add('SII_6716_2', value=0.5, min=-10.0, max=10.0, vary=False)
        theta.add('SII_6731_2', value=0.5, min=-10.0, max=10.0, vary=False)
        theta.add('OIII_2', value=-2.0, min=-10.0, max=10.0, vary=False)
        theta.add('OI_2', value=-2.0, min=-10.0, max=10.0, vary=False)

        #Base population parameters
        #Age, Metallicity, and the two IMF slopes
        theta.add('age_2', value=2.1 + (np.random.random() - 0.5)*1.6, min=0.5, max=4.0,vary=True)
        theta.add('Z_2', value=-0.4+ (np.random.random() - 0.5)*0.55, min=-1.0, max=0.2,vary=True)
        theta.add('imf_x1_2', value=1.3, min=0.5, max=3.5, vary=False)
        theta.add('imf_x2_2', value=2.35, min=0.5, max=3.5, vary=False)

        #Strengths of skylines
        # theta.add('O2_Scale_2', value=0.0, min=-100000000, max=100000000, vary=False)
        # theta.add('sky_Scale_2', value=0.0, min=-100000000, max=100000000, vary=False)
        # theta.add('OH_Scale_2', value=0.0, min=-100000000, max=100000000, vary=False)
        # theta.add('NaD_sky_scale_2', value=0.0, min=-100000000, max=100000000, vary=False)


        # Option to rescale the error bars up or down
        theta.add('ratio', value=0.3+ (np.random.random() - 0.5)*0.29, min=0.0, max=0.9999999, vary=True)
        
        theta0 = theta.copy()
            
        thetas_in.append(theta0)
    
    
        # SF.plot_fit(theta0, fit0.fit_settings)
        logLams, template0, model0 = SF.get_best_fit_template(theta0, fit0.fit_settings,convolve = True)
        
        lamdas_out = np.exp(logLams)
        
        f = si.interp1d(lamdas_out, template0)
        template = f(lamdas[:-1])
        
        # model,loglam = np.genfromtxt('/Users/kbmb68/Documents/MNELLS/MUSE_obs/LSQ13cwp/FORS2/template.txt')
        
        # plt.figure()
        # plt.plot(lamdas_out,template0)
        # plt.plot(lamdas[:-1],template)
        # # plt.plot(np.exp(fit0.fit_settings['log_lam_template']),model)
        # plt.show()
 
        # plt.figure()
        # plt.plot(lamdas,np.exp(logLams))
        # plt.show()
        
    
##%%
# if 1==1:

        theta = LM.Parameters()
        # LOSVD parameters
        theta.add('Vel', value=1600, min=-1000.0, max=10000.0)
        theta.add('sigma', value=250.0, min=10.0, max=500.0)

        # Abundance of Na. Treat this separately, since it can vary up to +1.0 dex
        theta.add('Na', value=0.1, min=-0.45, max=1.0, vary=True)

        # Abundance of elements which can vary positively and negatively
        theta.add('Ca', value=0.1, min=-0.45, max=0.45, vary=False)
        theta.add('Fe', value=0.1, min=-0.45, max=0.45, vary=False)
        theta.add('C', value=-0.1, min=-0.45, max=0.45, vary=False)
        theta.add('N', value=-0.1, min=-0.45, max=0.45, vary=False)
        theta.add('Ti', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('Mg', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('Si', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('Ba', value=0.0, min=-0.45, max=0.45, vary=False)

        # Abundance of elements which can only vary above 0.0
        theta.add('as_Fe', value=0.20, min=0.0, max=0.45, vary=True)
        theta.add('Cr', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Mn', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Ni', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Co', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Eu', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Sr', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('K', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('V', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Cu', value=0.0, min=0.0, max=0.45, vary=False)

        # Emission line kinematics
        # Each line is fixed to the same velocity and sigma
        theta.add('Vel_em', value=1600, min=0.0, max=10000)
        theta.add('sig_em', value=200.0, min=10.0, max=500.0)

        # Emission line strengths
        # These are log flux- they get exponentiated in the likelihood function
        theta.add('Ha', value=1.0, min=-10.0, max=10.0, vary=False)
        theta.add('Hb', value=0.3, min=-10.0, max=10.0, vary=True)
        theta.add('NII', value=0.5, min=-10.0, max=10.0, vary=False)
        theta.add('SII_6716', value=0.5, min=-10.0, max=10.0, vary=False)
        theta.add('SII_6731', value=0.5, min=-10.0, max=10.0, vary=False)
        theta.add('OIII', value=-2.0, min=-10.0, max=10.0, vary=False)
        theta.add('OI', value=-2.0, min=-10.0, max=10.0, vary=False)

        # Base population parameters
        # Age, Metallicity, and the two IMF slopes
        theta.add('age', value=8.0, min=2.0, max=14.0, vary=True)
        theta.add('Z', value=-0.1, min=-1.0, max=0.2, vary=True)
        theta.add('imf_x1', value=1.3, min=0.5, max=3.5, vary=False)
        theta.add('imf_x2', value=2.35, min=0.5, max=3.5, vary=False)

        # Strengths of skylines
        theta.add('O2_Scale', value=0.0, min=-100000000, max=100000000, vary=False)
        theta.add('sky_Scale', value=0.0, min=-100000000, max=100000000, vary=False)
        theta.add('OH_Scale', value=0.0, min=-100000000, max=100000000, vary=False)
        theta.add('NaD_sky_scale', value=0.0, min=-100000000, max=100000000, vary=False)

        # Option to rescale the error bars up or down
        theta.add('ln_f', value=0.0, min=-5.0, max=5.0, vary=False)

        """
        WOLIER Addition
        """

        # theta.add('Vel_2', value=1600, min=-1000.0, max=10000.0)
        # theta.add('sigma_fit', value=314.0, min=10.0, max=500.0)

        # Abundance of Na. Treat this separately, since it can vary up to +1.0 dex
        theta.add('Na_2', value=0.2, min=-0.45, max=1.0, vary=False)

        # Abundance of elements which can vary positively and negatively
        theta.add('Ca_2', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('Fe_2', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('C_2', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('N_2', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('Ti_2', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('Mg_2', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('Si_2', value=0.0, min=-0.45, max=0.45, vary=False)
        theta.add('Ba_2', value=0.0, min=-0.45, max=0.45, vary=False)

        # Abundance of elements which can only vary above 0.0
        theta.add('as_Fe_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Cr_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Mn_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Ni_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Co_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Eu_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Sr_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('K_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('V_2', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Cu_2', value=0.0, min=0.0, max=0.45, vary=False)

        # Emission line kinematics
        # Each line is fixed to the same velocity and sigma
        theta.add('Vel_em_2', value=1600, min=0.0, max=10000)
        theta.add('sig_em_2', value=200.0, min=10.0, max=500.0)

        # Emission line strengths
        # These are log flux- they get exponentiated in the likelihood function
        theta.add('Ha_2', value=1.0, min=-10.0, max=10.0, vary=False)
        theta.add('Hb_2', value=0.3, min=-10.0, max=10.0, vary=True)
        theta.add('NII_2', value=0.5, min=-10.0, max=10.0, vary=False)
        theta.add('SII_6716_2', value=0.5, min=-10.0, max=10.0, vary=False)
        theta.add('SII_6731_2', value=0.5, min=-10.0, max=10.0, vary=False)
        theta.add('OIII_2', value=-2.0, min=-10.0, max=10.0, vary=False)
        theta.add('OI_2', value=-2.0, min=-10.0, max=10.0, vary=False)

        # Base population parameters
        # Age, Metallicity, and the two IMF slopes
        theta.add('age_2', value=1.40, min=0.5, max=4.0, vary=True)
        theta.add('Z_2', value=0.0, min=-1.0, max=0.2, vary=True)
        theta.add('imf_x1_2', value=1.3, min=0.5, max=3.5, vary=False)
        theta.add('imf_x2_2', value=2.35, min=0.5, max=3.5, vary=False)

        # Strengths of skylines
        # theta.add('O2_Scale_2', value=0.0, min=-100000000, max=100000000, vary=False)
        # theta.add('sky_Scale_2', value=0.0, min=-100000000, max=100000000, vary=False)
        # theta.add('OH_Scale_2', value=0.0, min=-100000000, max=100000000, vary=False)
        # theta.add('NaD_sky_scale_2', value=0.0, min=-100000000, max=100000000, vary=False)

        # Option to rescale the error bars up or down
        theta.add('ratio', value=0.6, min=0.0, max=0.9999999, vary=True)



        print('Setting up the fit')
        # fit=SpectralFit(lamdas, model, model/100., pixel_weights, fit_wavelengths, FWHM_gal, instrumental_resolution=instrumental_resolution, skyspecs=skyspecs, element_imf=element_imf)

        # fit=SpectralFit(lamdas[:-1], template, template/100., pixel_weights, fit_wavelengths, FWHM_gal, instrumental_resolution=instrumental_resolution, skyspecs=skyspecs, element_imf=element_imf)
        lamdas_in = lamdas[:-1]
        
        
        # #Now switch the weights of these pixels to 0
        # pixel_weights=np.ones_like(template)
        # pixel_weights[~pixel_mask]=0.0
        
        pixel_weights_in = pixel_weights[:-1]
        fit=SpectralFit(lamdas_in, template, template/100.,
                    pixel_weights_in, fit_wavelengths, FWHM_gal, 
                    base_template_location=base_template_location,
                    varelem_template_location=varelem_template_location, 
                    instrumental_resolution=instrumental_resolution, 
                    skyspecs=skyspecs, element_imf=element_imf)

        fit.set_up_fit()


        variables=[thing for thing in theta if theta[thing].vary]
        ndim=len(variables)
        #Vice versa, plus add in the fixed value
        fixed=[ "{}={},".format(thing, theta[thing].value) for thing in theta if not theta[thing].vary]


        stds=[]
        n_general=1
        n_positive=1
        n_emission_lines=1

        n_general_2=0
        n_positive_2=0
        n_emission_lines_2=1


        #Add in all these standard deviations
        #Kinematic parameters
        stds.extend([100.0, 50.0])
        #General parameters
        stds.extend([0.1]*n_general)
        #Positive parameters
        stds.extend([0.1]*n_positive)
        #Emission lines
        stds.extend([100.0, 50.0])
        stds.extend([1.0]*n_emission_lines)
        #Age
        stds.extend([7.0])
        #Z, imf1, imf2
        # stds.extend([0.1, 0.1, 0.1])
        stds.extend([0.1])
        #Sky
        #stds.extend([100.0,  100.0,  100.0, 100.0])
        #ln_f
        # stds.extend([0.5])


        # stds.extend([100.0, 50.0])
        #General parameters
        stds.extend([0.1]*n_general_2)
        #Positive parameters
        stds.extend([0.1]*n_positive_2)
        #Emission lines
        stds.extend([100.0, 50.0])
        stds.extend([1.0]*n_emission_lines_2)
        #Age
        stds.extend([7.0])
        #Z, imf1, imf2
        # stds.extend([0.1, 0.1, 0.1])
        stds.extend([0.1])

        # ratio of two components
        stds.extend([0.5])




        stds=np.array(stds)


        assert len(stds)==len(variables), "You must have the same number of dimensions for the Gaussian ball as variables!"

        nwalkers = 70
        nsteps = 500

        #Now get the starting values for each parameter, as well as the prior bounds
        start_values, bounds=SF.get_start_vals_and_bounds(theta)
        p0=SF.get_starting_positions_for_walkers(start_values, stds, nwalkers, bounds)
        ###################################################################################################


        # ###################################################################################################
        #Do the sampling
        #This may take a while!



        print("Running the fitting with {} walkers for {} steps".format(nwalkers, nsteps))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[theta, variables, bounds], pool=None)
        result = sampler.run_mcmc(p0, nsteps, progress=True)


        ####################################################################################################

        #get rid of the burn-in
        burnin=np.array(nsteps-5000).clip(0)
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
        print("\tDone")

        log_prob_samples2 = sampler.get_log_prob(discard=burnin, flat=True)
        log_prior_samples = sampler.get_blobs(discard=burnin, flat=True)


        #Get the 16th, 50th and 84th percentiles of the marginalised posteriors for each parameter
        best_results = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))))
        #If your posterior surface isn't a nice symmetric Gaussian, then the vector of median values for each parameter (as we're doing here)
        #could very well correspond to an unlikely area of parameter space and you'll need to do something different to this!


        for v, r in zip(variables, best_results):
            print("{}: {:.3f} +{:.2f}/-{:.2f}".format(v, r[0], r[1], r[2]))

        print('\n')





        for v, r in zip(variables[2:10], best_results[2:10]):
            print("{}: {:.3f} +{:.2f}/-{:.2f}".format(v, np.log10(10**r[0]/10**best_results[4,0]),
                                                      np.log10(10**r[1]/10**best_results[4,1]),
                                                      np.log10(10**r[2]/10**best_results[4,2])))
        print('\n')

        for v, r in zip(variables[17:-7], best_results[17:-7]):
            print("{}: {:.3f} +{:.2f}/-{:.2f}".format(v, np.log10(10**r[0]/10**best_results[4,0]),
                                                      np.log10(10**r[1]/10**best_results[4,1]),
                                                      np.log10(10**r[2]/10**best_results[4,2])))



        print('\n Second pop compared to self \n')

        for v, r in zip(variables[17:-7], best_results[17:-7]):
            print("{}: {:.3f} +{:.2f}/-{:.2f}".format(v, np.log10(10**r[0]/10**best_results[19,0]),
                                                      np.log10(10**r[1]/10**best_results[19,1]),
                                                      np.log10(10**r[2]/10**best_results[19,2])))

        print('\n')

        results_theta=LM.Parameters()
        thetas_out1.append(results_theta)


        for v, r in zip(variables, best_results):
            # print(v, r)
            results_theta.add('{}'.format(v), value=r[0], vary=False)
        #and include the things we kept fixed originally too:
        [results_theta.add('{}'.format(thing), value=theta[thing].value, vary=False) for thing in theta if not theta[thing].vary]


        # SF.plot_fit(results_theta, fit.fit_settings)

        frame_storevals['sigma_in'][i] += theta0['sigma'].value
        frame_storevals['sigma_fit'][i] += theta['sigma'].value
        frame_storevals['age1_in'][i] += theta0['age'].value
        frame_storevals['age1_fit'][i] += theta['age'].value
        frame_storevals['age2_in'][i] += theta0['age_2'].value
        frame_storevals['age2_fit'][i] += theta['age_2'].value
        frame_storevals['Z1_in'][i] += theta0['Z'].value
        frame_storevals['Z1_fit'][i] += theta['Z'].value
        frame_storevals['Z2_in'][i] += theta0['Z_2'].value
        frame_storevals['Z2_fit'][i] += theta['Z_2'].value
        frame_storevals['ratio_in'][i] += theta0['ratio'].value
        frame_storevals['ratio_fit'][i] += theta['ratio'].value
        
        
        import corner
        corner.corner(samples, labels=variables)    

    # SF.get_best_fit_template(results_theta, fit.fit_settings)
    # model0 = np.genfromtxt('/Users/kbmb68/Documents/MNELLS/MUSE_obs/LSQ13cwp/FORS2/template.txt')
    
    # plt.figure()
    # # plt.plot(lamdas,np.exp(fit.fit_settings['log_galaxy'])-1,c='b')
    # plt.plot(lamdas[:-1],template,c='r')
    # plt.plot(np.exp(model0[1]),model0[0],c='g')
    # plt.show()


def plot(x, y, title, xlab, ylab, df):
    # plt.figure()
    df.plot.scatter(x=x, y=y)
    # plt.ylabel(ylab)
    # plt.xlabel(xlab)
    plt.title(title)
    # plt.savefig('plots_out/{}vs{}.png'.format(x,y))
    # plt.show()

    return

titles = np.array(['sigma', 'age1', 'age2', 'Z1', 'Z2', 'ratio'])
xs = np.array(['sigma_in', 'age1_in', 'age2_in', 'Z1_in', 'Z2_in', 'ratio_in'])
ys = np.array(['sigma_fit','age1_fit', 'age2_fit', 'Z1_fit', 'Z2_fit', 'ratio_fit'])
xlabel = 'input'
ylabel = 'fitted'

for i in range(len(titles)):
    plot(xs[i], ys[i], titles[i], xlabel, ylabel, frame_storevals)

# plt.figure()
# plt.scatter(age1_in, age1_fit)
# plt.title('Sigma')
# plt.show()
#
# plt.figure()
# plt.scatter(age2_in, age2_fit)
# plt.show()
#
# plt.figure()
# plt.scatter(Z1_in, Z1_fit)
# plt.show()
#
# plt.figure()
# plt.scatter(Z2_in, Z2_fit)
# plt.show()
#
# plt.figure()
# plt.scatter(ratio_in, ratio_fit)
# plt.show()

if 1==0:
        theta = LM.Parameters()
        # LOSVD parameters
        theta.add('Vel', value=1600, min=-1000.0, max=10000.0)
        theta.add('sigma', value=300.0, min=10.0, max=500.0)

        # Abundance of Na. Treat this separately, since it can vary up to +1.0 dex
        theta.add('Na', value=0.5, min=-0.45, max=1.0, vary=True)

        # Abundance of elements which can vary positively and negatively
        theta.add('Ca', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('Fe', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('C', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('N', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('Ti', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('Mg', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('Si', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('Ba', value=0.0, min=-0.45, max=0.45, vary=True)

        # Abundance of elements which can only vary above 0.0
        theta.add('as_Fe', value=0.0, min=0.0, max=0.45, vary=True)
        theta.add('Cr', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Mn', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Ni', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Co', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Eu', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Sr', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('K', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('V', value=0.0, min=0.0, max=0.45, vary=False)
        theta.add('Cu', value=0.0, min=0.0, max=0.45, vary=False)

        # Emission line kinematics
        # Each line is fixed to the same velocity and sigma
        theta.add('Vel_em', value=1600, min=0.0, max=10000)
        theta.add('sig_em', value=200.0, min=10.0, max=500.0)

        # Emission line strengths
        # These are log flux- they get exponentiated in the likelihood function
        theta.add('Ha', value=1.0, min=-10.0, max=10.0, vary=False)
        theta.add('Hb', value=0.3, min=-10.0, max=10.0, vary=True)
        theta.add('NII', value=0.5, min=-10.0, max=10.0, vary=False)
        theta.add('SII_6716', value=0.5, min=-10.0, max=10.0, vary=False)
        theta.add('SII_6731', value=0.5, min=-10.0, max=10.0, vary=False)
        theta.add('OIII', value=-2.0, min=-10.0, max=10.0, vary=False)
        theta.add('OI', value=-2.0, min=-10.0, max=10.0, vary=False)

        # Base population parameters
        # Age, Metallicity, and the two IMF slopes
        theta.add('age', value=12.0, min=2.0, max=14.0, vary=True)
        theta.add('Z', value=0.0, min=-1.0, max=0.2, vary=True)
        theta.add('imf_x1', value=1.3, min=0.5, max=3.5, vary=False)
        theta.add('imf_x2', value=2.35, min=0.5, max=3.5, vary=False)

        # Strengths of skylines
        theta.add('O2_Scale', value=0.0, min=-100000000, max=100000000, vary=False)
        theta.add('sky_Scale', value=0.0, min=-100000000, max=100000000, vary=False)
        theta.add('OH_Scale', value=0.0, min=-100000000, max=100000000, vary=False)
        theta.add('NaD_sky_scale', value=0.0, min=-100000000, max=100000000, vary=False)

        # Option to rescale the error bars up or down
        theta.add('ln_f', value=0.0, min=-5.0, max=5.0, vary=False)

        fit = SpectralFit(lamdas[:-1], template, template / 100., pixel_weights, fit_wavelengths, FWHM_gal,
                          instrumental_resolution=instrumental_resolution, skyspecs=skyspecs, element_imf=element_imf)
        print("Setting up fit")
        fit.set_up_fit()

        variables = [thing for thing in theta if theta[thing].vary]
        ndim = len(variables)
        # Vice versa, plus add in the fixed value
        fixed = ["{}={},".format(thing, theta[thing].value) for thing in theta if not theta[thing].vary]

        stds = []
        n_general = 1
        n_positive = 9
        n_emission_lines = 1

        n_general_2 = 1
        n_positive_2 = 9
        n_emission_lines_2 = 1

        # Add in all these standard deviations
        # Kinematic parameters
        stds.extend([100.0, 50.0])
        # General parameters
        stds.extend([0.1] * n_general)
        # Positive parameters
        stds.extend([0.1] * n_positive)
        # Emission lines
        stds.extend([100.0, 50.0])
        stds.extend([1.0] * n_emission_lines)
        # Age
        stds.extend([7.0])
        # Z, imf1, imf2
        # stds.extend([0.1, 0.1, 0.1])
        stds.extend([0.1])
        # Sky
        # stds.extend([100.0,  100.0,  100.0, 100.0])
        # ln_f
        # stds.extend([0.5])

        # stds.extend([100.0, 50.0])
        # General parameters
        # stds.extend([0.1] * n_general_2)
        # Positive parameters
        # stds.extend([0.1] * n_positive_2)
        # Emission lines
        # stds.extend([100.0, 50.0])
        # stds.extend([1.0] * n_emission_lines_2)
        # Age
        # stds.extend([7.0])
        # Z, imf1, imf2
        # stds.extend([0.1, 0.1, 0.1])
        # stds.extend([0.1])
        #
        # ratio of two components
        # stds.extend([0.5])

        stds = np.array(stds)

        assert len(stds) == len(
            variables), "You must have the same number of dimensions for the Gaussian ball as variables!"

        nwalkers = 70
        nsteps = 500

        # Now get the starting values for each parameter, as well as the prior bounds
        start_values, bounds = SF.get_start_vals_and_bounds(theta)
        p0 = SF.get_starting_positions_for_walkers(start_values, stds, nwalkers, bounds)
        ###################################################################################################

        # ###################################################################################################
        # Do the sampling
        # This may take a while!

        print("Running the fitting with {} walkers for {} steps".format(nwalkers, nsteps))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[theta, variables, bounds], pool=None)
        result = sampler.run_mcmc(p0, nsteps, progress=True)

        ####################################################################################################

        # get rid of the burn-in
        burnin = np.array(nsteps - 5000).clip(0)
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
        print("\tDone")

        log_prob_samples2 = sampler.get_log_prob(discard=burnin, flat=True)
        log_prior_samples = sampler.get_blobs(discard=burnin, flat=True)

        # Get the 16th, 50th and 84th percentiles of the marginalised posteriors for each parameter
        best_results = np.array(
            list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))))
        # If your posterior surface isn't a nice symmetric Gaussian, then the vector of median values for each parameter (as we're doing here)
        # could very well correspond to an unlikely area of parameter space and you'll need to do something different to this!

        for v, r in zip(variables, best_results):
            print("{}: {:.3f} +{:.2f}/-{:.2f}".format(v, r[0], r[1], r[2]))

        print('\n')

        for v, r in zip(variables[2:10], best_results[2:10]):
            print("{}: {:.3f} +{:.2f}/-{:.2f}".format(v, np.log10(10 ** r[0] / 10 ** best_results[4, 0]),
                                                      np.log10(10 ** r[1] / 10 ** best_results[4, 1]),
                                                      np.log10(10 ** r[2] / 10 ** best_results[4, 2])))
        print('\n')

        for v, r in zip(variables[17:-7], best_results[17:-7]):
            print("{}: {:.3f} +{:.2f}/-{:.2f}".format(v, np.log10(10 ** r[0] / 10 ** best_results[4, 0]),
                                                      np.log10(10 ** r[1] / 10 ** best_results[4, 1]),
                                                      np.log10(10 ** r[2] / 10 ** best_results[4, 2])))

        print('\n Second pop compared to self \n')

        for v, r in zip(variables[17:-7], best_results[17:-7]):
            print("{}: {:.3f} +{:.2f}/-{:.2f}".format(v, np.log10(10 ** r[0] / 10 ** best_results[19, 0]),
                                                      np.log10(10 ** r[1] / 10 ** best_results[19, 1]),
                                                      np.log10(10 ** r[2] / 10 ** best_results[19, 2])))

        print('\n')

        results_theta = LM.Parameters()

        thetas_out.append(results_theta)

        for v, r in zip(variables, best_results):
            # print(v, r)
            results_theta.add('{}'.format(v), value=r[0], vary=False)
        # and include the things we kept fixed originally too:
        [results_theta.add('{}'.format(thing), value=theta[thing].value, vary=False) for thing in theta if
         not theta[thing].vary]

        SF.plot_fit(results_theta, fit.fit_settings)