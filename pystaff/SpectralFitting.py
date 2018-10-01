
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np 
import scipy.constants as const

#Relative imports are such a headache
try:
    from . import SpectralFitting_functs as SF
    from . import CvD_SSP_tools as CvDTools
#autodoc doesn't like the above imports, so add this to make the documentation work
except ValueError:
    import SpectralFitting_functs as SF
    import CvD_SSP_tools as CvDTools


#ToDo:

#Mask emission lines from polynomial fit



class SpectralFit(object):

    c_light=const.c/1000.0


    def __init__(self, lamdas, flux, noise, pixel_weights, fit_wavelengths, FWHM_gal, skyspecs=None, element_imf='kroupa', instrumental_resolution=None, vac_or_air='vac'):

        if not np.unique(np.array([flux.size, lamdas.size, noise.size])).size == 1:
            raise ValueError('LAMDAS, FLUX and NOISE must be the same length!')

        if not np.unique(np.array([flux.shape, lamdas.shape, noise.shape])).size ==1:
            raise ValueError('LAMDAS, FLUX and NOISE must be the same shape!')

        if not np.all(np.array([flux.ndim, lamdas.ndim, noise.ndim]) ==1):
            raise ValueError('LAMDAS, FLUX and NOISE must all be 1D arrays')

        #Check if spacing is uniform. Round to avoid floating point issues
        if not len(np.unique(np.ediff1d(lamdas.round(decimals=8))))==1:
            raise ValueError('LAMDAS must be on a uniform wavelength grid, with no jumps or changes in wavelength spacing!')

        if instrumental_resolution is not None:
            if not np.unique(np.array([flux.size, lamdas.size, noise.size, instrumental_resolution.size])).size == 1:
                raise ValueError('If INSTRUMENTAL_RESOLUTION is given, it must be the same length as FLUX, LAMDAS and NOISE')


        self.lin_lam=lamdas
        self.lin_flux=flux
        self.lin_skyspecs=skyspecs
        self.lin_noise=noise
        self.lin_weights=pixel_weights
        self.fit_wavelengths=fit_wavelengths
        self.element_imf=element_imf
        self.FWHM_gal=FWHM_gal
        self.instrumental_resolution=instrumental_resolution
        self.vac_or_air=vac_or_air


    def set_up_fit(self):

        #rebin the spectra into log_lamda rather than lamda
        self.rebin_spectra()

        #The elements which can change in the CvD12 models
        positive_only_elems=['as/Fe+', 'Cr+','Mn+','Ni+','Co+','Eu+','Sr+','K+','V+','Cu+']
        Na_elem=['Na']
        normal_elems=['Ca', 'Fe', 'C', 'N', 'Ti', 'Mg', 'Si', 'Ba']
        self.elements_to_fit=(positive_only_elems, Na_elem, normal_elems)

        #Make sure we have a small amount of padding, so the templates are slightly longer than the models
        pad=500.0
        self.lam_range_temp = np.array([self.lam_range_gal[0]-pad, self.lam_range_gal[1]+pad])

        #Clip the lam_range_temp to be between the min and max of the models, just in case it isn't
        if np.any(self.lam_range_temp<3501) or np.any(self.lam_range_temp>24997.58):
            raise ValueError('The templates only extend from 3501 to 24997.58A! Lam_range_temp is {}'.format(self.lam_range_temp))


        #Prepare the interpolators
        self.prepare_CVD2_interpolators()

        self.get_emission_lines(self.vac_or_air)

        self.dv = SpectralFit.c_light*np.log(self.lam_range_temp[0]/self.lam_range_gal[0])

        self.fit_settings={'log_galaxy':self.log_galaxy, 
                            'log_noise':self.log_noise, 
                            'log_skyspecs':self.log_skyspecs, 
                            'log_weights':self.log_weights,
                            'emission_lines':self.emission_lines, 
                            'velscale':self.velscale, 
                            'goodpixels':self.goodpixels, 
                            'dv':self.dv, 
                            'linear_interp':self.linear_interp, 
                            'correction_interps':self.correction_interps, 
                            'log_lam_template':self.log_lam_template, 
                            'log_lam':self.log_lam, 
                            'fit_wavelengths':self.fit_wavelengths, 
                            'c_light':SpectralFit.c_light,
                            'instrumental_resolution':self.log_instrumental_resolution}



    def rebin_spectra(self):

        loggalaxy, lognoise, log_skyspecs, log_inst_res, logweights, velscale, goodpixels, lam_range_gal, logLam = SF.rebin_spectrum(self.lin_lam, self.lin_flux, self.lin_noise, self.lin_weights, instrumental_resolution=self.instrumental_resolution, skyspecs=self.lin_skyspecs, c=SpectralFit.c_light)

        self.lam_range_gal=lam_range_gal
        self.velscale=velscale
        self.goodpixels=goodpixels

        self.log_lam=logLam
        self.log_galaxy=loggalaxy
        self.log_noise=lognoise
        self.log_weights=logweights

        #These may be None
        self.log_skyspecs=log_skyspecs
        self.log_instrumental_resolution=log_inst_res


    def prepare_CVD2_interpolators(self):

        self.linear_interp, self.logLam_template =CvDTools.prepare_CvD_interpolator_twopartIMF(self.lam_range_temp, self.velscale, verbose=True)
        self.correction_interps, self.log_lam_template=CvDTools.prepare_CvD_correction_interpolators(self.lam_range_temp, self.velscale, self.elements_to_fit, verbose=True, element_imf=self.element_imf)

    def get_emission_lines(self, vac_or_air='air'):

        self.emission_lines, self.line_names, self.line_wave=SF.emission_lines(self.log_lam_template, self.lam_range_gal, self.FWHM_gal, vac_or_air=vac_or_air, quiet=False)
        

    def likelihood(self, theta):

        return SF.lnlike(theta, self.fit_settings)


    def plot_fit(self, theta):

        #Helper function to call SF function

        chisq, chisq_per_dof, (fig, axs)=SF.plot_fit(theta, self.fit_settings)

        return chisq, chisq_per_dof, (fig, axs)




