import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import pandas as pd
from fractions import Fraction
from mpl_toolkits.axes_grid1 import make_axes_locatable
import colorcet as cc

#%%
co = [ 
    '#2465b3ff',  # blue
    '#b32430ff',   # red
    '#ba25baff',  # purple
    '#29bdccff',   # ligth blue
    '#ffb033ff',  # orange
    '#21a67aff',  # green
    '#6c3616ff',   # brown

]

#%%

class Signal_Noise():
    def __init__(self, wavelength, QY_F, epsilon_F_peak, c_F, stokes_shift, PD, filter_width, z, T, A_spot):
        
        self.wavelength = wavelength
        self.stokes_shift = stokes_shift
        self.filter_width = filter_width
        self.c_F = c_F
        self.epsilon_F_peak = epsilon_F_peak
        self.A_spot = A_spot
        self.z = z

        
        
        # %% detetcor
       
        A_PD = (1e-3)**2 # detector area (m²)
        
        if PD == 1: # initialize Silicon detetcor
            
            bandgap = 1.12*sc.e  # bandgap of silicon (eV)
            i_dark_25 = 100e-15 # dark current at 25°C (A)
            i_dark = i_dark_25 / np.exp(-bandgap/(2*sc.k*(273.15+25)))  # i_dark_0
            
            # responsitivity of the photo detector
            path_R_PD = r'Spectral_data/Si_Hamamatsu_S2386.txt' # 
            data = np.genfromtxt(path_R_PD, delimiter=',')
            wl_R_x = data[:, 0]
            R_lamdba_x = data[:, 1]/1000
            
    
            # assumption that after 1100 nm the responsitivity is zero due to bandgap
            wl_add = np.linspace(1100, 1800, 701)
            wl_R = np.concatenate((wl_R_x, wl_add))
            R_lamdba_add = np.zeros(len(wl_add))
            R_lamdba = np.concatenate((R_lamdba_x, R_lamdba_add))
            
            # interpolated data for calculation
            self.R_lamdba = np.interp(self.wavelength, wl_R, R_lamdba)
        
        elif PD == 2: # initialize InGaAs detector
            
            bandgap = 0.75*sc.e  # bandgap (eV)
            i_dark_25 = 90e-12# dark current at 25°C (A)
            i_dark = 90e-12 / np.exp(-bandgap/(2*sc.k*(273.15+25))) # i_dark_0
            
            # responsitivity of the photo detector
            path_R_PD = r'Spectral_data/InGaAs_Hamamatsu_G12180.txt'
            data = np.genfromtxt(path_R_PD, delimiter=',')
            wl_R_x = data[:, 0]
            R_lamdba_x = data[:, 1]
        
            
            # interpolated data for calculation
            self.R_lamdba = np.interp(self.wavelength, wl_R_x, R_lamdba_x)
            
            
        
        # %% objective
    
        NA = 0.3 # NA of objective
        self.CE = 0.5 * (1 - np.cos(np.arcsin(NA))) # collection efficiency 
     
    
        # %% model for loss of photons in biological tissue/media
    
        # intensity decays exponential 
        def biotissue(mu, z_x):
            return (1-np.exp(- mu * z_x))/(mu * z_x)
    
        # scattering coefficient (1/cm) for whole blood taken from https://omlc.org/news/dec14/Jacques_PMB2013/Jacques_PMB2013.pdf
        a = 22
        b = 0.66
        mu_s = a * (self.wavelength/500)**(-b)
    
        # absorption coefficient (1/cm) for water taken from https://omlc.org/spectra/water/data/segelstein81.txt
        path_water = r'Spectral_data/Water_Absorbance.txt'
        data = np.genfromtxt(path_water, delimiter=',', skip_header=2)
        self.wavelength_water = data[:, 0]
        mu_a_water = data[:, 1]
        
        # interpolate data for calculation
        mu_a_water = np.interp(self.wavelength, self.wavelength_water, mu_a_water)
        
        
        # absorption coefficient (1/cm /M) for hemoglobin taken from https://omlc.org/spectra/hemoglobin/
        path = r'Spectral_data/Hemoglobin_Absorbance.txt'
        data = np.genfromtxt(path, delimiter=',', skip_header=2)
        self.wavelength_Hb = data[:, 0]
        
        M_Hb = 64500 # molar mass of hemoglobin (g/mol)
        c_Hb = 150 # mean concentartion of hemoglobin (g/L)
    
        mu_a_HbO2 = 2.303 * data[:, 1] * c_Hb/M_Hb * 0.9 # oxygen saturation ~90% -> 0.9 
        mu_a_Hb = 2.303 * data[:, 2] * c_Hb/M_Hb * 0.1 # ~10% not oxyginated -> 0.1 
        
        wl_add = np.linspace(1300, 1700, 401)
        self.wavelength_Hb = np.concatenate((self.wavelength_Hb, wl_add))
        mu_a_HbO2_add = np.zeros(len(wl_add))
        mu_a_HbO2 = np.concatenate((mu_a_HbO2, mu_a_HbO2_add))
    
        mu_a_Hb_add = np.zeros(len(wl_add))
        mu_a_Hb = np.concatenate((mu_a_Hb, mu_a_Hb_add))
        
        # interpolate data for calculation
        mu_a_Hb = np.interp(self.wavelength, self.wavelength_Hb, mu_a_Hb)
        mu_a_HbO2 = np.interp(self.wavelength, self.wavelength_Hb, mu_a_HbO2)
    
        # add all absorption and scattering coefficients, z = tissue/media depth in cm -> Biotissue = fractional loss of photons per self.wavelength
        self.Biotissue = biotissue(mu_a_water + mu_a_Hb + mu_a_HbO2 + mu_s, self.z)
    
    
        # %%  Autofluorescence data
        
        # reads autofluorescence excitation and emission spectra and adjust it to self.wavelength range
        def read_spectra(path_af, Em):
            data = np.genfromtxt(path_af, delimiter='	', skip_header=1)
            wavelength_af = data[:, 0]
            af = data[:, 1]/np.max( data[:, 1])
            
            # adjuest the data, that after 800 nm there is no specrtum -> zeros
            if wavelength_af[-1] < self.wavelength[0]:
                AF = np.zeros(len(self.wavelength))
            else:
                AF = np.zeros(len(self.wavelength))
                wl_start = int(wavelength_af[0])-int(self.wavelength[0])
                wl_end = int(wavelength_af[-1])-int(self.wavelength[-1])        
                if  wl_start < 0: wl_start = 0
                if  wl_end > 0: wl_end = -1
                AF[wl_start:wl_end] = np.interp(self.wavelength[wl_start:wl_end], wavelength_af, af)
            
            if Em == 1:    
                AF = AF/ np.trapz(AF)
                
            return AF
    
    
        ## read excitation and emission spectra of endogenous fluorophores
        # Porphyrins 
        Em_Porphyrins = read_spectra(r'Spectral_data/Porphyren_emission.txt', 1)
        Ex_Porphyrins = read_spectra(r'Spectral_data/Porphyren_excitation.txt',0)  
        # Bilirubin
        Ex_Bilirubin = read_spectra(r'Spectral_data/Billirubin_excitation.txt',0)
        Em_Bilirubin = read_spectra(r'Spectral_data/Billirubin_emission.txt', 1)
        # Ribolfavin
        Ex_Ribolfavin = read_spectra(r'Spectral_data/Ribolfavin_excitation.txt',0)
        Em_Ribolfavin = read_spectra(r'Spectral_data/Ribolfavin_emission.txt', 1)
        # amino acids: Phenylalanine, Tryptophan,  Tyrosine
        Ex_Phenylalanine = read_spectra(r'Spectral_data/Phenylalanine_excitation.txt',0)
        Em_Phenylalanine = read_spectra(r'Spectral_data/Phenylalanine_emission.txt', 1)
        Ex_Tryrosine = read_spectra(r'Spectral_data/Tryrosine_excitation.txt',0)
        Em_Tryrosine = read_spectra(r'Spectral_data/Tryrosine_emission.txt', 1)
        Ex_Tryptophan = read_spectra(r'Spectral_data/Tryptophan_excitation.txt',0)
        Em_Tryptophan = read_spectra(r'Spectral_data/Tryptophan_emission.txt', 1)
    
    
        # list of a s selection of endogenous fluorophores present in human blood
        idx = ["Phenylalanine", "Tryptophan", 'Tyrosine', 'Riboflavin', 'NADH/NAD', 'Porphyrins', 'FAD', 'FMN', 'Retinol', 'Vitamin K','Vitamin D','Vitamin B12','Bilirubin'] # chosen endogenous fluorophores
    
        # properties of the endogenous fluorophores. If no specrtum is given, a gaussian shape with a peak wavelength given in the table was used
        d = {'Excitation [nm]': [Ex_Phenylalanine, Ex_Tryptophan, Ex_Tryrosine, Ex_Ribolfavin, 355, Ex_Porphyrins, 430, 445, 327,335,390,275,Ex_Bilirubin],
             'Emission [nm]': [Em_Phenylalanine, Em_Tryptophan, Em_Tryrosine, Em_Ribolfavin, 462, Em_Porphyrins, 535, 540, 510,480,480,305,Em_Bilirubin],
             'Concentration [mol/L]': np.array([98e-6, 70e-6, 77e-6, 28.4e-9, 0.3e-6, 7.5e-9, 74e-9,10.5e-9, 55.25e-5/286.4516, 1.5e-6/450.69574, 37.5e-9/384.6 , 500e-9/1355.4, 0.225e-2/584.673]),
             'Molar absorption coefficient [1/(M cm)]': np.array([195, 5579, 1405, 33000, 6220, 166000, 11500, 11500, 52770, 19900, 18300, 30800, 55000]),
             'Quantum yield': [0.022, 0.12, 0.13, 0.3, 0.6, 0.06, 0.04, 0.22, 0.15, 0.15, 0.15, 0.15, 0.1]}
    
        # create a dataframe with all parameters of the endogenous fluorophores
        self.Autofluor = pd.DataFrame(data=d, index=idx)
        
        # %% temperature dependent sources of background noise 
            
        # calculating the black body radiation based on Plank's law
        BBR = 2 * np.pi * sc.h * sc.c**2 / ((wl_R_x*1e-9)**5) * 1/(np.exp(sc.h * sc.c / (wl_R_x*1e-9 * sc.k * T)) - 1) * A_PD * R_lamdba_x
        
        self.I_BBR = np.trapz(BBR, wl_R_x*1e-9) # (A)
    
    
        # dark current
        def dark_current(T, i_dark, bandgap):
            return i_dark * np.exp(-bandgap/(2*sc.k*T))  # (A)
    
        self.I_Dark = dark_current(T, i_dark, bandgap) # dark current in (A)
        
        #%% LED backreflection - Reflectance at the surface glass-air -> 4% assumption

        self.R = 0.04
    
        # %% shifting excitation and emssion peak wavelength
        
        # theoretical emission spectrum with gaussian shape
    def gaussian(self, wave, FWHM, lambda_peak):
        FWHM = FWHM/(2*np.sqrt(2*np.log(2))) 
        return (1/(FWHM*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((wave-lambda_peak)/FWHM)**2)))
    
    def signal_noise(self, Int_LED):
        
        # the length of the simulation output is defined by the stokes shift and the width of the filter
        length = self.wavelength.size - self.stokes_shift - self.filter_width
        SNR = np.zeros((length))
        
        # interate over wavelength range in 1nm-steps
        for ix in range(length):
            lambda_Ex = self.wavelength[0] + filter_width/2 + ix # excitation self.wavelength
            
            # create a excitation bandpass filter
            T_SP = np.ones(len(self.wavelength)) * 1e-6  # OD6 filtering
            T_SP[int(lambda_Ex - filter_width/2 - self.wavelength[0]):int(lambda_Ex + filter_width/2 - self.wavelength[0])] = 0.98  # 98 % Transmission
        
            # create a single-color LED with a gausian spectrum 
            Ex = self.gaussian(self.wavelength, 60, lambda_Ex) # FWHM = 60 nm
            Ex = Ex * T_SP  # filter trasnmission applied on spectrum
            Phi_Ex = Int_LED * self.A_spot * Ex / (sc.c * sc.h / self.wavelength) # Photons/(s cm²)
            
            # create the gausian spectrum for emission
            lambda_Em = lambda_Ex + stokes_shift
            F_Em = self.gaussian(self.wavelength, 30, lambda_Em)
            
            F_A = self.gaussian(self.wavelength, 30, lambda_Ex)
            epsilon_F = F_A/np.max(F_A) *  self.epsilon_F_peak
            
            Phi_Em = np.trapz(Phi_Ex * (1 - np.exp(-1 * np.log(10) * epsilon_F * self.c_F * self.z)), self.wavelength) * F_Em * QY_F   # Photons/s
            
            # create a dicroic mirror
            lambda_x = lambda_Ex + stokes_shift / 2
            T_dichroit = np.ones(len(self.wavelength)) * 1e-3 # OD3 filtering
            T_dichroit[int(lambda_x - self.wavelength[0]):] = 0.98  # 98 % Transmission
            
            # create a emission bandpass-filter
            T_longpass = np.ones(len(self.wavelength)) * 1e-6  # OD6 filtering
            T_longpass[int(lambda_Em - filter_width/2 - self.wavelength[0]):int(lambda_Em +
                                                                          filter_width/2 - self.wavelength[0])] = 0.98  # 98 % Transmission
            
            Phi_Em = Phi_Em * T_dichroit * T_longpass * self.Biotissue * self.CE * (1 - self.R)  # Fluorophore fluorescence transmission  Photons/s
            I_Em_wl = Phi_Em * (sc.c * sc.h / self.wavelength) * self.R_lamdba  # Fluorophore fluorescence transmission (W)
            
            Phi_Ex_back = Phi_Ex * T_dichroit * T_longpass * self.R * self.CE # amount of photons which are backreflected to the detector  Photons/s
            I_Ex_wl = Phi_Ex_back * (sc.c * sc.h / self.wavelength) * self.R_lamdba    # in (W))
            
            # integrate photons to get intensity detected on camera
            I_Em = np.trapz(I_Em_wl, self.wavelength)
            
            I_Ex = np.trapz(I_Ex_wl, self.wavelength)
            
    
            # calculation of autofluorescnce background noise
            AF = np.zeros(self.Autofluor.shape[0])
            
            for i in range(self.Autofluor.shape[0]):
                if  type(self.Autofluor['Excitation [nm]'][i]) == int:
                    AF_A = self.gaussian(self.wavelength, 30, self.Autofluor['Excitation [nm]'][i])
                    Epsilon_AF = AF_A/np.max(AF_A) * self.Autofluor['Molar absorption coefficient [1/(M cm)]'][i] 
                    
                    Phi_Ex_AF = np.trapz(Phi_Ex * (1 - np.exp(-1*(Epsilon_AF * np.log(10)* self.Autofluor['Concentration [mol/L]'][i] * self.z))), self.wavelength)
                else:
                    Phi_Ex_AF = np.trapz(Phi_Ex * (1 - np.exp(-1*( self.Autofluor['Excitation [nm]'][i] * np.log(10) * self.Autofluor['Concentration [mol/L]'][i] * self.z))) , self.wavelength)
                
                if  type(self.Autofluor['Emission [nm]'][i]) == int:
                    AF_Em = self.gaussian(self.wavelength, 30, self.Autofluor['Emission [nm]'][i])
                else:
                    AF_Em = self.Autofluor['Emission [nm]'][i]
        
                Phi_Em_AF = Phi_Ex_AF * AF_Em * self.Autofluor['Quantum yield'][i] 
                
                Phi_Em_AF = Phi_Em_AF * T_dichroit * T_longpass * self.Biotissue * self.CE * (1 - self.R) #  Photons/s
                I_AF_wl_x = Phi_Em_AF * (sc.c * sc.h / self.wavelength) * self.R_lamdba   # LED transmission (electrons/s)
                AF[i] = np.trapz(I_AF_wl_x, self.wavelength) 
        
        
            I_AF = np.nansum(AF)

            # calculate signal-to-background (SNR) in dB-scale
            SNR[ix] = 10 * np.log10(I_Em/(I_AF + I_Ex + self.I_Dark + self.I_BBR))
            
            
                    


        return SNR

# %% Input:
## focal volume
A_spot = np.pi * 0.25**2 # focal spot (cm²)
z = 1 # thickness (cm)

## Fluorophore with properties similiar to a (6,5)-SWCNT
QY_F = 0.01  # quantum yield
l_f = 600 # mean length of (6,5)-SWCNT (nm)
n_c = 88 # C-atoms per nm of the SWCNT (C-atoms/nm)
epsilon_F_peak_C = 6700 #  absorption cross section of a C-atom (cm²/C-atom)
epsilon_F_peak = epsilon_F_peak_C * l_f * n_c # absorption cross section (cm²/CNT)
c_F = 1e-9 # fluorophore concentration (mol/L)
stokes_shift = 400 # stokes shift (nm)

## Excitation
power = np.logspace(-7, 1, 100)  # intensity in focus (W/cm²)

## filter setup
filter_width = 50 # band width for excitation and emssion bandpass filter (nm)

## ambient temperature (K)
T = 273.15+23  

## Photodiodes
txt = ['Si', 'InGaAs']

for PD in [1, 2]:

    # wavelength range (nm)
    if PD == 1:
        
        lambda_start = 175
        lambda_end = 1150
        
        wavelength_Si = np.linspace(lambda_start, lambda_end,
                                 lambda_end-lambda_start+1)
        
        si = Signal_Noise(wavelength_Si, QY_F, epsilon_F_peak, c_F, stokes_shift, PD, filter_width, z, T, A_spot)

        length_si = wavelength_Si.size - stokes_shift - filter_width
        
        
        SNR_si = np.zeros((len(power), length_si))
        
        for i in range(len(power)):

            SNR_si[i,:] = si.signal_noise(power[i])
            print(i) 

    elif PD == 2:
        
        lambda_start = 401
        lambda_end = 1700
        
        wavelength_InGaAs = np.linspace(lambda_start, lambda_end,
                             lambda_end-lambda_start+1)
        
        ingaas = Signal_Noise(wavelength_InGaAs, QY_F, epsilon_F_peak, c_F, stokes_shift, PD, filter_width, z, T, A_spot)


        length_InGaAs = wavelength_InGaAs.size - stokes_shift - filter_width
        
        SNR_ingaas = np.zeros((len(power), length_InGaAs))
        
        for i in range(len(power)):

            SNR_ingaas[i,:] = ingaas.signal_noise(power[i])
            print(i) 


#%% plot results

lambda_start = 175
lambda_end = 1700 

wavelength = np.linspace(lambda_start, lambda_end,
                         lambda_end-lambda_start+1, dtype = 'int16')

vmax =  +72
vmin = -72

SNR = np.empty((100, 1076))
SNR[:] = np.nan
SNR[:,:526] = SNR_si

fig = plt.figure(figsize = (6.5,5))
ax = fig.subplots(2)

length = wavelength.size - stokes_shift - filter_width

X,Y=np.meshgrid(wavelength[:length]+stokes_shift + filter_width/2,power)

im1 = ax[0].contourf(X,Y,SNR, cmap=cc.cm.CET_D1A, levels = 30, vmin = vmin, vmax = vmax)
contour = ax[0].contour(X, Y, SNR, levels=[20], colors='black', linewidths=2)
ax[0].set_yscale('log')

ax[0].set_ylabel('Excitation intensity (W/cm²)')
ax[0].set_xticklabels([])
ax[0].yaxis.set_ticks_position('both')
ax[0].xaxis.set_ticks_position('both')
ax[0].tick_params(direction="in", which = 'both')
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax= cax1)
cbar1.set_ticks([-50, -25,0,25, 50])

cbar1.set_label('SNR (dB)')  

SNR = np.empty((100, 1076))
SNR[:] = np.nan
SNR[:,-850:] = SNR_ingaas[:,:]

length = wavelength.size - stokes_shift - filter_width

X,Y=np.meshgrid(wavelength[:length]+stokes_shift + filter_width/2,power)

im2 = ax[1].contourf(X,Y,SNR, cmap=cc.cm.CET_D1A, levels = 30, vmin = vmin, vmax = vmax)
contour = ax[1].contour(X, Y, SNR, levels=[20], colors='black', linewidths=2)
ax[1].set_yscale('log')
ax[1].set_xlabel('Peak emission wavelength (nm)')
ax[1].set_ylabel('Excitation intensity (W/cm²)')
ax[1].yaxis.set_ticks_position('both')
ax[1].xaxis.set_ticks_position('both')
ax[1].tick_params(direction="in", which = 'both')
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im1, cax= cax2)
cbar2.set_ticks([-50, -25,0,25, 50])
cbar2.set_label('SNR (dB)')

fig.tight_layout()

# np.savetxt(r'U:\InGaAs vs Si\Paper\Python Auswertungen\SNR_Excitation_15_11_2024.csv', SNR, delimiter=',')

plt.savefig(r'U:\InGaAs vs Si\Paper\Main Figures\Figure 1\1d.png', dpi=500)
plt.savefig(r'U:\InGaAs vs Si\Paper\Main Figures\Figure 1\1d.svg')