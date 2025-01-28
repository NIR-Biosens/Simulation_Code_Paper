import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import pandas as pd
from fractions import Fraction

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

def signal_noise(wavelength, QY_F, epsilon_F_peak, c_F, stokes_shift, Int_LED, PD, filter_width, z, T, A_spot):
    
    # theoretical emission spectrum with gaussian shape
    def gaussian(wavelength, FWHM, lambda_peak):
        FWHM = FWHM/(2*np.sqrt(2*np.log(2))) 
        return (1/(FWHM*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((wavelength-lambda_peak)/FWHM)**2)))
    
    #%%
    print('signal-to-noise calculation:')
    print('Fluorophore: ')
    print('quantum yield =  ' + str(QY_F))
    print('molar absorption coefficient =  ' + str(epsilon_F_peak) + '1/(M cm)')
    print('concnetration =  ' + str(c_F))
    print('stokes shift =  ' + str(stokes_shift) + ' nm')
    print('Excitation source: color LED')
    print('excitation irradiance = ' + str(Int_LED) + 'W/cm²')
    
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
        R_lamdba = np.interp(wavelength, wl_R, R_lamdba)
        print('PD = Si')
    
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
        R_lamdba = np.interp(wavelength, wl_R_x, R_lamdba_x)
        
        print('PD = InGaAs')
        
    
    # %% objective

    NA = 0.3 # NA of objective
    CE = 0.5 * (1 - np.cos(np.arcsin(NA))) # collection efficiency 
    
    print('NA = ' + str(NA) + ' (CE = ' + str((CE*100).round(2)) + ' %)')

    # %% model for loss of photons in biological tissue/media

    # intensity decays exponential 
    def biotissue(mu, z):
        return (1-np.exp(- mu * z))/(mu * z)
        # return np.exp(- mu * z)

    # scattering coefficient (1/cm) for whole blood taken from https://omlc.org/news/dec14/Jacques_PMB2013/Jacques_PMB2013.pdf
    a = 22
    b = 0.66
    mu_s = a * (wavelength/500)**(-b)

    # absorption coefficient (1/cm) for water taken from https://omlc.org/spectra/water/data/segelstein81.txt
    path_water = r'U:/InGaAs vs Si/Spektren/Spectrum of water absorption coefficient.csv'
    data = np.genfromtxt(path_water, delimiter=',', skip_header=2)
    wavelength_water = data[:, 0]
    mu_a_water = data[:, 1]
    
    # interpolate data for calculation
    mu_a_water = np.interp(wavelength, wavelength_water, mu_a_water)
    
    
    # absorption coefficient (1/cm /M) for hemoglobin taken from https://omlc.org/spectra/hemoglobin/
    path = r'U:/InGaAs vs Si/Spektren/Hemoglobin Absorbance.txt'
    data = np.genfromtxt(path, delimiter=',', skip_header=2)
    wavelength_Hb = data[:, 0]
    
    M_Hb = 64500 # molar mass of hemoglobin (g/mol)
    c_Hb = 150 # mean concentartion of hemoglobin (g/L)

    mu_a_HbO2 = 2.303 * data[:, 1] * c_Hb/M_Hb * 0.9 # oxygen saturation ~90% -> 0.9 
    mu_a_Hb = 2.303 * data[:, 2] * c_Hb/M_Hb * 0.1 # ~10% not oxyginated -> 0.1 
    
    wl_add = np.linspace(1300, 1700, 401)
    wavelength_Hb = np.concatenate((wavelength_Hb, wl_add))
    mu_a_HbO2_add = np.zeros(len(wl_add))
    mu_a_HbO2 = np.concatenate((mu_a_HbO2, mu_a_HbO2_add))

    mu_a_Hb_add = np.zeros(len(wl_add))
    mu_a_Hb = np.concatenate((mu_a_Hb, mu_a_Hb_add))
    
    # interpolate data for calculation
    mu_a_Hb = np.interp(wavelength, wavelength_Hb, mu_a_Hb)
    mu_a_HbO2 = np.interp(wavelength, wavelength_Hb, mu_a_HbO2)

    # add all absorption and scattering coefficients, z = tissue/media depth in cm -> Biotissue = fractional loss of photons per wavelength
    Biotissue = biotissue(mu_a_water + mu_a_Hb + mu_a_HbO2 + mu_s, z)

    print('tissue/media thickness z = ' + str(z) + ' cm')

    # %%  Autofluorescence data
    # reads autofluorescence excitation and emission spectra and adjust it to wavelength range, data is given from 200 to 800 nm
    def read_spectra(path_af, Em):
        data = np.genfromtxt(path_af, delimiter='	', skip_header=1)
        wavelength_af = data[:, 0]
        af = data[:, 1]/np.max( data[:, 1])
        
        # adjuest the data, that after 800 nm there is no specrtum -> zeros
        if wavelength_af[-1] < wavelength[0]:
            AF = np.zeros(len(wavelength))
        else:
            AF = np.zeros(len(wavelength))
            wl_start = int(wavelength_af[0])-int(wavelength[0])
            wl_end = int(wavelength_af[-1])-int(wavelength[-1])        
            if  wl_start < 0: wl_start = 0
            if  wl_end > 0: wl_end = -1
            AF[wl_start:wl_end] = np.interp(wavelength[wl_start:wl_end], wavelength_af, af)
        
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
    Autofluor = pd.DataFrame(data=d, index=idx)
    
    # %% temperature dependent sources of background noise 
    
    print('ambient temperature T = ' + str(T) + ' K')

    # calculating the black body radiation based on Plank's law
    BBR = 2 * np.pi * sc.h * sc.c**2 / ((wl_R_x*1e-9)**5) * 1/(np.exp(sc.h * sc.c / (wl_R_x*1e-9 * sc.k * T)) - 1) * A_PD * R_lamdba_x
    
    I_BBR = np.trapz(BBR, wl_R_x*1e-9)


    # dark current
    def dark_current(T, i_dark, bandgap):
        return i_dark * np.exp(-bandgap/(2*sc.k*T))  # (A)

    I_Dark = dark_current(T, i_dark, bandgap) # dark current in A

    # %% shifting excitation and emssion peak wavelength
    
    # reading the ICG absorption spectra
    path_af = r'Spectral_data/ICG_Abs.txt'
    data = np.genfromtxt(path_af, delimiter='	', skip_header=4)
    A_wl = data[:, 0] # wavelength (nm)
    A = data[:, 1] # absorption 
    A = np.interp(np.arange(A_wl[0], A_wl[-1]), A_wl, A)
    Amax_idx = np.argmax(A) # save the psotion of the peak of the spectrum
    
    # reading the ICG emission spectra
    path_af = r'Spectral_data/ICG_Em.txt'
    data = np.genfromtxt(path_af, delimiter='	', skip_header=4)
    Em_wl = data[:, 0] # wavelength (nm)
    Em = data[:, 1] # emission 
    Em = np.interp(np.arange(Em_wl[0], Em_wl[-1]), Em_wl, Em)
    Emmax_idx = np.argmax(Em) # save the psotion of the peak of the spectrum
    
    # shift the absotrpion and emission spectra with respect to the postion of the peak
    def shift_peak(spectrum, peak_index, x):
    # Create a new array filled with zeros
        new_spectrum = np.zeros_like(spectrum)
        
        # Calculate the shift
        shift = int(x - peak_index)
        
        # Shift the spectrum
        if shift > 0:
            new_spectrum[shift:] = spectrum[:-shift]
        elif shift < 0:
            new_spectrum[:shift] = spectrum[-shift:]
        else:
            new_spectrum = spectrum
        
        return new_spectrum
    

    # the length of the simulation output is defined by the stokes shift and the width of the filter
    length = wavelength.size - stokes_shift - filter_width
    SNR = np.zeros((length))
    I_Em_out = np.zeros((length))
    I_Ex_out = np.zeros((length))
    I_AF_out = np.zeros((length))
    I_Dark_out = np.zeros((length))
    I_BBR_out = np.zeros((length))
    
    # Reflectance at the surface glass-air -> 4% assumption
    R = 0.04
    
    # interate over wavelength range in 1nm-steps
    for ix in range(length):
        lambda_Ex = wavelength[0] + filter_width/2 + ix # excitation wavelength
        
        # create a excitation bandpass filter
        T_SP = np.ones(len(wavelength)) * 1e-6  # OD6 filtering
        T_SP[int(lambda_Ex - filter_width/2 - wavelength[0]):int(lambda_Ex + filter_width/2 - wavelength[0])] = 0.98  # 98 % Transmission

        # create a single-color LED with a gausian spectrum 
        Ex = gaussian(wavelength, 60, lambda_Ex) # FWHM = 60 nm
        Ex = Ex * T_SP  # filter trasnmission applied on spectrum
        Phi_Ex = Int_LED * A_spot * Ex / (sc.c * sc.h / wavelength) # Photons/s 
        
        # create the gausian spectrum for emission
        
        lambda_A = filter_width + ix - 5
        A_x = shift_peak(A, Amax_idx, lambda_A)
        A_x = A_x[:len(wavelength)]
        F_A = A_x/np.max(A_x)
        epsilon_F = F_A/np.max(F_A) *  epsilon_F_peak
        
        lambda_Em = lambda_A + stokes_shift
        Em_x = shift_peak(Em, Emmax_idx, lambda_Em)
        Em_x = Em_x[:len(wavelength)]
        F_Em = Em_x/np.trapz(Em_x)   

        Phi_Em = np.trapz(Phi_Ex * (1 - np.exp(-1 * np.log(10) * epsilon_F * c_F * z)), wavelength) * F_Em * QY_F   # Photons/s
        
        # create a dicroic mirror
        lambda_x = (lambda_Em + lambda_A)/2
        T_dichroit = np.ones(len(wavelength)) * 1e-3 # OD3 filtering
        T_dichroit[int(lambda_x):] = 0.98  # 98 % Transmission
        
        # create a emission bandpass-filter
        lambda_x = lambda_Em + filter_width/2 - 5
        T_longpass = np.ones(len(wavelength)) * 1e-6  # OD6 filtering
        T_longpass[int(lambda_x - filter_width/2):int(lambda_x + filter_width/2)] = 0.98  # 98 % Transmission
        
        Phi_Em = Phi_Em * T_dichroit * T_longpass * Biotissue * CE * (1 - R)  # Fluorophore fluorescence transmission  Photons/s
        I_Em_wl = Phi_Em * (sc.c * sc.h / wavelength) * R_lamdba  # Fluorophore fluorescence transmission (W)
        
        Phi_Ex_back = Phi_Ex * T_dichroit * T_longpass * R * CE # amount of photons which are backreflected to the detector  Photons/s
        I_Ex_wl = Phi_Ex_back * (sc.c * sc.h / wavelength) * R_lamdba    # in (W))
        
        # integrate photons to get intensity detected on camera
        I_Em = np.trapz(I_Em_wl, wavelength)
        
        I_Ex = np.trapz(I_Ex_wl, wavelength)
        
        # print when emission = 805 nm (ICG emission peak)
        if lambda_Em == 805:
            print('\n---------------------------------------------------------------------------------------')
            print('\nemission = 805 nm (ICG)')
            print('\nemission current: I = {0:2.3f} nA \n'.format(I_Em*1e9))
            
            mu = (np.trapz(Phi_Em)/np.trapz(Phi_Ex)).round(10)
            print('efficiency (# detected photons/# excitation photons): ' + str(Fraction(str(mu))))

        # calculation of autofluorescnce background noise
        AF = np.zeros(Autofluor.shape[0])
        
        for i in range(Autofluor.shape[0]):
            if  type(Autofluor['Excitation [nm]'][i]) == int:
                AF_A = gaussian(wavelength, 30, Autofluor['Excitation [nm]'][i])
                Epsilon_AF = AF_A/np.max(AF_A) * Autofluor['Molar absorption coefficient [1/(M cm)]'][i] 
                
                Phi_Ex_AF = np.trapz(Phi_Ex * (1 - np.exp(-1*(Epsilon_AF * np.log(10)* Autofluor['Concentration [mol/L]'][i] * z))), wavelength)
            else:
                Phi_Ex_AF = np.trapz(Phi_Ex * (1 - np.exp(-1*( Autofluor['Excitation [nm]'][i] * np.log(10) * Autofluor['Concentration [mol/L]'][i] * z))) , wavelength)
            
            if  type(Autofluor['Emission [nm]'][i]) == int:
                AF_Em = gaussian(wavelength, 30, Autofluor['Emission [nm]'][i])
            else:
                AF_Em = Autofluor['Emission [nm]'][i]
    
            Phi_Em_AF = Phi_Ex_AF * AF_Em * Autofluor['Quantum yield'][i] 
            
            Phi_Em_AF = Phi_Em_AF * T_dichroit * T_longpass * Biotissue * CE * (1 - R) #  Photons/s
            I_AF_wl_x = Phi_Em_AF * (sc.c * sc.h / wavelength) * R_lamdba   # spectral current (A)
            AF[i] = np.trapz(I_AF_wl_x, wavelength) # integrated current (A)


        # sum over all autofluorescence sources
        I_AF = np.nansum(AF)


        # calculate signal-to-background (SNR) in dB-scale
        SNR[ix] = 10 * np.log10(I_Em/(I_AF + I_Ex + I_Dark + I_BBR))
        
        # save the idividual photo currents
        I_Em_out[ix] = I_Em
        I_Ex_out[ix] = I_Ex
        I_AF_out[ix] = I_AF
        I_Dark_out[ix] = I_Dark
        I_BBR_out[ix] = I_BBR
    #%% print maximum SNR
    
    print('\n---------------------------------------------------------------------------------------')
    print('Maximal Signal-to-Noise Ratio: ' + str(SNR.max().round(2)) + ' dB at ' +
          str(int(np.argmax(SNR) + wavelength[0] + stokes_shift + filter_width/2)) + ' nm  \n')

    return SNR, I_Em_out, I_Ex_out, I_Dark_out, I_BBR_out, I_AF_out

# %% Input:
 
## focal volume
A_spot = np.pi * 0.25**2 # focal spot (cm²)
z = 1 # thickness (cm)

## Fluorophore with properties similiar to a ICG
QY_F = 0.05  # quantum yield
epsilon_F_peak = 156000  # molar absorption coefficient (1/(M cm))
c_F = 1e-9 # fluorophore concentration (mol/L)
stokes_shift = 805-779 # stokes shift (nm)

## Excitation
Int_LED = 250e-3  # intensity in focus (W/cm²)

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

        length_Si = wavelength_Si.size - stokes_shift - filter_width
        
        SNR_Si, I_Em_out_Si, I_Ex_out_Si, I_Dark_out_Si, I_BBR_out_Si, I_AF_out_Si = signal_noise(wavelength_Si, QY_F, epsilon_F_peak, c_F,
                       stokes_shift, Int_LED, PD, filter_width, z, T, A_spot)

    elif PD == 2:
        
        lambda_start = 800
        lambda_end = 1700
        
        wavelength_InGaAs = np.linspace(lambda_start, lambda_end,
                             lambda_end-lambda_start+1)

        length_InGaAs = wavelength_InGaAs.size - stokes_shift - filter_width
        
        SNR_InGaAs, I_Em_out_InGaAs, I_Ex_out_InGaAs, I_Dark_out_InGaAs, I_BBR_out_InGaAs, I_AF_out_InGaAs = signal_noise(wavelength_InGaAs, QY_F, epsilon_F_peak, c_F,
                       stokes_shift, Int_LED, PD, filter_width, z, T, A_spot)


#%% plot results of SNR
    
fig = plt.figure( figsize = (5,3))
ax = fig.add_subplot()

wl = wavelength_Si[:length_Si]+stokes_shift + filter_width/2

ax.plot(wl, SNR_Si, label=txt[0], c = '#0051b3ff')
ax.set_ylabel('Signal-to-noise ratio (dB)', color='#0051b3ff')

wl2 = wavelength_InGaAs[:length_InGaAs]+stokes_shift +filter_width/2

ax2 = ax.twinx()
ax2.plot(wl2, SNR_InGaAs, label=txt[1], c = '#b3000fff')
ax2.set_ylabel('Signal-to-noise ratio (dB)', color='#b3000fff')
         
ax.set_xlabel('Peak emission wavelength (nm)')
ax.set_xlim([250, 1700])
ax.set_ylim([-30, 22])

ax.xaxis.set_ticks_position('both')
ax.tick_params(direction="in", which = 'both')
ax2.tick_params(direction="in", which = 'both')

fig.tight_layout()

plt.savefig(r'U:\InGaAs vs Si\Paper\Main Figures\Figure 1\S4b.png', dpi=500)
plt.savefig(r'U:\InGaAs vs Si\Paper\Main Figures\Figure 1\S4b.svg')

#%% plot results of spectral distribution of the current sources 

fig = plt.figure( figsize = (5,3))
ax = fig.add_subplot()

ax.plot(wl, I_Em_out_Si*1e9, c = co[1], label = 'Emission')
ax.plot(wl, I_Ex_out_Si*1e9, c = co[5], label = 'Excitation')
ax.plot(wl, I_Dark_out_Si*1e9, c = co[0], label = 'Dark current')
ax.plot(wl, I_BBR_out_Si*1e9, c = 'black', label = 'Plank radiation')
ax.plot(wl, I_AF_out_Si*1e9, c = co[2], label = 'Autofluorescence')

ax.plot(wl2, I_Em_out_InGaAs*1e9, c = co[1], label = 'Emission', ls = '--')
ax.plot(wl2, I_Ex_out_InGaAs*1e9, c = co[5], label = 'Excitation', ls = '--')
ax.plot(wl2, I_Dark_out_InGaAs*1e9, c = co[0], label = 'Dark current', ls = '--')
ax.plot(wl2, I_BBR_out_InGaAs*1e9, c = 'black', label = 'Plank radiation', ls = '--')
ax.plot(wl2, I_AF_out_InGaAs*1e9, c = co[2], label = 'Autofluorescence', ls = '--')

ax.set_yscale('log')

ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')

ax.tick_params(direction="in", which = 'both')
ax.set_xlim([250, 1700])
ax.set_xlabel('Peak emission wavelength (nm)')
ax.set_ylabel('Photo current (nA)')
ax.set_ylim([1e-12, 1e1])

ax.legend()

fig.tight_layout()
plt.savefig(r'U:\InGaAs vs Si\Paper\Main Figures\Figure 1\S4d.png', dpi=500)
plt.savefig(r'U:\InGaAs vs Si\Paper\Main Figures\Figure 1\S4d.svg')