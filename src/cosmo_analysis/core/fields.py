# TODO: Description of file for documentation

# TODO: Description of functions for documentation

from ..core import constants
import yt
import unyt
import numpy as np

## Simple fields
def _density_squared(field, data):  
	return data[(constants.gasPart, "Density")]**2
def _res(field, data):  
    return (data[(constants.gasPart, "Masses")]/data[(constants.gasPart, "Density")])**(1./3.)
def _inv_vol_sq(field, data):  
    return (data[(constants.gasPart, "Masses")]/data[(constants.gasPart, "Density")])**(-2)
def _particle_position_cylindrical_z_abs(field, data):
    return np.abs(data[(constants.gasPart, "particle_position_cylindrical_z")])
def _den_low_res(field,data):
    trans = np.zeros(data[(constants.gasPart, "particle_mass")].shape)
    ind = np.where(data[(constants.gasPart, "particle_mass")] > 0) 
    trans[ind] = data[(constants.gasPart, "particle_mass")][ind].in_units('Msun').d/(constants.lowResMock**2)
    return data.ds.arr(trans, "Msun/pc**2").in_base(data.ds.unit_system.name)
def _stardensity(field, data):  
    return data[(constants.starPart, "Masses")].in_units('Msun')/yt.YTArray(constants.starFigSize/constants.starBufferSize*constants.starFigSize/constants.starBufferSize*1000.*1000.,'pc**2')   
# GADGET3 temperature conversion
temperature_values = []
mu_values = []
T_over_mu_values = []
current_temperature = 1e1#2.73*(pf.current_redshift+1)
final_temperature = 1e9
dlogT = 1.
def calc_mu_table_local(temperature):
    tt = np.array([1.0e+01, 1.0e+02, 1.0e+03, 1.0e+04, 1.3e+04, 2.1e+04, 3.4e+04, 6.3e+04, 1.0e+05, 1.0e+09])
    mt = np.array([1.18701555, 1.15484424, 1.09603514, 0.9981496, 0.96346395, 0.65175895, 0.6142901, 0.6056833, 0.5897776, 0.58822635])
    logttt= np.log(temperature)
    logmu = np.interp(logttt,np.log(tt),np.log(mt)) # linear interpolation in log-log space
    return np.exp(logmu)  
def convert_T_over_mu_to_T(T_over_mu):
    global current_temperature
    while current_temperature < final_temperature:
        temperature_values.append(current_temperature)
        current_mu = calc_mu_table_local(current_temperature)
        mu_values.append(current_mu)
        T_over_mu_values.append(current_temperature/current_mu)
        current_temperature = np.exp(np.log(current_temperature)+dlogT)
    logT_over_mu = np.log(T_over_mu)
    logT = np.interp(logT_over_mu, np.log(T_over_mu_values), np.log(temperature_values)) # linear interpolation in log-log space
    return np.exp(logT)
def _temperature(field, data):
    return data[(constants.gasPart, "Temperature")].in_units("K")
def _temperature_GADGET3(field, data):
    HYDROGEN_MASSFRAC = 0.76
    gamma=5.0/3.0
    GAMMA_MINUS1=gamma-1.
    PROTONMASS=unyt.mp.in_units('g')
    BOLTZMANN=unyt.kb.in_units('J/K')
    u_to_temp_fac=(4.0 / (8.0 - 5.0 * (1.0 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1
    # Assume cosmic abundances
    gamma = 5.0/3.0
    T_over_mu_GADGET= data[constants.gasPart, "InternalEnergy"].in_units('J/g')*(gamma-1)*unyt.mp.in_units('g')/unyt.kb.in_units('J/K')
    return yt.YTArray(convert_T_over_mu_to_T(T_over_mu_GADGET), 'K') # now T

## Snapshot depending fields
def makeGlobalFields(centerSnap,part):
    def _elevation(field, data):  
        return data[(part, "particle_position_z")].in_units('kpc') - centerSnap[2]
    def _x_centered(field, data):
        return data[(part, "particle_position_x")].in_units('kpc') - centerSnap[0]
    def _y_centered(field, data):
        return data[(part, "particle_position_y")].in_units('kpc') - centerSnap[1]
    def _metallicityG3(field,data):
        return (data[(part, 'Metallicity')]+yt.YTArray(2.041e-6,''))/constants.zSolar
    def _metallicityMassG3(field,data):
        return (data[(part, 'MetallicityG3')]*constants.zSolar*data[(part,'particle_mass')])
    def _metallicityMass2(field,data):
        return (data[(part, 'Metallicity')]*constants.zSolar*data[(part,'particle_mass')])
    return (_elevation,_x_centered,_y_centered,_metallicityG3,_metallicityMassG3,_metallicityMass2)
# Star fields
def makeStarFields(snap,snht):
    co = yt.utilities.cosmology.Cosmology(hubble_constant=0.702, omega_matter=0.272,omega_lambda=0.728, omega_curvature=0.0)
    def _youngStar_Mass(field, data):  
        trans = np.zeros(data[(constants.starPart, "particle_mass")].shape)
        # Selects stars younger than constants.youngStarAge (in Myr)
        if not snht.cosmo:
            starAge = data[(constants.starPart, "StellarFormationTime")]*1e3
            ind = np.where(starAge > (snap.current_time.in_units('Myr').d - constants.youngStarAge)) 
        else:
            starAgeA  = data[(constants.starPart, "StellarFormationTime")]
            cutOffAge = co.a_from_t(co.quan(snap.current_time.in_units('Myr').d - constants.constants.youngStarAge,"Myr"))
            ind = np.where(starAgeA > cutOffAge) 
      
        trans[ind] = data[(constants.starPart, "particle_mass")][ind].in_units('code_mass')
        return data.ds.arr(trans, "code_mass").in_base(data.ds.unit_system.name)
    def _sfr_den_low_res(field,data):
        trans = np.zeros(data[(constants.starPart, "particle_mass")].shape)
        # Selects stars younger than constants.youngStarAge (in Myr)
        if not snht.cosmo:
            starAge = data[(constants.starPart, "StellarFormationTime")]*1e3
            ind = np.where(starAge > (snap.current_time.in_units('Myr').d - constants.youngStarAge)) 
        else:
            starAgeA  = data[(constants.starPart, "StellarFormationTime")]
            cutOffAge = co.a_from_t(co.quan(snap.current_time.in_units('Myr').d - constants.youngStarAge,"Myr"))
            ind = np.where(starAgeA > cutOffAge) 
            
        trans[ind] = data[(constants.starPart, "particle_mass")][ind].in_units('Msun').d/((constants.lowResMock/1e3)**2)/(constants.youngStarAge*1e6)
        return data.ds.arr(trans, "Msun/yr/kpc**2").in_base(data.ds.unit_system.name)
    return (_youngStar_Mass,_sfr_den_low_res)

## Adding the fields
def add_field_to_snaps(snapArr,snhtArr,centArr):
    for i,snap in enumerate(snapArr):
        globalFields = makeGlobalFields(centArr[i],constants.gasPart)
        snap.add_field((constants.gasPart, "density_squared"), function=_density_squared, units="g**2/cm**6", sampling_type="particle",force_override=True)
        snap.add_field((constants.gasPart, "elevation"), function=globalFields[0], units="kpc", sampling_type="particle", display_name="Elevation",force_override=True,take_log=False)
        snap.add_field((constants.gasPart, "resolution"), function=_res, units="pc", sampling_type="particle", display_name="Resolution $\Delta$ x",force_override=True,take_log=True)
        snap.add_field((constants.gasPart, "inv_volume_sq"), function=_inv_vol_sq, units="pc**(-6)", sampling_type="particle", display_name="Inv squared volume",force_override=True,take_log=True)
        snap.add_field((constants.gasPart, "z_abs"), function=_particle_position_cylindrical_z_abs, take_log=False,units="cm",sampling_type="particle",force_override=True) 
        snap.add_field((constants.gasPart, "x_centered"), function=globalFields[1],display_name="x",units="kpc",sampling_type="particle",force_override=True,take_log=False) 
        snap.add_field((constants.gasPart, "y_centered"), function=globalFields[2],display_name="y",units="kpc",sampling_type="particle",force_override=True,take_log=False) 
        snap.add_field((constants.gasPart, "den_low_res"), function=_den_low_res,display_name="Density",units="Msun/pc**2",sampling_type="particle",force_override=True,take_log=False) 
        if (constants.gasPart,"InternalEnergy") in snap.field_list:
            snap.add_field((constants.gasPart, 'TemperatureG3'), function=_temperature_GADGET3,force_override=True,sampling_type="particle", display_name="Temperature",take_log=False, units="K")
            snap.add_field((constants.gasPart, 'TemperatureG3log'), function=_temperature_GADGET3,force_override=True,sampling_type="particle", display_name="Temperature",take_log=True, units="K")
        else:
            snap.add_field((constants.gasPart, 'TemperatureG3'), function=_temperature,force_override=True,sampling_type="particle", display_name="Temperature",take_log=False, units="K")
            snap.add_field((constants.gasPart, 'TemperatureG3log'), function=_temperature,force_override=True,sampling_type="particle", display_name="Temperature",take_log=True, units="K")

        # Star fields
        if constants.starPart in snap.particle_types:
            globalFieldsStar = makeGlobalFields(centArr[i],constants.starPart)
            starFields       = makeStarFields(snap,snhtArr[i])
            if (constants.starPart,"StellarFormationTime") in snap.field_list: snap.add_field((constants.starPart, "particle_mass_young_stars"),function=starFields[0],display_name="Young Stellar Mass",units="code_mass",sampling_type="particle",force_override=True,take_log=True)
            snap.add_field((constants.starPart, "x_centered"), function=globalFieldsStar[1],display_name="x",units="kpc",sampling_type="particle",force_override=True,take_log=False) 
            snap.add_field((constants.starPart, "y_centered"), function=globalFieldsStar[2],display_name="y",units="kpc",sampling_type="particle",force_override=True,take_log=False) 
            if (constants.starPart,"StellarFormationTime") in snap.field_list: snap.add_field((constants.starPart, "sfr_den_low_res"), function=starFields[1],display_name="SFR Density",units="Msun/yr/kpc**2",sampling_type="particle",force_override=True,take_log=False) 
            snap.add_field((constants.starPart, "stardensity"), function=_stardensity, units="Msun/pc**2",display_name="Star density", sampling_type="local",force_override=True,take_log=True)
            # Metallicity fields
            if "Metallicity" in np.array(snap.field_list)[:,1]:
                snap.add_field((constants.starPart, "MetallicityG3"), function=globalFieldsStar[3], force_override=True,take_log=True, display_name="Metallicity", units="",sampling_type='particle')
                snap.add_field((constants.starPart, "MetallicityMassG3"), function=globalFieldsStar[4], force_override=True,take_log=True, display_name="Metal Mass", units="Msun",sampling_type='particle')
                snap.add_field((constants.starPart, "MetallicityMass2"), function=globalFieldsStar[5], force_override=True,take_log=True, display_name="Metal Mass", units="Msun",sampling_type='particle')

        # Metallicity fields
        if "Metallicity" in np.array(snap.field_list)[:,1]:
            snap.add_field((constants.gasPart, "MetallicityG3"), function=globalFields[3], force_override=True,take_log=True, display_name="Metallicity", units="",sampling_type='particle')           
            snap.add_field((constants.gasPart, "MetallicityMassG3"), function=globalFields[4], force_override=True,take_log=True, display_name="Metal Mass", units="Msun",sampling_type='particle')
            snap.add_field((constants.gasPart, "MetallicityMass2"), function=globalFields[5], force_override=True,take_log=True, display_name="Metal Mass", units="Msun",sampling_type='particle')