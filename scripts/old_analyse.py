# =========================================
#  ANALYSIS OF GADGET3/4 AGORA SIMULATIONS
#  By Pablo Granizo Cuadrado
#  13/06/2025
# =========================================

# INITIALIZATION

# Import modules
import numpy as np
import yt
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.colors as col
import math
import bisect as bsct
import unyt
import h5py
import os.path
import sys
import copy
import io
import shutil
import argparse
import yt.extensions.astro_analysis.halo_analysis
from   mpl_toolkits.axes_grid1 import AxesGrid
from   yt.frontends.halo_catalog.data_structures import HaloDataset
from   yt.utilities.cosmology import Cosmology
from   matplotlib.offsetbox import AnchoredText
from   scipy.stats import kde
from   matplotlib  import rc_context
from   PIL import Image

# Ignore yt logs
yt.set_log_level(0)
yt.enable_parallelism()

# HELPING FUNCTIONS

# Get the index of the closest but bigger element in a list than your value 
def getClosestIdx(myList, myNumber):
        return bsct.bisect_left(myList, myNumber)

# Get the idx of maximum number in iterable
def maxIdx(arr):
    maxAux = 0
    for i,el in enumerate(arr):
        if el > arr[maxAux]:
            maxAux = i
    return maxAux

# Get the n maximum numbers idx in iterable (ONLY SEND COPIES)
def findNmax(arr, N):
    minV = min(arr)
    maxes = [0]*N
    for i in range(N):
        max1 = maxIdx(arr)
        maxes[i] = max1
        arr[max1] = minV
    return maxes

# Pads a number with 3 zeros (default) and makes it a string
def padN(n,pad=3):
    return str(n).zfill(pad)

# Transformations between scale factor and redshift
def getZ(a):
    return 1/a-1
def getA(z):
    return 1/(z+1)

# Calculate the actual center of a simulation snapshot
def findCenter(sim, snapshotN, lim=20):
    snap   = sim.ytFull[snapshotN]
    cutOff = snap.sphere("center",(lim,"kpc"))
    den    = np.array(cutOff["PartType0", "Density"].to("Msun/pc**3"))
    x      = np.array(cutOff["PartType0", "x"].to("pc"))
    y      = np.array(cutOff["PartType0", "y"].to("pc"))
    z      = np.array(cutOff["PartType0", "z"].to("pc"))
    cenIdx = maxIdx(den)
    cen    = np.array([x[cenIdx],y[cenIdx],z[cenIdx]])
    return (cen,cen/1e3*unyt.kpc)

# Centering calc2 (center of mass)
def findCenter2(sim, snapshotN, lim=20):
    snap   = sim.ytFull[snapshotN]
    cutOff = snap.sphere("center",(lim,"kpc"))
    cen    = cutOff.quantities.center_of_mass(use_gas=True, use_particles=False).in_units("pc")
    return (cen.d,cen.d/1e3*unyt.kpc)

# Centering calc3 (like AGORA Paper)
def findCenter3(sim, snapshotN):
    snap      = sim.ytFull[snapshotN]

    v, cen1   = snap.find_max(("gas", "density"))
    bigCutOff = snap.sphere(cen1, (30.0, "kpc"))
    cen2      = bigCutOff.quantities.center_of_mass(use_gas=True, use_particles=False).in_units("kpc")
    cutOff    = snap.sphere(cen2,(1.0,"kpc"))
    cen       = cutOff.quantities.max_location(("gas", "density"))
    center    = np.array([cen[1].d,cen[2].d,cen[3].d])*1e3

    return (center,center/1e3*unyt.kpc)

# Centering calc4 (like AGORA Paper, for CosmoRun)
def findCenter4(sim, idx):
    snap     = sim.ytFull[idx]
    projPath = projListPath
    f        = np.loadtxt(projPath,skiprows=4)
    idx0     = getClosestIdx(f[:,0],0.99999) # Finds end of first projection
    tIdx     = getClosestIdx(f[0:idx0,1],sim.snap[idx].a)

    cen1     = np.array([f[tIdx,3],f[tIdx,4],f[tIdx,5]]) # In Mpc
    center   = snap.arr([cen1[0], cen1[1], cen1[2]],'code_length')
    sp       = snap.sphere(center, (2,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (2*0.5,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (2*0.25,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    center   = center.to("code_length")

    cen1     = np.array([center[0].d,center[1].d,center[2].d]) 

    return   (cen1,center)

# Centering calc5 (AGORA code shortcut, for isolated)
def findCenter5(sim, snapshotN):
    snap   = sim.ytFull[snapshotN]
    cen    = snap.arr([6.184520935812296e+21, 4.972678132728082e+21, 6.559067311284074e+21], 'cm')
    cen    = cen.to("pc")
    cent   = np.array([cen[0].d,cen[1].d,cen[2].d])
    return (cen,cen/1e3*unyt.kpc)

# Centering calc6 (just 0 0 0)
def findCenter6(sim, snapshotN):
    cen    = np.array([0,0,0])
    return (cen,cen/1e3*unyt.kpc)

# Centering calc7 (like 4, but expanded in scope)
def findCenter7(sim, idx):
    snap     = sim.ytFull[idx]
    projPath = projListPath
    f        = np.loadtxt(projPath,skiprows=4)
    idx0     = getClosestIdx(f[:,0],0.99999) # Finds end of first projection
    tIdx     = getClosestIdx(f[0:idx0,1],sim.snap[idx].a)

    cen1     = np.array([f[tIdx,3],f[tIdx,4],f[tIdx,5]]) # In Mpc
    center   = snap.arr([cen1[0], cen1[1], cen1[2]],'code_length')
    sp       = snap.sphere(center, (40,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (10,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (2,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (2*0.5,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (2*0.25,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    center   = center.to("code_length")

    cen1     = np.array([center[0].d,center[1].d,center[2].d]) 

    return   (cen1,center)

# Print a hdf5 file structure
def h5printR(item, leading = ''):
    for key in item:
        if isinstance(item[key], h5py.Dataset):
            print(leading + key + ': ' + str(item[key].shape))
        else:
            print(leading + key)
            h5printR(item[key], leading + '  ')
def h5print(filename):
    h5printR(filename, '  ')

# Gets the idx of the snapshot with the time closest to the one inputted
def tToIdx(sim,time):
    dist = abs(sim.snap[0].time-time)
    idx = 0
    for i in range(len(sim.snap)):
        nDist = abs(sim.snap[i].time-time)
        if nDist < dist:
            dist = nDist
            idx = i
    return idx

# Gets the idx of the snapshot with the redshift closest to the one inputted
def zToIdx(sim,z):
    dist = abs(sim.snap[0].z-z)
    idx = 0
    for i in range(len(sim.snap)):
        nDist = abs(sim.snap[i].z-z)
        if nDist < dist:
            dist = nDist
            idx = i
    return idx

# Calculates the critical density based on cosmology and redshift
def getCritRho(co,z):
    return co.critical_density(z).to("Msun/kpc**3")
def getCritRho200(co,z):
    return getCritRho(co,z)*200
# Calculates the mean density based on cosmology and redshift
def getMeanRho(co,z):
    rho_crit0 = getCritRho(co,0)
    rho_mean0 = co.omega_matter * rho_crit0
    return rho_mean0 * (1 + z)**3
def getMeanRho200(co,z):
    return getMeanRho(co,z)*200
# Calculates the density at virial radius according to Paper IV (based on the factor by Bryan & Norman (1998))
def getVirRho(co,z):
    Hz = co.hubble_parameter(z)
    H0 = co.hubble_parameter(0.0)
    Ez = (Hz / H0)
    # Matter fraction at z
    Omega_z = (co.omega_matter * (1 + z)**3 / Ez**2)
    x = Omega_z - 1.0
    # Δc fits from the literature:
    #   • Flat‐curvature (ΩR = 0):   Δc = 18π² + 82x – 39x²
    #   • No‐Λ universe  (ΩΛ = 0):   Δc = 18π² + 60x – 32x²
    Delta_c     = 18*np.pi**2 + 82*x - 39*x**2
    #Delta_c     = 18*np.pi**2 + 60*x - 32*x**2
    return Delta_c*getCritRho(co,z)

# Gets data for multiple particles
def getData(selector,field,parts,units = 0):
    results = []
    for part in parts:
        if units == 0:
            results = np.concatenate((results,selector[(part,field)]))
        else:
            results = np.concatenate((results,selector[(part,field)].to(units)))
    return results

# Calculates the virial radius at present redshift via multiple methods
def getRvir(sim,idx,method="Vir",rvirlim=500):
    f.write(f"    Initiating virial radius finder for snapshot at {idx} using method {method} and limited by {rvirlim}\n")
    co = yt.utilities.cosmology.Cosmology(hubble_constant=0.702, omega_matter=0.272,omega_lambda=0.728, omega_curvature=0.0)
    methodDict = {"Crit":getCritRho200,"Mean":getMeanRho200,"Vir":getVirRho}
    snapshot   = sim.ytFull[idx]
    targetDen  = float(methodDict[method](co,sim.snap[idx].z))
    sp         = snapshot.sphere(sim.snap[idx].ytcen,(500,"kpc"))
    allMass    = getData(sp, "particle_mass", sim.snap[idx].pType, units = "Msun")
    allR       = getData(sp, "particle_position_spherical_radius", sim.snap[idx].pType, units = "kpc")
    #allMass    = sp[("all","particle_mass")].in_units("Msun")
    #allR       = sp[("all","particle_position_spherical_radius")].in_units("kpc")
    f.write(f"    Sorting particles by radius...\n")
    idx   = np.argsort(allR)
    mSort = np.array(allMass)[idx]
    rSort = np.array(allR)[idx]
    cumM  = np.cumsum(mSort)
    denR  = cumM/(4/3*np.pi*rSort**3) 

    idxAtVir = np.argmin(np.abs(denR-targetDen))
    f.write(f"    Found rvir: {rSort[idxAtVir]:.3f} enclosing {cumM[idxAtVir]:.3E} Msun, with predicted {targetDen*(4/3*np.pi*rSort[idxAtVir]**3):.3E}\n")
    return rSort[idxAtVir]

# SIMULATION DATA OBJECTS
from dataclasses import dataclass, field
@dataclass
class particleList:
    # Global
    x:  np.ndarray = None
    y:  np.ndarray = None
    z:  np.ndarray = None
    m:  np.ndarray = None
    vx: np.ndarray = None
    vy: np.ndarray = None
    vz: np.ndarray = None
    # Gas
    d:  np.ndarray = None
    h:  np.ndarray = None
    t:  np.ndarray = None
    mt: np.ndarray = None

@dataclass
class snapshot:
    idx    : int
    ytIdx  : int
    time   : float
    z      : float
    a      : float

    p      : list             = field(default_factory=list)
    center : np.ndarray       = None
    ytcen  : yt.units.YTArray = None
    rvir   : float            = None
    cosmo  : int              = None
    pType  : list             = field(default_factory=list)

@dataclass
class simul:
    name  : str
    cosmo : int  = None
    ytFile: ...  = None
    ytFull: list = field(default_factory=list)
    snap  : list = field(default_factory=list)

# Load particle list for one snapshot
def loadParticles(sim,snapI,pI,typeP):
    # Load general
    sim.snap[snapI].p[pI]    = particleList()
    sim.snap[snapI].p[pI].x  = np.array(sim.ytFull[snapI].r[typeP, "Coordinates"].to("pc")  )[:,0]-sim.snap[snapI].center[0]
    sim.snap[snapI].p[pI].y  = np.array(sim.ytFull[snapI].r[typeP, "Coordinates"].to("pc")  )[:,1]-sim.snap[snapI].center[1]
    sim.snap[snapI].p[pI].z  = np.array(sim.ytFull[snapI].r[typeP, "Coordinates"].to("pc")  )[:,2]-sim.snap[snapI].center[2]
    sim.snap[snapI].p[pI].m  = np.array(sim.ytFull[snapI].r[typeP, "Masses"     ].to("Msun"))
    sim.snap[snapI].p[pI].vx = np.array(sim.ytFull[snapI].r[typeP, "Velocities" ].to("km/s"))[:,0]
    sim.snap[snapI].p[pI].vy = np.array(sim.ytFull[snapI].r[typeP, "Velocities" ].to("km/s"))[:,1]
    sim.snap[snapI].p[pI].vz = np.array(sim.ytFull[snapI].r[typeP, "Velocities" ].to("km/s"))[:,2]

    # Load specific
    if typeP == "PartType0":
        sim.snap[snapI].p[pI].d  = np.array(sim.ytFull[snapI].r[typeP, "Density"].to("Msun/pc**3"))
        sim.snap[snapI].p[pI].h  = np.array(sim.ytFull[snapI].r[typeP, "SmoothingLength"].to("pc"))
        if (typeP,"Temperature") in sim.ytFull[snapI].field_list:
            sim.snap[snapI].p[pI].t  = np.array(sim.ytFull[snapI].r[typeP, "Temperature"].to("K"))
        if (typeP,"Metallicity") in sim.ytFull[snapI].field_list:
            sim.snap[snapI].p[pI].mt = np.array(sim.ytFull[snapI].r[typeP, "Metallicity"])

# Load center for one snapshot
def loadCenters(sim,idx,overrideCenter=0,fun="3"):
    methodDict = {"1":findCenter,"2":findCenter2,"3":findCenter3,"4":findCenter4,"5":findCenter5,"6":findCenter6,"7":findCenter7}
    if overrideCenter == 1:
        cen = sim.ytFull[idx].domain_center
        sim.snap[idx].center = np.array([cen[0].d,cen[1].d,cen[2].d])*1e3
    else:
        cens = methodDict[fun](sim,idx)
        sim.snap[idx].center = cens[0]
        sim.snap[idx].ytcen  = cens[1]

# Default list with particle types (ideally this would automatically be figure out). Gas, DM and Stars
defListPart = ["PartType0","PartType1","PartType4"]
# Checks is particle types exist in a snapshot
def getPtypes(sim,idx,defList):
    # Heisenbug! (we force YT to load the particle types from the snapshots, without this it returns None)
    _ = sim.ytFull[idx].particle_type_counts
    return [part for part in defList if part in sim.ytFull[idx].particle_types]

# Load one snapshot
def loadSnapshot(cosmological,sim,idx,trueIdx,overrideCenter=0,loadAllP=0,verboseLvl=1,sameCenter=0,centerFun="6"):
    # Initialize the snapshot struct
    a = 1
    z = 0
    if cosmological:
        z = float(sim.ytFull[idx].parameters["Redshift"])
        a = float(sim.ytFull[idx].parameters["Time"])

    sim.snap[idx] = snapshot(idx,trueIdx,float(sim.ytFull[idx].current_time.in_units("Myr")),z,a)
    sim.snap[idx].cosmo = cosmological
    sim.snap[idx].pType = getPtypes(sim,idx,defListPart)
    
    if verboseLvl > 0: f.write(f"  - Loading snapshot {idx} true index {sim.snap[idx].ytIdx} at {sim.snap[idx].time} Myr \n")
    if cosmological:
        f.write(f"    With Redshit: {sim.snap[idx].z} Scale: {sim.snap[idx].a} \n")
    if sameCenter != 0:
        sim.snap[idx].center = np.array(sameCenter)
    else:
        loadCenters(sim,idx,overrideCenter=overrideCenter,fun=centerFun)
    if verboseLvl > 1: f.write(f"    Center initialized at {sim.snap[idx].center} pc \n")
    if verboseLvl > 2: f.write(f"    YT Center initialized at {sim.snap[idx].ytcen} \n")

    # Getting virial radius after calculating center
    if cosmological: sim.snap[idx].rvir = getRvir(sim,idx)
    
    # Load each type of particle
    if loadAllP:
        nParts = sum(np.array(sim.ytFull[idx].parameters["NumPart_ThisFile"]) > 0)
        typeP = ["PartType"+str(i) for i in range(nParts)]
        sim.snap[idx].p = [None]*len(typeP)
        for i in range(len(typeP)):
            f.write(f"    - Loading {typeP[i]} \n")
            loadParticles(sim,idx,i,typeP[i])

# Load a full simulation
def loadSim(sim,ytData,allowedSnaps=0,overrideCenter=0,loadAllP=0,cosmological=0,
            verboseLvl=1,sameCenter=0,centerDefs=0):
    f.write(f"Loading from yt file: {ytData} \n")
    f.write(f"  Loading YT snapshots into list \n")

    sim.ytFile = ytData

    # Load only some snapshots if desired
    if allowedSnaps != 0:
        sim.ytFull = [None]*len(allowedSnaps)
        f.write(f"  Loading only snapshots: {allowedSnaps} \n")
        for i in range(len(allowedSnaps)):
            sim.ytFull[i] = ytData[allowedSnaps[i]]
            if verboseLvl > 1: f.write(f"    Loaded yt snapshot {i} from file: {ytData[allowedSnaps[i]]} \n")
    else:
        sim.ytFull = [None]*len(ytData)
        ytStr = [None]*len(ytData)
        for i in range(len(ytData)):
            #ytStr[i] = str(ytData[i].filename)[:]
            ytStr[i] = 0
        f.write(f"  Loading all snapshots found {ytStr} \n")    
        for i in range(len(ytData)):
            sim.ytFull[i] = ytData[i]
            if verboseLvl > 1: f.write(f"    Loaded yt snapshot {i} from file: {ytData[i]} \n")

    sim.cosmo = cosmological
    sim.snap  = [None]*len(sim.ytFull)
    centerAlg = centerDefs[1] if cosmological else centerDefs[0] 

    for i in range(len(sim.ytFull)):
        if not cosmological:
            sim.ytFull[i].hubble_constant  = 0.71
            sim.ytFull[i].omega_lambda     = 0.73
            sim.ytFull[i].omega_matter     = 0.27
            sim.ytFull[i].omega_curvature  = 0.0
            sim.ytFull[i].current_redshift = 0.0

        trueIdx = i
        if allowedSnaps != 0:
            trueIdx = allowedSnaps[i]

        loadSnapshot(cosmological,sim,i,trueIdx,loadAllP=loadAllP,verboseLvl=verboseLvl,
                     sameCenter=sameCenter,centerFun=centerAlg,overrideCenter=overrideCenter)

## LOADING FUNCTIONS
verbosity        = 10
gadgetUnitsIso   = {'UnitLength_in_cm'         : 3.08568e+21,
                    'UnitMass_in_g'            : 1.989e+43,
               	    'UnitVelocity_in_cm_per_s' : 100000}

gadgetUnitsCosmo = {'UnitLength_in_cm'         : 3.0868e+24,
                    'UnitMass_in_g'            : 1.989e+43,
                    'UnitVelocity_in_cm_per_s' : 100000}

# Finds if a snapshot is cosmological or not
def isCosmological(path):
    with h5py.File(path, 'r') as f:
        # Read the parameter file loaded in the snapshot, looking for the attribute that sets comoving simulations
        if "Parameters" in f:
            attrs = f["Parameters"].attrs
            if "ComovingIntegrationOn" not in attrs:
                f.write(f"ERROR: ComovingIntegrationOn attribute was not detected in this snapshot. Either the snapshot is corrupt, or you " \
                        "should yell at the dev of this script so they implement further snapshot compatibility with what you are trying to load.\n")
            if attrs["ComovingIntegrationOn"] > 0.1:
                return True
        else: # Happens in G3 snapshots
            attrs = f["Header"].attrs
            if "OmegaLambda" in attrs:
                return True
    return False

# All powerful load function
def load(name, path, centerDefs, boundbox = 0, verboseLvl = verbosity,
          overrideCenter = 0, allowedSnaps = 0, sameCenter = 0, loadP = 0):
    simulation = simul(name)
    # Load yt file
    ytDataset   = None

    # Figure out if snapshots are inside output dir
    if os.path.isdir(os.path.join(path,"output")):
        path = os.path.join(path,"output")
    
    # Figure out if they are inside multiple directories or not
    snapdirs = [name for name in os.listdir(path) if name.startswith("snapdir") and os.path.isdir(os.path.join(path,name))]
    if len(snapdirs) > 0:
        # Finds the snapshots on the first directory (sufficent for figuring out if it's cosmological and in pieces or not)
        pathDirs = [os.path.join(path,snapdirectory) for snapdirectory in snapdirs]
        snapshot_files = [os.path.join(pathDirs[0],file) for file in os.listdir(pathDirs[0]) if os.path.isfile(os.path.join(pathDirs[0], file)) and file.startswith("snapshot") and file.endswith(".hdf5")]
        cosmological = isCosmological(snapshot_files[0])

        # Find out if the snapshot has multiple snapshots or not
        isMultiple = False
        for fileS in snapshot_files:
            if fileS.endswith(".0.hdf5"):
                isMultiple = True
        suffix = "snapdir_???/snapshot_???.hdf5" if not isMultiple else "snapdir_???/snapshot_???.0.hdf5"
        pathPattern = os.path.join(path,suffix)
    else:
        # Finds all snapshot files and uses it to discover if it's cosmological or not
        snapshot_files = [os.path.join(path,file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file)) and file.startswith("snapshot") and file.endswith(".hdf5")]
        cosmological = isCosmological(snapshot_files[0])

        # Find out if the snapshot has multiple snapshots or not
        isMultiple = False
        for fileS in snapshot_files:
            if fileS.endswith(".0.hdf5"):
                isMultiple = True
        suffix = "snapshot_???.hdf5" if not isMultiple else "snapshot_???.0.hdf5"
        pathPattern = os.path.join(path,suffix)

    # Set the units and use our hard earned path pattern to load the whole Gadget timeseries dataset
    unitBase = gadgetUnitsCosmo if cosmological else gadgetUnitsIso
    f.write(f"\n- Started loading simulation {name} \n")
    f.write(f"  With cosmology {cosmological}\n")
            
    if not boundbox:
        ytDataset = yt.load(pathPattern,unit_base=unitBase)
    else:
        ytDataset = yt.load(pathPattern,unit_base=unitBase,bbox = boundbox)
    # Load simulation
    loadSim(simulation, ytDataset, verboseLvl=verboseLvl, overrideCenter=overrideCenter, cosmological = cosmological,
            allowedSnaps=allowedSnaps, centerDefs=centerDefs, sameCenter=sameCenter, loadAllP = loadP)

    return simulation

## PARSE USER VARIABLES
# Set input variable behaviour
parser = argparse.ArgumentParser(description="Analyse a list of simulations")
parser.add_argument("-s", dest="simPath", nargs="+", default=[], help="Every simulation dir name that is going to be projected into a movie." \
" The path is taken as simDir/(input path)")

parser.add_argument("-names", dest="simName", nargs="+", default=[], help="Every simulation name that is going to be projected into a movie." \
" The default is taking the same name as the path. If you want to maintain this default behaviour for a particular simulation, set its name as -1")

parser.add_argument("-snaps", dest="snapshots", nargs="+", type=int, default=[], action="append", help="Set the snapshot numbers to be analysed." \
" The default is to analyse every one. If you want to enable this for a particular simulation, set it as -1.")

parser.add_argument("-zt", dest="ztime", nargs="+", type=float, default=[], help="Sets times or redshifts to find and analyse." \
" This supersedes -snaps and is set off by default. It will find the closest snapshot to the ones you set for all simulations.")

parser.add_argument("-a", dest="analysis", nargs="+", default=["NSFF"], help="List of analysis routines to be run.\n" \
"NSFF: Runs all the NSFF plots on Paper II of the AGORA collaboration")

parser.add_argument("-sdir", dest="simDir", default="", help="Directory where all the simulation directories sit. Default is the same one as the script")

parser.add_argument("-center", dest="centerAlg", nargs=2, default=["3","7"], help="Code for the default centering algorithm used in isolated and cosmological simulations (in that order)")

parser.add_argument("-log", dest="logFile", default="analysisLog.txt", help="Path and name of the log file")

parser.add_argument("-save", dest="saveFile", default="autoAnalysisFigures", help="Path and name of the results directory")

parser.add_argument("-proj", dest="projFile", default="outputlist_projection.txt", help="Path to file with projection timestamps (for centering calculations). " \
"Defaults to a file with name outputlist_projection.txt in the same directory as the analysis script")

# Parse the user input
args = parser.parse_args()
simsPath      = args.simPath
simsName      = args.simName
snapshots     = args.snapshots
ztList        = args.ztime
analysisList  = args.analysis
simDir        = args.simDir
centerAlg     = args.centerAlg
logPath       = args.logFile
savePath      = args.saveFile
projListPath  = args.projFile

# Create a logging file
f = open(logPath, "w", buffering=1, encoding="utf-8")

# Terminate if incompatible inputs are encountered
if snapshots != [] and len(simsPath) != len(snapshots):
    f.write(f"ERROR: You gave {len(simsPath)} simulations but {len(snapshots)} snapshot lists")
    sys.exit(1)

# Handle defaults
nSims = len(simsPath)
os.makedirs(savePath, exist_ok=True)
if simsName  == []: simsName  = [-1]*nSims
if snapshots == []: snapshots = [[-1] for _ in range(nSims)]

for i in range(nSims):
    if simsName[i]  == -1  : simsName[i]  = simsPath[i]
    if snapshots[i] == [-1]: snapshots[i] = 0 

# Write header of log file
f.write("Starting analysis with arguments:\n\n")
f.write(f"  - Simulation dirs      : {simsPath}\n")
f.write(f"  - Simulation names     : {simsName}\n")
f.write(f"  - Simulation snapshots : {snapshots}\n")
f.write(f"  - Using times/redshifts: {ztList}\n")
f.write(f"  - Analysis performed   : {analysisList}\n")
f.write(f"  - Simulation directory : {simDir}\n")
f.write(f"  - Centering Algorithms : {centerAlg}\n")
f.write(f"  - Logging file path    : {logPath}\n")
f.write(f"  - Output folder        : {savePath}\n")
f.write(f"  - Center Data Path     : {projListPath}\n")

f.write(f"\n------------- \n")
f.write(f" -- SETUP -- \n")
f.write(f"------------- \n")

## LOADING SIMULATIONS
f.write(f"\nLoading simulations... \n")
sims = [None]*nSims
for i in range(nSims):
    sims[i] = load(simsName[i],os.path.join(simDir,simsPath[i]), centerDefs=centerAlg, allowedSnaps = snapshots[i])

## OPTIONS
f.write(f"\nLoading Finished! Setting up plotting options... \n")
# Figure options
figSize   = 8  # Fig size for plt plots
ytFigSize = 12 # Window size for the yt plot
fontSize  = 12 # Fontsize in yt plots

# Modified color maps
starCmap = copy.copy(plt.get_cmap('autumn'))
starCmap.set_bad(color='k') # (log(1e4) - log(1e1)) / (log(1e6) - log(1e1)) = 0.6
starCmap.set_under(color='k')
mColorMap = copy.copy(plt.get_cmap('algae'))
mColorMap.set_bad(mColorMap(0.347)) # (0.02041 - 0.01) / (0.04 - 0.01) = 0.347
mColorMap.set_under(mColorMap(0))
mColorMap2 = copy.copy(plt.get_cmap('algae'))
mColorMap2.set_bad(mColorMap2(0))
mColorMap2.set_under(mColorMap2(0))

# Scaffolding options (Global parameters for how the plotting functions below should operate)
verboseLevel    = 15                    # How detailed is the console log, higher is more detailed
showErrorGlobal = 0                     # If the dispersion is plotted or not
errorLimGlobal  = (-1,1)                # Y limits on the dispersion plot
showAll         = False                  # If the plots are showed in the notebook or not
saveAll         = True                 # If the plots are saved to disk or not

# Analysis options
figWidth       = 30          # Kpc (Big box)
buffSize       = 800         # N   (Default bin number for coordinate-type histograms) 
youngStarAge   = 20          # Myr (Age to be considered for SFR density calculations)
lowResMock     = 750         # pc  (Resolution of the mock observations)
starFigSize    = 80          # kpc (Size of star map)
starBufferSize = 400         # N   (Buffer size of star map)
gasPart        = "PartType0" # Field for gas SPH particles 
starPart       = "PartType4" # Field for star particles
zSolar         = 0.0204      # Solar metallicity

# ADD NEW FIELDS
f.write(f"\nCreating new YT fields... \n")
# Types of simulations for array ordering
snapArr = []
snhtArr = []
centArr = []
for i in range(len(sims)):
    for j in range(len(sims[i].ytFull)):
        snapArr.append(sims[i].ytFull[j])
        snhtArr.append(sims[i].snap[j])
        centArr.append(sims[i].snap[j].ytcen)

## Simple fields
def _density_squared(field, data):  
	return data[(gasPart, "Density")]**2
def _res(field, data):  
    return (data[(gasPart, "Masses")]/data[(gasPart, "Density")])**(1./3.)
def _inv_vol_sq(field, data):  
    return (data[(gasPart, "Masses")]/data[(gasPart, "Density")])**(-2)
def _particle_position_cylindrical_z_abs(field, data):
    return np.abs(data[(gasPart, "particle_position_cylindrical_z")])
def _den_low_res(field,data):
    trans = np.zeros(data[(gasPart, "particle_mass")].shape)
    ind = np.where(data[(gasPart, "particle_mass")] > 0) 
    trans[ind] = data[(gasPart, "particle_mass")][ind].in_units('Msun').d/(lowResMock**2)
    return data.ds.arr(trans, "Msun/pc**2").in_base(data.ds.unit_system.name)
def _stardensity(field, data):  
    return data[(starPart, "Masses")].in_units('Msun')/yt.YTArray(starFigSize/starBufferSize*starFigSize/starBufferSize*1000.*1000.,'pc**2')   
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
    return data[(gasPart, "Temperature")].in_units("K")
def _temperature_GADGET3(field, data):
    HYDROGEN_MASSFRAC = 0.76
    gamma=5.0/3.0
    GAMMA_MINUS1=gamma-1.
    PROTONMASS=unyt.mp.in_units('g')
    BOLTZMANN=unyt.kb.in_units('J/K')
    u_to_temp_fac=(4.0 / (8.0 - 5.0 * (1.0 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1
    # Assume cosmic abundances
    gamma = 5.0/3.0
    T_over_mu_GADGET= data[gasPart, "InternalEnergy"].in_units('J/g')*(gamma-1)*unyt.mp.in_units('g')/unyt.kb.in_units('J/K')
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
        return (data[(part, 'Metallicity')]+yt.YTArray(2.041e-6,''))/zSolar
    def _metallicityMassG3(field,data):
        return (data[(part, 'MetallicityG3')]*zSolar*data[(part,'particle_mass')])
    def _metallicityMass2(field,data):
        return (data[(part, 'Metallicity')]*zSolar*data[(part,'particle_mass')])
    return (_elevation,_x_centered,_y_centered,_metallicityG3,_metallicityMassG3,_metallicityMass2)
# Star fields
def makeStarFields(snap,snht):
    co = yt.utilities.cosmology.Cosmology(hubble_constant=0.702, omega_matter=0.272,omega_lambda=0.728, omega_curvature=0.0)
    def _youngStar_Mass(field, data):  
        trans = np.zeros(data[(starPart, "particle_mass")].shape)
        # Selects stars younger than youngStarAge (in Myr)
        if not snht.cosmo:
            starAge = data[(starPart, "StellarFormationTime")]*1e3
            ind = np.where(starAge > (snap.current_time.in_units('Myr').d - youngStarAge)) 
        else:
            starAgeA  = data[(starPart, "StellarFormationTime")]
            cutOffAge = co.a_from_t(co.quan(snap.current_time.in_units('Myr').d - youngStarAge,"Myr"))
            ind = np.where(starAgeA > cutOffAge) 
      
        trans[ind] = data[(starPart, "particle_mass")][ind].in_units('code_mass')
        return data.ds.arr(trans, "code_mass").in_base(data.ds.unit_system.name)
    def _sfr_den_low_res(field,data):
        trans = np.zeros(data[(starPart, "particle_mass")].shape)
        # Selects stars younger than youngStarAge (in Myr)
        if not snht.cosmo:
            starAge = data[(starPart, "StellarFormationTime")]*1e3
            ind = np.where(starAge > (snap.current_time.in_units('Myr').d - youngStarAge)) 
        else:
            starAgeA  = data[(starPart, "StellarFormationTime")]
            cutOffAge = co.a_from_t(co.quan(snap.current_time.in_units('Myr').d - youngStarAge,"Myr"))
            ind = np.where(starAgeA > cutOffAge) 
            
        trans[ind] = data[(starPart, "particle_mass")][ind].in_units('Msun').d/((lowResMock/1e3)**2)/(youngStarAge*1e6)
        return data.ds.arr(trans, "Msun/yr/kpc**2").in_base(data.ds.unit_system.name)
    return (_youngStar_Mass,_sfr_den_low_res)

## Adding the fields
for i,snap in enumerate(snapArr):
    globalFields = makeGlobalFields(centArr[i],gasPart)
    snap.add_field((gasPart, "density_squared"), function=_density_squared, units="g**2/cm**6", sampling_type="particle",force_override=True)
    snap.add_field((gasPart, "elevation"), function=globalFields[0], units="kpc", sampling_type="particle", display_name="Elevation",force_override=True,take_log=False)
    snap.add_field((gasPart, "resolution"), function=_res, units="pc", sampling_type="particle", display_name="Resolution $\Delta$ x",force_override=True,take_log=True)
    snap.add_field((gasPart, "inv_volume_sq"), function=_inv_vol_sq, units="pc**(-6)", sampling_type="particle", display_name="Inv squared volume",force_override=True,take_log=True)
    snap.add_field((gasPart, "z_abs"), function=_particle_position_cylindrical_z_abs, take_log=False,units="cm",sampling_type="particle",force_override=True) 
    snap.add_field((gasPart, "x_centered"), function=globalFields[1],display_name="x",units="kpc",sampling_type="particle",force_override=True,take_log=False) 
    snap.add_field((gasPart, "y_centered"), function=globalFields[2],display_name="y",units="kpc",sampling_type="particle",force_override=True,take_log=False) 
    snap.add_field((gasPart, "den_low_res"), function=_den_low_res,display_name="Density",units="Msun/pc**2",sampling_type="particle",force_override=True,take_log=False) 
    if (gasPart,"InternalEnergy") in snap.field_list:
        snap.add_field((gasPart, 'TemperatureG3'), function=_temperature_GADGET3,force_override=True,sampling_type="particle", display_name="Temperature",take_log=False, units="K")
        snap.add_field((gasPart, 'TemperatureG3log'), function=_temperature_GADGET3,force_override=True,sampling_type="particle", display_name="Temperature",take_log=True, units="K")
    else:
        snap.add_field((gasPart, 'TemperatureG3'), function=_temperature,force_override=True,sampling_type="particle", display_name="Temperature",take_log=False, units="K")
        snap.add_field((gasPart, 'TemperatureG3log'), function=_temperature,force_override=True,sampling_type="particle", display_name="Temperature",take_log=True, units="K")

    # Star fields
    if starPart in snap.particle_types:
        globalFieldsStar = makeGlobalFields(centArr[i],starPart)
        starFields       = makeStarFields(snap,snhtArr[i])
        if (starPart,"StellarFormationTime") in snap.field_list: snap.add_field((starPart, "particle_mass_young_stars"),function=starFields[0],display_name="Young Stellar Mass",units="code_mass",sampling_type="particle",force_override=True,take_log=True)
        snap.add_field((starPart, "x_centered"), function=globalFieldsStar[1],display_name="x",units="kpc",sampling_type="particle",force_override=True,take_log=False) 
        snap.add_field((starPart, "y_centered"), function=globalFieldsStar[2],display_name="y",units="kpc",sampling_type="particle",force_override=True,take_log=False) 
        if (starPart,"StellarFormationTime") in snap.field_list: snap.add_field((starPart, "sfr_den_low_res"), function=starFields[1],display_name="SFR Density",units="Msun/yr/kpc**2",sampling_type="particle",force_override=True,take_log=False) 
        snap.add_field((starPart, "stardensity"), function=_stardensity, units="Msun/pc**2",display_name="Star density", sampling_type="local",force_override=True,take_log=True)
        # Metallicity fields
        if "Metallicity" in np.array(snap.field_list)[:,1]:
            snap.add_field((starPart, "MetallicityG3"), function=globalFieldsStar[3], force_override=True,take_log=True, display_name="Metallicity", units="",sampling_type='particle')
            snap.add_field((starPart, "MetallicityMassG3"), function=globalFieldsStar[4], force_override=True,take_log=True, display_name="Metal Mass", units="Msun",sampling_type='particle')
            snap.add_field((starPart, "MetallicityMass2"), function=globalFieldsStar[5], force_override=True,take_log=True, display_name="Metal Mass", units="Msun",sampling_type='particle')

    # Metallicity fields
    if "Metallicity" in np.array(snap.field_list)[:,1]:
        snap.add_field((gasPart, "MetallicityG3"), function=globalFields[3], force_override=True,take_log=True, display_name="Metallicity", units="",sampling_type='particle')           
        snap.add_field((gasPart, "MetallicityMassG3"), function=globalFields[4], force_override=True,take_log=True, display_name="Metal Mass", units="Msun",sampling_type='particle')
        snap.add_field((gasPart, "MetallicityMass2"), function=globalFields[5], force_override=True,take_log=True, display_name="Metal Mass", units="Msun",sampling_type='particle')
    
f.write(f"\nFields created, setting up plotting functions... \n")

# PLOTTING MACRO FUNCTIONS (rerun whenever config options are changed for new defaults)
# Return the frame for an animation
def saveFrame(figure,verbose):
    buf = io.BytesIO()
    figure.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
    buf.seek(0)
    img = np.array(Image.open(buf))
    if verbose > 8: f.write(f"  Figure saved to buffer\n")
    buf.close()
    plt.close(figure)
    return img

def setLegend(uAx,sims,idx):
    timeInfo = [None]*len(sims)
    for i in range(len(sims)):
        timeInfo[i] = " t="+str(round(sims[i].snap[idx[i]].time))+" Myr"
        if sims[i].cosmo: timeInfo[i] = " z="+str(round(sims[i].snap[idx[i]].z,2))
    uAx.legend([sims[i].name+timeInfo[i] for i in range(len(sims))])

def handleFig(figure, switches, message, saveFigPath, verbose):
    # Shows figure
    if switches[0]:
        if verbose > 10: f.write(f"  Showing to screen\n")
        plt.show()

    # Return the frame for an animation
    if switches[1]:     
        return saveFrame(figure,verbose)
    
    # Saves figure with message path as title
    if switches[2]: 
        fullPath = os.path.join(savePath,"placeholder.png")
        if message != 0:
            fullPath = os.path.join(savePath,message.replace(" ","_")+".png")
        elif saveFigPath == 0:
            if verbose > 3: f.write(f"WARNING: TITLE NOT SPECIFIED FOR THIS FIGURE, PLEASE SPECIFY A TITLE\n")

        if saveFigPath != 0: fullPath = os.path.join(saveFigPath,message.replace(" ","_")+".png")
        if verbose > 8: f.write(f"  Saving figure to {fullPath}\n")
        figure.savefig(fullPath, bbox_inches='tight', pad_inches=0.03, dpi=300)
        plt.close(figure)

# Plot a cool multipanel for multiple simulations and/or projections
def ytMultiPanel(sims, idx, zField = ["Density"], axisProj = 0, part = gasPart, zFieldUnit = "g/cm**2", cM = "algae", takeLog=1, zFieldLim = [1.5e-4,1e-1], wField=0, zWidth=figWidth,bSize=buffSize,
                flipOrder=0,
                verbose=verboseLevel, plotSize=ytFigSize, saveFig=saveAll, saveFigPath=0, showFig=showAll, message=0, fsize=fontSize, animate=0):
        
    if message != 0: f.write(f"\n{message}\n")
    # Option setup
    numP = len(zField)
    numS = len(sims)
    if isinstance(axisProj  ,list) == False: axisProj   = [axisProj  ]*numS
    if isinstance(part      ,list) == False: part       = [part      ]*numP
    if isinstance(zFieldUnit,list) == False: zFieldUnit = [zFieldUnit]*numP
    if isinstance(cM        ,list) == False: cM         = [cM        ]*numP
    if isinstance(takeLog   ,list) == False: takeLog    = [takeLog   ]*numP
    if isinstance(wField    ,list) == False: wField     = [wField    ]*numP
    if isinstance(zWidth    ,list) == False: zWidth     = [zWidth    ]*numP
    if isinstance(zFieldLim ,list) == False: zFieldLim  = [zFieldLim ]*numP
    if isinstance(bSize     ,list) == False: bSize      = [bSize     ]*numP
    if not all(isinstance(lIdx, list) for lIdx in zFieldLim):
        zFieldLimAux = [None]*numP
        for j in range(numP): zFieldLimAux[j] = zFieldLim 
        zFieldLim = zFieldLimAux

    rowIter = zField
    colIter = sims
    if flipOrder:
        rowIter = sims
        colIter = zField
    # Panel fig setup
    panelSize = (len(rowIter), len(colIter))
    panelFig = plt.figure()
    loc = "right" if not flipOrder else "bottom"
    panelGrid = AxesGrid(panelFig,(0,0,1,1),nrows_ncols=panelSize,axes_pad=0.02,label_mode="1",share_all=False,cbar_location=loc,cbar_mode="edge",cbar_size="5%",cbar_pad="2%")

    # Loading snapshots
    snapArr  = [sims[i].ytFull[idx[i]] for i in range(numS)]
    titleArr = [sims[i].name           for i in range(numS)]

    for i,snap in enumerate(snapArr):
        if verbose > 9: f.write(f"  - Projecting {sims[i].name} Time {sims[i].snap[idx[i]].time:.1f} Redshift {sims[i].snap[idx[i]].z:.2f} Axis {axisProj[i]}\n")
        for j,pField in enumerate(zField):
            iterRow, iterCol = (j,i)
            if flipOrder: iterRow, iterCol = (i,j)

            if verbose > 10: f.write(f"    Projecting field {pField} Particle {part[j]} Weight {wField[j]} Width {zWidth[j]} Unit {zFieldUnit[j]} Lim {zFieldLim[j]}\n")
            # Setup projection of pField of snap
            if takeLog[j] == 0: snap.field_info[(part[j], pField)].take_log = False
            if wField[j] != 0:
                fig1 = yt.ProjectionPlot(snap, axisProj[i], (part[j], pField), window_size=plotSize, weight_field=(part[j],wField[j]), fontsize=fsize, center=sims[i].snap[idx[i]].ytcen)
            else:
                if part[j]=="PartType4":
                    fig1 = yt.ParticleProjectionPlot(snap, axisProj[i], (part[j], pField), window_size=plotSize, depth=(zWidth[j],"kpc"), fontsize=fsize, center=sims[i].snap[idx[i]].ytcen)
                else:
                    fig1 = yt.ProjectionPlot(snap, axisProj[i], (part[j], pField), window_size=plotSize, fontsize=fsize, center=sims[i].snap[idx[i]].ytcen)
            fig1.set_width(zWidth[j],"kpc")
            if zFieldUnit[j] != 0: fig1.set_unit((part[j], pField), zFieldUnit[j])
            if not(zFieldLim[j][0] == 0 and zFieldLim[j][1]  == 0): fig1.set_zlim((part[j], pField), zmin=zFieldLim[j][0], zmax=zFieldLim[j][1])
            fig1.set_cmap(field=(part[j], pField), cmap=cM[j])
            fig1.set_buff_size(bSize[j])
            if ((not flipOrder) and iterRow == 0) or (flipOrder and iterCol == 0):  fig1.annotate_timestamp(redshift=True)

            # Transfers yt plot to plt axes and renders the figure
            fullPlot        = fig1.plots[part[j], pField]
            fullPlot.figure = panelFig
            fullPlot.axes   = panelGrid[iterCol+iterRow*len(colIter)].axes
            
            fullPlot.cax = panelGrid.cbar_axes[iterRow]
            if flipOrder: fullPlot.cax = panelGrid.cbar_axes[iterCol]

            if verbose > 11: f.write(f"    Rendering\n")
            fig1._setup_plots()

            if ((not flipOrder) and iterRow == 0) or (flipOrder and iterCol == 0): 
                nameTag = AnchoredText(titleArr[i], loc=2, prop=dict(size=9), frameon=True)
                panelGrid[iterCol+iterRow*len(colIter)].axes.add_artist(nameTag)

    handleFig(panelFig,[showFig,animate,saveFig],message,saveFigPath,verbose)
    
# Plots field projections, defaults to standard density projection
def ytProjPanel(simArr, idxArr, verbose=verboseLevel, plotSize=ytFigSize, saveFig=saveAll, saveFigPath=0, showFig=showAll,
                message=0, twoAxis=True, axisProj = [2,0], part = "PartType0", bSize=buffSize, zField = "Density",
                zFieldUnit = "g/cm**2", cM = "algae",takeLog=1, zFieldLim = (1.5e-4,1e-1), zWidth=figWidth, fsize=fontSize,
                wField=0, ovHalo=0, animate=0):
        
    if message != 0: f.write(f"\n{message}\n")
    # Option setup
    axNum = 1
    if twoAxis: axNum = 2

    # Panel fig setup
    panelSize = (axNum, len(simArr))
    panelFig = plt.figure()
    panelGrid = AxesGrid(panelFig,(0,0,1,1),nrows_ncols=panelSize,axes_pad=0.1,label_mode="1",share_all=True,cbar_location="right",cbar_mode="single",cbar_size="5%",cbar_pad="2%")

    # Loading snapshots
    snapArr  = [simArr[i].ytFull[idxArr[i]] for i in range(len(simArr))]
    titleArr = [simArr[i].name              for i in range(len(simArr))]

    # Start of the fig making
    if verbose > 9: f.write(f"  Setup complete - Starting fig making for {zField}\n")
    for i,snap in enumerate(snapArr):
        if verbose > 9: f.write(f"  - Projecting {simArr[i].name} at time {simArr[i].snap[idxArr[i]].time:.1f} Myr Redshift {simArr[i].snap[idxArr[i]].z:.2f}\n")

        # Sets plotting options as detailed
        if verbose > 11: f.write(f"    Projecting in axis {axisProj[0]}\n")
        if takeLog == 0: snap.field_info[(part, zField)].take_log = False
        if wField != 0:
            fig1 = yt.ProjectionPlot(snap, axisProj[0], (part, zField), window_size=plotSize, weight_field=(part,wField), fontsize=fsize, center=simArr[i].snap[idxArr[i]].ytcen)
        else:
            if part=="PartType4":
                fig1 = yt.ParticleProjectionPlot(snap, axisProj[0], (part, zField), window_size=plotSize, depth=(zWidth,"kpc"), fontsize=fsize, center=simArr[i].snap[idxArr[i]].ytcen)
            else:
                fig1 = yt.ProjectionPlot(snap, axisProj[0], (part, zField), window_size=plotSize, fontsize=fsize, center=simArr[i].snap[idxArr[i]].ytcen)
        fig1.set_width(zWidth,"kpc")
        if zFieldUnit != 0: fig1.set_unit((part, zField), zFieldUnit)
        if zFieldLim != 0:  fig1.set_zlim((part, zField), zmin=zFieldLim[0], zmax=zFieldLim[1])
        fig1.set_cmap(field=(part, zField), cmap=cM)
        fig1.set_buff_size(bSize)
        fig1.annotate_timestamp(redshift=True)

        # Plots a second axis if specified
        if twoAxis:
            if verbose > 11: f.write(f"    Projecting in axis {axisProj[1]}\n")
            if wField != 0:
                fig2 = yt.ProjectionPlot(snap, axisProj[1], (part, zField), window_size=plotSize, weight_field=(part,wField), fontsize=fsize, center=simArr[i].snap[idxArr[i]].ytcen) 
            else:
                if part=="PartType4":
                    fig2 = yt.ParticleProjectionPlot(snap, axisProj[1], (part, zField), window_size=plotSize, depth=(zWidth,"kpc"), fontsize=fsize, center=simArr[i].snap[idxArr[i]].ytcen)
                else:
                    fig2 = yt.ProjectionPlot(snap, axisProj[1], (part, zField), window_size=plotSize, fontsize=fsize, center=simArr[i].snap[idxArr[i]].ytcen)

            fig2.set_width(zWidth,"kpc")
            if zFieldUnit != 0: fig2.set_unit((part, zField), zFieldUnit)
            if zFieldLim != 0:  fig2.set_zlim((part, zField), zmin=zFieldLim[0], zmax=zFieldLim[1])
            fig2.set_cmap(field=(part, zField), cmap=cM)
            fig2.set_buff_size(bSize)
            fig2.annotate_timestamp(redshift=True)

        # Transfers yt plot to plt axes and renders the figure
        if verbose > 11: f.write(f"    Rendering {simArr[i].name}\n")
        fullPlot = fig1.plots[part, zField]
        fullPlot.figure = panelFig
        fullPlot.axes = panelGrid[i].axes
        fullPlot.cax = panelGrid.cbar_axes[i]

        if twoAxis:
            fullPlot2 = fig2.plots[part, zField]
            fullPlot2.figure = panelFig
            fullPlot2.axes = panelGrid[len(simArr)+i].axes
            fullPlot2.cax = panelGrid.cbar_axes[len(simArr)+i]
            fig2._setup_plots()

        fig1._setup_plots()

        # Overplot halos if prompted to and passed
        if ovHalo != 0:
            if verbose > 10: f.write(f"    Overplotting halos\n")
            haloData = ovHalo[0][i].all_data()
            haloFilt = ovHalo[1]
            #print(haloData['particle_position_x'][haloFilt[i]].in_units("kpc"))
            #print(simArr[i].snap[idxArr[i]].center[0],simArr[i].snap[idxArr[i]].center[1],simArr[i].snap[idxArr[i]].center[2])
            xc = np.array(haloData['particle_position_x'][haloFilt[i]].in_units("kpc"))-simArr[i].snap[idxArr[i]].center[0]/1e3
            yc = np.array(haloData['particle_position_y'][haloFilt[i]].in_units("kpc"))-simArr[i].snap[idxArr[i]].center[1]/1e3
            zc = np.array(haloData['particle_position_z'][haloFilt[i]].in_units("kpc"))-simArr[i].snap[idxArr[i]].center[2]/1e3
            rc = np.array(haloData['virial_radius'][haloFilt[i]].in_units("kpc"))*1e3
            #print(xc,yc,zc,rc)
            for j in range(len(xc)):
                panelGrid.axes_all[i].add_patch(plt.Circle((xc[j],yc[j]),rc[j],ec="r",fc="none"))
                
                if twoAxis:
                    panelGrid.axes_all[len(simArr)+i].add_patch(plt.Circle((yc[j],zc[j]),rc[j],ec="r",fc="none"))

        # Sets title
        panelGrid.axes_all[i].set_title(titleArr[i])
    
    handleFig(panelFig,[showFig,animate,saveFig],message,saveFigPath,verbose)
    
# Plots phase space 2D histograms, defaults to gas phase (Density, Temperature, Mass)
def ytPhasePanel(simArr, idxArr, depositionAlg="ngp", verbose=verboseLevel, plotSize=ytFigSize, saveFig=saveAll,
                 saveFigPath=0, showFig=showAll,message=0, blackLine=0, panOver=0, part = "PartType0", zLog=1,
                 zFields = ["Density","Temperature","Masses"], zFieldUnits = ["g/cm**3","K","Msun"], cM = "algae", animate = 0,
                 zFieldLim = (1e3,1e8,1e-29,1e-21,10,1e7), zWidth=15, fsize=12, wField=0, xb=300, yb=300, grid=True, axAspect = 1):
    if message != 0: f.write(f"\n{message}\n")
    # Panel fig setup
    if isinstance(wField,list) == False: wField = [wField]*len(simArr)
    if panOver == 0:
        panelSize = (1, math.ceil(len(simArr)))
    else:
        panelSize = panOver
    panelFig = plt.figure(figsize=(1,1))
    panelGrid = AxesGrid(panelFig,(0,0,0.4*panelSize[1],0.4*panelSize[0]),aspect=False,nrows_ncols=panelSize,axes_pad=0.1,
                         label_mode="1",share_all=True,cbar_location="right",cbar_mode="single",cbar_size="5%",cbar_pad="2%")
    
    if zFieldLim   == 0: zFieldLim   = [0,0,0,0,0,0]
    if zFieldUnits == 0: zFieldUnits = [0,0,0]

    # Loading snapshots
    snapArr  = [simArr[i].ytFull[idxArr[i]] for i in range(len(simArr))]
    titleArr = [simArr[i].name              for i in range(len(simArr))]

    # Getting the black line
    if blackLine:
        if verbose > 10: f.write(f"  Calculating avg profile with {simArr[0].name}\n")
        sp = snapArr[0].sphere(simArr[0].snap[idxArr[0]].ytcen,(zWidth,"kpc"))

        p1 = yt.ProfilePlot(sp,(part,zFields[0]),(part,zFields[1]),weight_field=(part,zFields[2]), n_bins=30, x_log=False, accumulation=False)
        
        p1.set_log((part,zFields[0]),True)
        p1.set_log((part,zFields[1]),True)

        if zFieldUnits[0] != 0: p1.set_unit((part,zFields[0]), zFieldUnits[0])
        if zFieldLim[2] != 0 or zFieldLim[3] != 0: p1.set_xlim(zFieldLim[2], zFieldLim[3])
        if zFieldUnits[1] != 0: p1.set_unit((part,zFields[1]), zFieldUnits[1])

        cil = p1.profiles[0].x.in_units(zFieldUnits[0]).d
        bin = p1.profiles[0][zFields[1]].in_units(zFieldUnits[1]).d
        goodbin = []
        goodcil = []
        for i in range(len(bin)):
            if abs(bin[i]) > 1e-33:
                goodbin.append(bin[i]) 
                goodcil.append(cil[i])
           
    # Start of the fig making
    for i,snap in enumerate(snapArr):
        if verbose > 9: f.write(f"  - Plotting {simArr[i].name}\n")
        sp = snap.sphere(simArr[i].snap[idxArr[i]].ytcen,(zWidth,"kpc"))
        # Plot phase with specified parameters
        if zLog != 1:
            snap.field_info[(part, zFields[2])].take_log = False
        
        if wField[i] != 0:
            fig1 = yt.ParticlePhasePlot(sp,  (part, zFields[0]),(part, zFields[1]),(part, zFields[2]), deposition=depositionAlg,
                                         figure_size=plotSize, weight_field=(part,wField[i]), fontsize=fsize, x_bins=xb, y_bins=yb)
        else:
            fig1 = yt.ParticlePhasePlot(sp,  (part, zFields[0]),(part, zFields[1]),(part, zFields[2]), deposition=depositionAlg,
                                         figure_size=plotSize, fontsize=fsize, x_bins=xb, y_bins=yb)
            
        
        if zFieldUnits[0] != 0: fig1.set_unit((part, zFields[0]), zFieldUnits[0])
        if zFieldUnits[1] != 0: fig1.set_unit((part, zFields[1]), zFieldUnits[1])
        if zFieldUnits[2] != 0: fig1.set_unit((part, zFields[2]), zFieldUnits[2])
            
        if zFieldLim[0] != 0 or zFieldLim[1] != 0: fig1.set_zlim((part, zFields[2]), zmin=zFieldLim[0], zmax=zFieldLim[1])
        if zFieldLim[2] != 0 or zFieldLim[3] != 0: fig1.set_xlim(zFieldLim[2], zFieldLim[3])
        if zFieldLim[4] != 0 or zFieldLim[5] != 0: fig1.set_ylim(zFieldLim[4], zFieldLim[5])

        fig1.set_log((part,zFields[2]),bool(zLog))

        fig1.set_cmap(field=(part, zFields[2]), cmap=cM)

        #fig1.annotate_text(0,0,"t="+str(round(sims[i].snap[idx[i]].time))+" Myr\n z="+str(round(sims[i].snap[idx[i]].z,2)))

        # Transfers yt plot to plt axes and renders the figure
        if verbose > 11: f.write(f"    Rendering {simArr[i].name}\n")
        fullPlot = fig1.plots[part, zFields[2]]
        fullPlot.figure = panelFig
        fullPlot.axes = panelGrid[i].axes
        if i == 0:
            fullPlot.cax = panelGrid.cbar_axes[i]
        
        fig1._setup_plots()

        if blackLine:
            panelGrid.axes_all[i].plot(goodcil,goodbin,"k--")
        panelFig.canvas.draw()
        if grid: panelGrid.axes_all[i].grid()
        panelGrid.axes_all[i].set_title(titleArr[i])
        panelGrid.axes_all[i].set_box_aspect(axAspect)
        #panelGrid.axes_all[i].set_aspect(axAspect)
        
    handleFig(panelFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

# Plots a binned field
def plotBinned(sims,idx,binFields,nBins,rLim,logOverload=0,legOverload=0,diffSims=0,blLine=0,wField=0,spLim=0,binFunction=0,part=gasPart,setUnits=0,setLogs=(False,True),ylims=0,xlims=0,animate=0,
               xylabels=0,plotTitle=0,errorLim=errorLimGlobal,message=0,verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,showError=showErrorGlobal,axAspect=1):
    if message != 0: f.write(f"\n{message}\n")
    # Initialize figures
    if showError == 1:
        if isinstance(plotSize,list) == False: plotSize = [plotSize,plotSize*1.2] 
        uFig = plt.figure(figsize=(plotSize[0], plotSize[1]))
        uAx  = plt.subplot2grid((4,1),(0,0),rowspan=3)
        uAx2 = plt.subplot2grid((4,1),(3,0),rowspan=1)
    elif showError == 2:
        if isinstance(plotSize,list) == False: plotSize = [plotSize,plotSize*1.5] 
        uFig = plt.figure(figsize=(plotSize[0], plotSize[1]))
        uAx  = plt.subplot2grid((5,1),(0,0),rowspan=3)
        uAx2 = plt.subplot2grid((5,1),(3,0),rowspan=2)
    else:
        if isinstance(plotSize,list) == False: plotSize = [plotSize,plotSize] 
        uFig = plt.figure(figsize=(plotSize[0], plotSize[1]))
        uAx  = plt.subplot2grid((4,1),(0,0),rowspan=4)
 
    allYbin = [None]*len(sims)
    XGlobal = None
    for k,sn in enumerate(sims):

        # Setting up parameters
        if verbose > 9: f.write(f"  Started {sn.name}\n")
        splimit = rLim[1]
        if spLim != 0: splimit = spLim
        weightField = None
        if wField != 0: weightField = wField

        # Actual binning proccess
        sp = sims[k].ytFull[idx[k]].sphere(sims[k].snap[idx[k]].ytcen,(splimit,"kpc"))
        p1 = yt.ProfilePlot(sp,(part,binFields[0]),(part,binFields[1]),weight_field=weightField, n_bins=nBins, x_log=setLogs[0], accumulation=False)
        p1.set_log((part,binFields[0]),setLogs[0])
        p1.set_log((part,binFields[1]),setLogs[1])
        if setUnits[0] != 0: p1.set_unit((part,binFields[0]), setUnits[0])
        if setUnits[1] != 0: p1.set_unit((part,binFields[1]), setUnits[1])
        p1.set_xlim(rLim[0],rLim[1])

        # Extract bins to perform further operations
        if setUnits[0] != 0: 
            xData = p1.profiles[0].x.in_units(setUnits[0]).d 
        else: 
            xData = p1.profiles[0].x.d
        if setUnits[1] != 0: 
            bin = p1.profiles[0][binFields[1]].in_units(setUnits[1]).d
        else: 
            bin = p1.profiles[0][binFields[1]].d

        # If there is a postproccessing function, do that
        if binFunction != 0: bin = binFunction(xData,bin)

        # Performs the difference between two bins of two simulation sets (of the same length!!)
        if diffSims != 0:
            sp2 = diffSims[0][k].ytFull[diffSims[1][k]].sphere(diffSims[0][k].snap[diffSims[1][k]].ytcen,(splimit,"kpc"))
            p2 = yt.ProfilePlot(sp2,(part,binFields[0]),(part,binFields[1]),weight_field=weightField, n_bins=nBins, x_log=setLogs[0], accumulation=False)
            p2.set_log((part,binFields[0]),setLogs[0])
            p2.set_log((part,binFields[1]),setLogs[1])
            if setUnits != 0: p2.set_unit((part,binFields[0]), setUnits[0])
            if setUnits != 0: p2.set_unit((part,binFields[1]), setUnits[1])
            p2.set_xlim(rLim[0],rLim[1])

            xData2 = p2.profiles[0].x.in_units(setUnits[0]).d
            bin2   = p2.profiles[0][binFields[1]].in_units(setUnits[1]).d
            if binFunction != 0: bin2 = binFunction(xData2,bin2)
            bin = np.array(bin)-np.array(bin2)

        allYbin[k] = bin
        XGlobal = xData
        uAx.plot(xData,bin,".--")

        # Setups log scaling depending on options
        setLogsPlot = setLogs
        if logOverload != 0: setLogsPlot = logOverload
        if setLogsPlot[0]: uAx.semilogx()
        if setLogsPlot[1]: uAx.semilogy()
        uAx.set_box_aspect(axAspect)
        
    # Plots dispersion of codes
    if showError == 1:
        if verbose > 10: f.write(f"  Plotting dispersion\n")
        allYbin = np.array(allYbin)
        error = [None]*nBins
        for i in range(nBins):
            average = np.mean(allYbin[:,i])
            if average != 0:
                error[i] = (allYbin[:,i]-average)/average
            else:
                error[i] = np.zeros_like(allYbin[:,i])
        error = np.array(error)
        for i in range(len(sims)):
            uAx2.plot(XGlobal,error[:,i],".")
            if setLogs[0]: uAx2.semilogx()
            
        if xlims != 0: uAx2.set_xlim(xlims[0],xlims[1])
        uAx2.set_ylim(errorLim[0],errorLim[1])
        uAx2.grid()

        if xylabels != 0: uAx2.set_ylabel("Residual ($\\frac{\sigma - \overline{\sigma}}{\overline{\sigma}}$)")
        if xylabels != 0: uAx2.set_xlabel(xylabels[0])
    elif showError == 2:
        allYbin = np.array(allYbin)
        for i in range(len(sims)):
            uAx2.plot(XGlobal,allYbin[i,:]/1e8,".")

        if xlims != 0: uAx2.set_xlim(xlims[0],xlims[1])
        uAx2.set_ylim(errorLim[0],errorLim[1])
        uAx2.grid()
        uAx2.set_xscale('log')
        uAx2.set_yscale('symlog',linthresh=0.01) 
        if xylabels != 0: uAx2.set_ylabel("Log "+xylabels[1])
        if xylabels != 0: uAx2.set_xlabel(xylabels[0])

        if blLine != 0:
            uAx2.axvline(x = blLine, color='k', linestyle ='--', linewidth=2, alpha=0.7)
    else:
        if xylabels != 0: uAx.set_xlabel(xylabels[0])
    
    # Set limits and labels
    if xylabels != 0:  uAx.set_ylabel(xylabels[1])
    if plotTitle != 0: uAx.set_title(plotTitle)
    if xlims != 0: uAx.set_xlim(xlims[0],xlims[1])
    if ylims != 0: uAx.set_ylim(ylims[0],ylims[1])
    
    # Threshold of sff
    if blLine != 0:
        uAx.axvline(x = blLine, color='k', linestyle ='--', linewidth=2, alpha=0.7)

    uAx.grid()
    
    if legOverload == 0:
        setLegend(uAx,sims,idx)
    else:
        uAx.legend(legOverload)
    
    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

# Calculates the velocity dispersion for a given particle type
def plotRotDisp(sims,idx,nBins,rLim,part,titlePlot=0,verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,message=0,ylims=(0,170),animate=0):
    if message != 0: f.write(f"\n{message}\n")
    uFig = plt.figure(figsize=(plotSize, plotSize*1.5))
    uAx  = plt.subplot2grid((5,1),(0,0),rowspan=3)
    uAx2 = plt.subplot2grid((5,1),(3,0),rowspan=1)
    uAx3 = plt.subplot2grid((5,1),(4,0),rowspan=1)
	
    snapArr  = [sims[i].ytFull[idx[i]] for i in range(len(sims))]
    centArr  = [sims[i].snap[idx[i]].ytcen for i in range(len(sims))]
	
    # New fields (this one is complicated, I need first to get each particle to have a field with its rotational vel from the bin it is in, and then using that as mean calculate the dispersion)
    for i,snap in enumerate(snapArr):

		# Get the velocity bins
        sp = snap.sphere(centArr[i],(rLim,"kpc"))    
        rotProf = yt.ProfilePlot(sp,(part,"particle_position_cylindrical_radius"),(part,"particle_velocity_cylindrical_theta"),weight_field=(part,"Masses"), n_bins=nBins, x_log=False)
        rotProf.set_log((part,"particle_position_cylindrical_radius"),False)
        rotProf.set_log((part,"particle_velocity_cylindrical_theta"),False)
        rotProf.set_unit((part,"particle_velocity_cylindrical_theta"), 'km/s')
        rotProf.set_unit((part,"particle_position_cylindrical_radius"), 'kpc')
        rotProf.set_xlim(0, rLim-1)
        rotProf.set_ylim((part,"particle_velocity_cylindrical_theta"), 0, 250)
        rotCilLocal = rotProf.profiles[0].x.in_units('kpc').d
        rotBinLocal = rotProf.profiles[0]["particle_velocity_cylindrical_theta"].in_units('km/s').d
	
        # Defined the functions each time, so that the rotCil and rotBin are properly set and correspond to the correct snapshot
        def make_field_func(rotCil,rotBin):
            # Get the x velocity of a particle using its binned rot vel and its angle
            def _particle_rot_vx(field, data):
                trans = np.zeros(data[(part,"particle_velocity_x")].shape)
                dr = 0.5*(rotCil[1]-rotCil[0])
                # Go through each bin
                for rad, vrot in zip(rotCil,rotBin):
                    # Select the indices of particles inside this bin
                    ind = np.where( (data[(part,"particle_position_cylindrical_radius")].in_units("kpc") >= (rad - dr)) & \
                                    (data[(part,"particle_position_cylindrical_radius")].in_units("kpc") <  (rad + dr)) )
                    # For the particles inside this bin, calculate their x velocity with the bin average 
                    trans[ind] = -np.sin(data[(part, "particle_position_cylindrical_theta")][ind]) * vrot * 1e5 
                # Return trans but with whatever code units this dataset uses
                return data.ds.arr(trans, "cm/s").in_base(data.ds.unit_system.name)
            
            # Get the y velocity of a particle using its binned rot vel and its angle
            def _particle_rot_vy(field, data):
                trans = np.zeros(data[(part,"particle_velocity_y")].shape)
                dr = 0.5*(rotCil[1]-rotCil[0])
                # Go through each bin
                for rad, vrot in zip(rotCil,rotBin):
                    # Select the indices of particles inside this bin
                    ind = np.where( (data[(part,"particle_position_cylindrical_radius")].in_units("kpc") >= (rad - dr)) & \
                                    (data[(part,"particle_position_cylindrical_radius")].in_units("kpc") <  (rad + dr)) )
                    # For the particles inside this bin, calculate their x velocity with the bin average 
                    trans[ind] = np.cos(data[(part, "particle_position_cylindrical_theta")][ind]) * vrot * 1e5 
                # Return trans but with whatever code units this dataset uses
                return data.ds.arr(trans, "cm/s").in_base(data.ds.unit_system.name)
            return _particle_rot_vx, _particle_rot_vy

        vx_func, vy_func = make_field_func(rotCilLocal, rotBinLocal)
		
        snap.add_field((part, "particle_rot_vx"), function=vx_func, take_log=False,units="cm/s",sampling_type="particle",force_override=True) 
        snap.add_field((part, "particle_rot_vy"), function=vy_func, take_log=False,units="cm/s",sampling_type="particle",force_override=True) 
		
        # Take the dispersion with respect to the velocity obtained from each bin
        def _particle_vel_disp(field, data):
            return (data[(part, "particle_velocity_x")] - data[(part, "particle_rot_vx")])**2 + \
				   (data[(part, "particle_velocity_y")] - data[(part, "particle_rot_vy")])**2 + \
				   (data[(part, "particle_velocity_z")])**2 
    
        snap.add_field((part, "particle_vel_disp"), function=_particle_vel_disp, take_log=False,units="cm**2/s**2",sampling_type="particle",force_override=True) 
        
        def _particle_velocity_z_squared(field, data):
            return (data[(part, "particle_velocity_z")])**2 
        snap.add_field((part, "particle_velocity_z_squared"), function=_particle_velocity_z_squared, take_log=False, units="cm**2/s**2", sampling_type="particle", force_override=True) 

    allYbin = [None]*len(sims)
    allYbinZ = [None]*len(sims)
    XGlobal = None

    for k,sn in enumerate(sims):

        if verbose > 9: f.write(f"  Started {sn.name}\n")
        sp = sims[k].ytFull[idx[k]].sphere(sims[k].snap[idx[k]].ytcen,(rLim,"kpc"))
        
        p1 = yt.ProfilePlot(sp,(part,"particle_position_cylindrical_radius"),(part,"particle_vel_disp"),weight_field=(part,"Masses"), n_bins=nBins, x_log=False)
        p1.set_log((part,"particle_position_cylindrical_radius"), False)
        p1.set_unit((part,"particle_position_cylindrical_radius"), "kpc")
        p1.set_xlim(1e-3, rLim-1)
        
        cil = p1.profiles[0].x.in_units('kpc').d
        bins = np.sqrt(p1.profiles[0]["particle_vel_disp"]).in_units('km/s').d

        # Vertical z speed dispersion
        p2 = yt.ProfilePlot(sp, (part, "particle_position_cylindrical_radius"), (part, "particle_velocity_z_squared"),weight_field=(part,"Masses"), n_bins=nBins, x_log=False)
        p2.set_log((part,"particle_position_cylindrical_radius"), False)
        p2.set_unit((part,"particle_position_cylindrical_radius"), "kpc")
        p2.set_xlim(1e-3, rLim-1)
        allYbinZ[k] = np.sqrt(p2.profiles[0]["particle_velocity_z_squared"]).in_units('km/s').d
        
        allYbin[k] = bins
        XGlobal = cil
        uAx.plot(cil,bins,".--")
    
    allYbin = np.array(allYbin)
    error = [None]*nBins
    if verbose > 10: f.write(f"  Plotting dispersion\n")
    for i in range(nBins):
        average = np.mean(allYbin[:,i])
        if average != 0:
            error[i] = (allYbin[:,i]-average)/average
        else:
            error[i] = np.zeros_like(allYbin[:,i])
    error = np.array(error)
    for i in range(len(sims)):
        uAx2.plot(XGlobal,error[:,i],".")
    uAx2.set_xlim(0,(rLim-1))
    uAx2.set_ylim(-1,1)
    uAx2.set_ylabel("Residual ($\\frac{\sigma - \overline{\sigma}}{\overline{\sigma}}$)")
    uAx2.grid()
	
    allYbinZ = np.array(allYbinZ)
    if verbose > 10: f.write(f"  Plotting dispersion ratio\n")
    for i in range(len(sims)):
        dispRatio = allYbinZ[i,:]/allYbin[i,:]
        uAx3.plot(XGlobal,dispRatio,".")
    uAx3.set_xlim(0,(rLim-1))
    uAx3.set_ylim(0,1)
    uAx3.set_ylabel("Vertical dispersion ratio ($\\frac{\sigma_z}{\sigma}$)")
    uAx3.grid()
    uAx3.set_xlabel("Cylidrincal radius (Kpc)")

    uAx.set_xlim(0,14)
    if ylims != 0: uAx.set_ylim(ylims[0],ylims[1])

    
    uAx.set_ylabel("Velocity dispersion (km/s)")
    if titlePlot != 0: uAx.set_title(titlePlot)
    uAx.grid()

    setLegend(uAx,sims,idx)

    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

# CLUMP FINDING AND LOADING
def findHalos(simArr, idxArr, partT, mainPath = simDir,haloMethod = "fof", hopThresh = 4e9, fofLink = 0.0012, hardLimits = True, overWrite = True, clumpLim  = (1e6,8e8), verbose = verboseLevel):
    if verbose > 8: f.write(f"\nInitiating halo finding for particle type: {partT} and Method: {haloMethod}\n") 
    # Initialize halo arrays
    haloSims = [None]*len(simArr)
    haloFilt = [None]*len(simArr)
    temp = None
    for i in range(len(simArr)):
        # Load parameters and paths
        if verbose > 9: f.write(f"  - Loading halos for {simArr[i].name}\n")
        snap = simArr[i].ytFull[idxArr[i]]
        haloDirSim = os.path.join("Halos","Halo_"+haloMethod+"_"+partT+"_"+simArr[i].name.replace(" ","_"))
        haloPath = os.path.join(mainPath,haloDirSim)
        haloDirPath  = os.path.join(haloPath,snap.basename[:snap.basename.find(".")])
        haloFilePath = os.path.join(haloDirPath,snap.basename[:snap.basename.find(".")]+".0.h5")

        # Do the halo finding if no halos detected
        if os.path.exists(haloDirPath) == False or overWrite:
            # Explain what files are being modified or not
            if os.path.exists(haloDirPath) == False:
                if verbose > 9: f.write(f"    No halos detected in {haloDirPath}\n")
            elif overWrite:
                if verbose > 9: f.write(f"    Overwriting halos detected in {haloDirPath}\n")
            
            if verbose > 9: f.write(f"    Initializing halo finding to be saved in {haloFilePath}\n")
            
            # Configure the halo catalog and halo finding method
            if      haloMethod == "hop":
                hopConf = hopThresh
                if isinstance(hopThresh, list): hopConf = hopThresh[i]
                hc = yt.extensions.astro_analysis.halo_analysis.HaloCatalog(data_ds=snap, data_source=snap.all_data(), finder_method="hop", output_dir=haloPath,finder_kwargs={"threshold": hopConf, "dm_only": False, "ptype": partT})
            elif    haloMethod == "fof":
                fofConf = fofLink
                if isinstance(fofLink, list): fofConf = fofLink[i]
                hc = yt.extensions.astro_analysis.halo_analysis.HaloCatalog(data_ds=snap, data_source=snap.all_data(), finder_method="fof", output_dir=haloPath,finder_kwargs={"link": fofConf, "dm_only": False, "ptype": partT})
            
            # Add filters and calculate the actual halo
            #hc.add_filter('quantity_value', 'particle_mass', '>', clumpLim[0], 'Msun') # exclude halos with less than 30 particles
            #hc.add_filter('quantity_value', 'particle_mass', '<', clumpLim[1], 'Msun') # exclude the most massive halo (threshold 1e8.4 is hand-picked, so one needs to be careful!)
            hc.create() 

        # Delete need for cosmological parameters
        def _parse_parameter_file_no_cosmo(self):
            # List of attributes expected by the halo dataset.
            for attr in [
                "cosmological_simulation",
                "cosmology",
                "current_redshift",
                "current_time",
                "dimensionality",
                "domain_dimensions",
                "domain_left_edge",
                "domain_right_edge",
                "domain_width",
                "hubble_constant",
                "omega_lambda",
                "omega_matter",
                "unique_identifier",
            ]:
                try:
                    setattr(self, attr, getattr(self.real_ds, attr))
                except AttributeError:
                    # If the attribute is missing, assign a default value or None
                    defVal = {"current_time": 0}
                    if attr in defVal:
                        setattr(self, attr, defVal[attr])
                    else:
                        setattr(self, attr, None)
        # Monkey-patch the method.
        HaloDataset._parse_parameter_file = _parse_parameter_file_no_cosmo

        # Now load the halos from disk file
        if verbose > 9: f.write(f"    Loading halo from file{haloFilePath}\n")
        halo_ds  = yt.load(haloFilePath)
        hc = yt.extensions.astro_analysis.halo_analysis.HaloCatalog(halos_ds=halo_ds)
        hc.load()

        haloSims[i] = hc.halos_ds
        temp = haloSims[i].all_data()
        haloFilt[i] = np.ones(len(temp['particle_mass'].in_units("Msun")),dtype=bool)

        if hardLimits:    
            # Get the masses in Msun.
            mass = temp['particle_mass'][:].in_units("Msun")
            # Create a boolean mask for halos within the desired mass limits.
            keep = (mass >= clumpLim[0]) & (mass <= clumpLim[1])
            # Find the indices of the halos to keep.
            haloFilt[i] = np.where(keep)[0]
        
    if verbose > 10: f.write(f"  Halo loading successful!\n")
    return (haloSims,haloFilt)

# Plots the cumulative mass function of a collection of halos
def plotClumpMassF(sims,idx,haloData,nBins=20,mLim=(6,8.5),verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,message=0,animate=0):
    if message != 0: f.write(f"\n{message}\n")
    # Setup plot figures and axes
    uFig = plt.figure(figsize=(plotSize, plotSize))
    uAx  = plt.subplot2grid((4,1),(0,0),rowspan=4)

    # Calculate the cumulative mass function for each snapshot
    for k,sn in enumerate(sims):
        if verbose > 9: f.write(f"  Started {sn.name}\n")
        temp = haloData[0][k].all_data()
        clumpMass = temp['particle_mass'][haloData[1][k]].in_units("Msun")
            
        clumpLogMass = np.log10(clumpMass)
        hist = np.histogram(clumpLogMass, bins=nBins, range=(mLim[0],mLim[1]))
        dBin = hist[1][1]-hist[1][0]
        
        uAx.plot(hist[1][:-1]+dBin, np.cumsum(hist[0][::-1])[::-1],".--")
        uAx.semilogy()
    
    # Decorate the plot
    uAx.set_xlim(mLim[0],mLim[1])
    uAx.set_ylim(0.9,50)
    uAx.set_xlabel("$\mathrm{log[Newly\ Formed\ Stellar\ Clump\ Mass\ (M_{\odot})]}$")
    uAx.set_ylabel("$\mathrm{Cumulative Stellar\ Clump\ Counts, \ \ N_{clump}(> M)}$")
    uAx.set_title("Clump Cumulative Mass Function")
    uAx.grid()

    setLegend(uAx,sims,idx)

    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

def aFromT(time, eps = 0.1):
    co = yt.utilities.cosmology.Cosmology(hubble_constant=0.702, omega_matter=0.272,omega_lambda=0.728, omega_curvature=0.0)
    if time < eps: return 0
    return co.a_from_t(co.quan(time,"Myr"))

# Plot the total SFR of a simulation over time
def plotSFR(sims,idx,nBins=25,tLimPreset = [0,0],verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,message=0,yLims=[0,8],animate=0,xLims=0):
    if message != 0: f.write(f"\n{message}\n")
    # Setup plot figures and axes
    uFig = plt.figure(figsize=(figSize, figSize))
    uAx  = plt.subplot2grid((4,1),(0,0),rowspan=3)
    uAx2 = plt.subplot2grid((4,1),(3,0),rowspan=1)

    # Bin star ages into nBins and use that to estimate total SFR
    allYbin = [None]*len(sims)
    XGlobal = None
    for k,sn in enumerate(sims):
        tLim = [tLimPreset[0],tLimPreset[1]]
        if verbose > 9: f.write(f"  Started {sims[k].name} with time {sn.snap[idx[k]].time}\n")
        if tLimPreset[1] == 0: tLim[1] = sn.snap[idx[k]].time

        dt = (tLim[1]-tLim[0])/nBins
        timeX = np.linspace(tLim[0]+dt/2,tLim[1]-dt/2,nBins)
        starMass = [0]*nBins
        sfr = [0]*nBins

        prog = 0

        sp = sims[k].ytFull[idx[k]]
        allStarMass = np.array(sp.r["PartType4","Masses"].to("Msun"))
        
        allStarAge  = []
        binX        = None
        binLim      = None
        if sn.cosmo:
            allStarAge  = np.array(sp.r["PartType4","StellarFormationTime"])
            binX   = [aFromT(value) for value in timeX+dt/2]
            binLim = [aFromT(tLim[0]),aFromT(tLim[1])]
        else:
            allStarAge  = np.array(sp.r["PartType4","StellarFormationTime"])*1e3
            binX   = timeX+dt/2
            binLim = tLim

        for i in range(len(allStarAge)):
            if allStarAge[i] <= binLim[1] and allStarAge[i] >= binLim[0]:
                binIdx = getClosestIdx(binX,allStarAge[i])
                sfr[binIdx]      += allStarMass[i]/(dt*1e6)
                starMass[binIdx] += allStarMass[i]
            if i/len(allStarAge)*100-prog > 33:
                if verbose > 8: f.write(f"    {i/len(allStarAge)*100:.3f}%\n")
                prog = i/len(allStarAge)*100
        
        for i in range(nBins):
            if i == 0: continue
            starMass[i] += starMass[i-1]
            
        uAx.plot(timeX,sfr,".--")
        allYbin[k] = sfr
        XGlobal = timeX
    
    allYbin = np.array(allYbin)
    error = [None]*nBins
    for i in range(nBins):
        average = np.mean(allYbin[:,i])
        if average != 0:
            error[i] = (allYbin[:,i]-average)/average
        else:
            error[i] = np.zeros_like(allYbin[:,i])
    error = np.array(error)
    for i in range(len(sims)):
        uAx2.plot(XGlobal,error[:,i],".")
    uAx2.set_xlim(0,tLim[1])
    uAx2.set_ylim(-1,1)
    uAx2.set_ylabel("Residual ($\\frac{\sigma - \overline{\sigma}}{\overline{\sigma}}$)")
    uAx2.grid()
    uAx2.set_xlabel("Time (Myr)")

    if yLims != 0: uAx.set_ylim(yLims[0],yLims[1])
    if xLims != 0: uAx.set_xlim(xLims[0],xLims[1])
    
    uAx.set_ylabel("SFR ($\\frac{\mathrm{M}_{\odot}}{yr}$)")
    uAx.set_title("SFR Over time")
    uAx.grid()
    
    setLegend(uAx,sims,idx)

    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

# Plot the KS relation by binning cillindrically gas density and SFR
def plotKScil(sims,idx,nBins=50,rLim=0.5*figWidth,verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,message=0,animate=0):
    if message != 0: f.write(f"\n{message}\n")
    # Setup plot figures and axes
    uFig = plt.figure(figsize=(plotSize, plotSize))
    uAx  = plt.subplot2grid((4,1),(0,0),rowspan=4)

    for k,sn in enumerate(sims):
        if verbose > 9: f.write(f"  Started {sims[k].name} with time {sn.snap[idx[k]].time}\n")
        sp = sims[k].ytFull[idx[k]].sphere(sims[k].snap[idx[k]].ytcen,(rLim,"kpc"))
        
        # Calculate SFR den in cil bins
        p1 = yt.ProfilePlot(sp,(starPart,"particle_position_cylindrical_radius"),(starPart,"particle_mass_young_stars"),weight_field=None, n_bins=nBins, x_log=False)
        p1.set_log((starPart,"particle_position_cylindrical_radius"),False)
        p1.set_log((starPart,"particle_mass_young_stars"),True)
        p1.set_unit((starPart,"particle_mass_young_stars"), 'Msun')
        p1.set_unit((starPart,"particle_position_cylindrical_radius"), 'kpc')
        p1.set_xlim(1e-3, rLim)

        cil = p1.profiles[0].x.in_units('kpc').d
        binsPrev = p1.profiles[0]["particle_mass_young_stars"].in_units('Msun').d/youngStarAge/1e6

        dr = 0.5*(cil[1]-cil[0])
        SFRbins = []
        for i in range(len(cil)):
            SFRbins.append(binsPrev[i]/(np.pi * (((cil[i]+dr))**2-((cil[i]-dr))**2) ))

        # Calculate Gas den in cil bins
        p2 = yt.ProfilePlot(sp,(gasPart,"particle_position_cylindrical_radius"),(gasPart,"Masses"),weight_field=None, n_bins=nBins, x_log=False, accumulation=False)
        p2.set_log((gasPart,"Masses"),True)
        p2.set_log((gasPart,"particle_position_cylindrical_radius"),False)
        p2.set_unit((gasPart,"particle_position_cylindrical_radius"), 'kpc')
        p2.set_unit((gasPart,"Masses"), 'Msun')
        p2.set_xlim(0, rLim)

        rcil = p2.profiles[0].x.in_units('kpc').d
        massB = p2.profiles[0]["Masses"].in_units('Msun').d

        dr = 0.5*(rcil[1]-rcil[0])
        DENbins = []
        for i in range(len(rcil)):
            DENbins.append(massB[i]/(np.pi * (((rcil[i]+dr)*1e3)**2-((rcil[i]-dr)*1e3)**2) ))	

        # Calculate KS with both binned results
        # Filter low surf density bins
        ind = np.where(np.array(SFRbins) > 1e-10)

        xKS = np.log10(np.array(DENbins)[ind])
        yKS = np.log10(np.array(SFRbins)[ind])
        uAx.scatter(xKS,yKS)

        setLegend(uAx,sims,idx)

    # Obs KS line from 2008 Bigiel
    t = np.arange(-2, 5, 0.01)
    uAx.plot(t, 1.37*t - 3.78, 'k--', linewidth = 2, alpha = 0.7)

    # Obs KS contour from 2008 Bigiel
    fBigiel = open("bilcontour.txt","r+")
    dataX = []
    dataY = []
    for line in fBigiel:
        data = np.asarray(line.split(", "),dtype=float)
        dataX.append(data[0]+ np.log10(1.36))
        dataY.append(data[1])

    uAx.fill(dataX,dataY,fill=True, color='b', alpha = 0.1, hatch='\\')
    uAx.set_xlabel("$\mathrm{log[Gas\ Surface\ Density\ (M_{\odot}/pc^2)]}$")
    uAx.set_ylabel("$\mathrm{log[Star\ Formation\ Rate\ Surface\ Density\ (M_{\odot}/yr/kpc^2)]}$")
    uAx.set_xlim(0,3)
    uAx.set_ylim(-4,1)
    uAx.set_title("Kennicutt–Schmidt relation with cilindrically binned data")
    uAx.grid()

    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

# Plot the KS relation by binning gas density and SFR in squares tiling the whole galaxy (mock observations)
def plotKSmock(sims,idx,fsize=fontSize,rLim=0.5*figWidth,verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,message=0,animate=0,resMock=lowResMock):
    axisProj     = 0
    zFieldLim1   = (1e0 , 1e3)
    zFieldLim2   = (3e-4, 3e-1)

    cmapDef = plt.get_cmap("tab10")
    if message != 0: f.write(f"\n{message}\n")

    # Setup plot figures and axes
    uFig = plt.figure(figsize=(plotSize, plotSize))
    uAx  = plt.subplot2grid((4,1),(0,0),rowspan=4)
    nMockBins = int(rLim*2*1e3/resMock)

    for k,sn in enumerate(sims):
        if verbose > 9: f.write(f"  Started {sn.name}\n")

        sp = sims[k].ytFull[idx[k]].sphere(sims[k].snap[idx[k]].ytcen,(rLim,"kpc"))
        
        # Calculate SFR den in mock rectangular bins
        f.write(f"  Plotting {sims[k].name} in {axisProj} Time {sims[k].snap[idx[k]].time:.1f} Myr\n")
        fig1 = yt.ParticlePhasePlot(sp,  (starPart, "x_centered"),(starPart, "y_centered"),(starPart, "sfr_den_low_res"), weight_field=None, deposition="cic", fontsize=fsize, x_bins=nMockBins, y_bins=nMockBins)
        fig1.set_zlim((starPart, "sfr_den_low_res"), zmin=zFieldLim2[0], zmax=zFieldLim2[1])
        fig1.set_xlim(-rLim,rLim)
        fig1.set_ylim(-rLim,rLim)

        # Calculate Gas den in cil bins
        fig2 = yt.ParticlePhasePlot(sp,  (gasPart, "x_centered"),(gasPart, "y_centered"),(gasPart, "den_low_res"), weight_field=None, deposition="cic", fontsize=fsize, x_bins=nMockBins, y_bins=nMockBins)
        fig2.set_zlim((gasPart, "den_low_res"), zmin=zFieldLim1[0], zmax=zFieldLim1[1])
        fig2.set_xlim(-rLim,rLim)
        fig2.set_ylim(-rLim,rLim)

        # Calculate KS with both binned results
        SFRbins = fig1.profile[starPart,"sfr_den_low_res"].reshape(1, nMockBins**2)[0]
        DENbins = fig2.profile[gasPart,"den_low_res"].reshape(1, nMockBins**2)[0]

        # Filter low surf density bins
        ind = np.where((np.array(SFRbins) > 1e-10)&(np.array(DENbins) > 1e-10))

        xKS = np.log10(np.array(DENbins[ind]))
        yKS = np.log10(np.array(SFRbins[ind]))

        uAx.scatter(xKS,yKS,alpha=0.1)

        # Drawing contours rather than scattering all the datapoints; see http://stackoverflow.com/questions/19390320/scatterplot-contours-in-matplotlib
        if len(xKS) > 10 and len(yKS) > 10:
            Gaussian_density_estimation_nbins = 20
            kernel = kde.gaussian_kde(np.vstack([xKS, yKS])) 
            xi, yi = np.mgrid[xKS.min():xKS.max():Gaussian_density_estimation_nbins*1j, yKS.min():yKS.max():Gaussian_density_estimation_nbins*1j]
            zi = np.reshape(kernel(np.vstack([xi.flatten(), yi.flatten()])), xi.shape)
            uAx.contour(xi, yi, zi, np.array([0.2]), linewidths=1.5, colors=cmapDef(k))    # 80% percentile contour
        else: f.write(f"  Insufficent data points (xKS {len(xKS)} and yKS {len(yKS)}). Skipping contour\n")
        
        setLegend(uAx,sims,idx)

    # Obs KS line from 2008 Bigiel
    t = np.arange(-2, 5, 0.01)
    uAx.plot(t, 1.37*t - 3.78, 'k--', linewidth = 2, alpha = 0.7)

    # Obs KS contour from 2008 Bigiel
    fBigiel = open("bilcontour.txt","r+")
    dataX = []
    dataY = []
    for line in fBigiel:
        data = np.asarray(line.split(", "),dtype=float)
        dataX.append(data[0]+ np.log10(1.36))
        dataY.append(data[1])
    
    uAx.fill(dataX,dataY,fill=True, color='b', alpha = 0.1, hatch='\\')
    
    uAx.set_xlabel("$\mathrm{log[Gas\ Surface\ Density\ (M_{\odot}/pc^2)]}$")
    uAx.set_ylabel("$\mathrm{log[Star\ Formation\ Rate\ Surface\ Density\ (M_{\odot}/yr/kpc^2)]}$")

    uAx.set_xlim(0,3)
    uAx.set_ylim(-4,1)
    
    uAx.set_title("Kennicutt–Schmidt relation with mock observations")
    uAx.grid()
    
    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

# Plot the total SFR of a simulation over time
def plotSFmass(sims,idx,nBins=50,zLim = [0,0],verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,message=0,yLims=[5e6,5e9],xLims=0,splimit=100,animate=0):
    if message != 0: f.write(f"\n{message}\n")
    # Setup plot figures and axes
    uFig = plt.figure(figsize=(8/10*figSize, 6/10*figSize))
    uAx  = plt.subplot2grid((4,1),(0,0),rowspan=4)

    co = yt.utilities.cosmology.Cosmology(hubble_constant=0.702, omega_matter=0.272,omega_lambda=0.728, omega_curvature=0.0)

    # Bin star ages into nBins and use that to estimate total mass
    for k,sn in enumerate(sims):
        if verbose > 9: f.write(f"  Started {sims[k].name} with time {sn.snap[idx[k]].time:.1f}\n")
        # Maybe limit to rvir?
        sp = sims[k].ytFull[idx[k]].sphere(sims[k].snap[idx[k]].ytcen,(splimit,"kpc"))
        # Gets limits from current snapshot and earliest recorded star
        if zLim[1] == 0: zLim[1] = sn.snap[idx[k]].z
        if zLim[0] == 0: zLim[0] = float(1/min(np.array(sp["PartType4","StellarFormationTime"]))-1)
        tLim = [float(co.t_from_z(zLim[0]).to("Myr")),float(co.t_from_z(zLim[1]).to("Myr"))]

        dt = (tLim[1]-tLim[0])/nBins
        timeX = np.linspace(tLim[0],tLim[1],nBins+1)
        zX    = np.array([co.z_from_t(co.quan(time, "Myr")) for time in timeX])
        dZ = (zX[-1]-zX[0])/nBins

        starMass = [0]*nBins

        allStarMass = np.array(sp["PartType4","Masses"].to("Msun"))
        allStarScale = np.array(sp["PartType4","StellarFormationTime"])
        allStarZ     = 1/allStarScale - 1
        starMass, edges = np.histogram(allStarZ,bins=zX[::-1],weights=allStarMass)
        histX = edges[0:-1]+dZ
          
        for i in range(nBins):
            if i == 0: continue
            starMass[-i-1] += starMass[-i]
        
        uAx.plot(histX,starMass,".--")
        uAx.semilogy()

    if yLims != 0: uAx.set_ylim(yLims[0],yLims[1])
    if xLims != 0: uAx.set_xlim(xLims[0],xLims[1])
    
    uAx.set_ylabel("Stellar Mass From Present Stars ($\mathrm{M}_{\odot}$)")
    uAx.set_xlabel("z")
    uAx.set_title("Stellar Mass Over time")
    uAx.grid()
    
    setLegend(uAx,sims,idx)

    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

# Plot the Ms/M200 ratio over time
def plotMsMh(sims,idx,verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,message=0,yLims=[1e-5,0.25],xLims=0,animate=0):
    if message != 0: f.write(f"\n{message}\n")
    # Setup plot figures and axes
    uFig = plt.figure(figsize=(8/10*figSize, 6/10*figSize))
    uAx  = plt.subplot2grid((4,1),(0,0),rowspan=4)

    # Idx setup (creates len(sims) lists, each one with the snap numbers for a sim)
    if not all(isinstance(lIdx, list) for lIdx in idx):
        idxArrAux = [None]*len(sims)
        for j in range(len(sims)): idxArrAux[j] = idx 
        idx = idxArrAux

    # At z 8,7,6,5,4
    z_fix   =[8,7,6,5,4]
    rvir_fix=[5.77,7.52,8.43,11.43,25.2]

    co = yt.utilities.cosmology.Cosmology(hubble_constant=0.702, omega_matter=0.272,omega_lambda=0.728, omega_curvature=0.0)

    # Bin star ages into nBins and use that to estimate total mass
    for k,sn in enumerate(sims):
        if verbose > 9: f.write(f"  Started {sims[k].name}\n")
 
        Mstar = [0]*len(idx[k])
        Mhalo = [0]*len(idx[k])
        Mrati = [0]*len(idx[k])
        zList = [0]*len(idx[k])

        # Loops over the snapshots in a sim
        for i in range(len(idx[k])):
            curSnap   = sims[k].snap[idx[k][i]]
            curYTSnap = sims[k].ytFull[idx[k][i]]
            index     = np.argmin(np.abs(np.array(z_fix)-curSnap.z))
            # Uses the mean rvir if z is sufficiently close
            if np.abs(z_fix[index]-curSnap.z) < 0.2:
                f.write(f"  Using the mean rvir from AGORA data, z for this snapshot is sufficiently close (dif < 0.2)\n")
                curRvir   = rvir_fix[index]
            else:
                f.write(f"  Using the rvir calculated from the snapshot\n")
                curRvir   = sims[k].snap[idx[k][i]].rvir
            if verbose > 10: f.write(f"  - Snapshot {idx[k][i]} with t = {curSnap.time:.1f} z = {curSnap.z:.2f}\n")
            if verbose > 11: f.write(f"      Mapped to z = {z_fix[index]:.2f} rvir = {curRvir:.2f}\n")

            # Get the stellar halo and halo cutoff and calculate the total mass at this redshift
            spGal = curYTSnap.sphere(curSnap.ytcen,(0.15*curRvir, "kpc"))
            spVir = curYTSnap.sphere(curSnap.ytcen,(curRvir, "kpc"))

            zList[i] = curSnap.z
            if starPart in sims[k].snap[idx[k][i]].pType:
                Mstar[i] = spGal[(starPart,"particle_mass")].in_units("Msun").sum()
            else:
                f.write(f"  Star particles not in this snapshot, setting to 0")
                Mstar[i] = 0

            Mhalo[i] = getData(spVir,"particle_mass", sims[k].snap[idx[k][i]].pType, units="Msun").sum()
            #Mhalo[i] = spVir[("all","particle_mass")].in_units("Msun").sum()
            if verbose > 12: f.write(f"      Stellar Mass = {Mstar[i]:.2E} | Halo Mass = {Mhalo[i]:.2E}\n")
            Mrati[i] = Mstar[i]/Mhalo[i]
        
        uAx.plot(zList,Mrati,".--")
        uAx.semilogy()

    if yLims != 0: uAx.set_ylim(yLims[0],yLims[1])
    if xLims != 0: uAx.set_xlim(xLims[0],xLims[1])
    
    uAx.set_ylabel("$M_{s}/M_{h}$")
    uAx.set_xlabel("z")
    uAx.set_title("Stellar-to-Halo Mass Ratio Over Time")
    uAx.grid()
    
    uAx.legend([sims[i].name for i in range(len(sims))])

    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)
        
# Create and save a movie from a frame list
def makeMovie(frames, interval=50, verbose=verboseLevel, saveFigPath=0, message=0):
    if message != 0: f.write(f"\n{message}\n")
    # Create an animation figure using the first frame
    fig_anim, ax_anim = plt.subplots()
    im = ax_anim.imshow(frames[0], animated=True)
    ax_anim.axis('off') 
    fig_anim.tight_layout()
    fig_anim.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    
    # Create animation by setting frames
    def update_frame(i):
        im.set_array(frames[i])
        return [im]
    anime = ani.FuncAnimation(fig_anim, update_frame, frames=len(frames), interval=interval, blit=True)

    # Saves figure with message path as title
    fullPath = os.path.join(savePath,"placeholder.png")
    if message != 0:
        fullPath = os.path.join(savePath,message.replace(" ","_")+".gif")
    elif saveFigPath == 0:
        if verbose > 3: f.write(f"WARNING: TITLE NOT SPECIFIED FOR THIS FIGURE, PLEASE SPECIFY A TITLE\n")

    if saveFigPath != 0: fullPath = os.path.join(saveFigPath,message.replace(" ","_")+".gif")
    if verbose > 8: f.write(f"  Saving animation to {fullPath}\n")

    with rc_context({"mathtext.fontset": "stix"}):
        anime.save(fullPath,dpi=300)
    plt.close(fig_anim)
    return anime

# Binning postproccessing functions
def binFunctionCilBins(cil,bin):
    dr = 0.5*(cil[1]-cil[0])
    newBin = []
    for i in range(len(cil)):
        newBin.append(bin[i]/(np.pi * (((cil[i]+dr)*1e3)**2-((cil[i]-dr)*1e3)**2) ))
    return newBin

def binFunctionSphBins(cil,bin):
    dr = 0.5*(cil[1]-cil[0])
    newBin = []
    for i in range(len(cil)):
        newBin.append(bin[i]/(4/3 * np.pi * (((cil[i]+dr)*1e3)**3-((cil[i]-dr)*1e3)**3) ))
    return newBin

def binFunctionCilBinsSFR(cil,bin):
    dr = 0.5*(cil[1]-cil[0])
    bin = np.array(bin)/youngStarAge
    newBin = []
    for i in range(len(cil)):
        newBin.append(bin[i]/(np.pi * (((cil[i]+dr)*1e3)**2-((cil[i]-dr)*1e3)**2) ))
    return newBin

def makeZbinFun(rlimit):
    def binFunctionZBins(zData,bin,rLim=rlimit):
        dh = (zData[1]-zData[0])
        newBin = []
        for i in range(len(zData)):
            newBin.append(bin[i]/(4*dh*1e3*rLim*1e3))
        return newBin
    return binFunctionZBins

# Define all the possible analysis
def NSFFanalysis(simsNS,idxNS,saveFigPath,extraText=""):
    widthIsoAnalysis = 30 # Corresponds to two times the approximate disk size
    ### Bins
    # Surface Density Binned
    plotBinned(simsNS,idxNS,("particle_position_cylindrical_radius","Masses"),50,(1e-3,widthIsoAnalysis/2-1),binFunction=binFunctionCilBins,
            setUnits=("kpc","Msun"),xlims=(0,widthIsoAnalysis/2-1),ylims=(1e-1,2*1e3),plotTitle="Cylindrically binned surface density",saveFigPath=saveFigPath,
            xylabels=("Cylidrincal radius (Kpc)","Surface density ($\\frac{\mathrm{M}_{\odot}}{pc^2}$)"),message="Surface Density"+extraText)
    # Vertical Surface Density
    plotBinned(simsNS,idxNS,("z_abs","Masses"),10,(1e-3,1.4),spLim=widthIsoAnalysis/2,message="Vertical Surface Density"+extraText,saveFigPath=saveFigPath,
            binFunction=makeZbinFun(widthIsoAnalysis/2),setUnits=("kpc","Msun"),xlims=(1e-3,1.4),ylims=(1e-1,3e3),plotTitle="Vertically binned surface density",
            xylabels=("Vertical height (Kpc)","Surface density ($\\frac{\mathrm{M}_{\odot}}{pc^2}$)"))
    # Average Height
    plotBinned(simsNS,idxNS,("particle_position_cylindrical_radius","z_abs"),50,(0,widthIsoAnalysis/2-1),message="Average Height"+extraText,spLim=widthIsoAnalysis/2,
            wField=(gasPart,"Masses"),setLogs=(False,False),setUnits=("kpc","kpc"),xlims=(0,widthIsoAnalysis/2-1),ylims=(0,0.45),saveFigPath=saveFigPath,
            plotTitle="Average cylindrical vertical height",xylabels=("Cylidrincal radius (Kpc)","Average vertical height (Kpc)"))
    # Velocity Profile
    plotBinned(simsNS,idxNS,("particle_position_cylindrical_radius","particle_velocity_cylindrical_theta"),50,(0,widthIsoAnalysis/2-1),saveFigPath=saveFigPath,
            message="Velocity Profile"+extraText,spLim=widthIsoAnalysis,wField=(gasPart,"Masses"),setLogs=(False,False),setUnits=("kpc","km/s"),
            xlims=(0,widthIsoAnalysis/2-1),ylims=(0,250),plotTitle="Velocity profile",xylabels=("Cylidrincal radius (Kpc)","Rotational velocity (km/s)"))
    # Velocity Dispersion 
    plotRotDisp(simsNS,idxNS,50,widthIsoAnalysis/2,part=gasPart,titlePlot="Velocity dispersion profile",message="Velocity Dispersion"+extraText,saveFigPath=saveFigPath)
    # Gas Density PDF
    plotBinned(simsNS,idxNS,("Density","Masses"),50,(1e-29,1e-21),message="Gas Density PDF"+extraText,spLim=widthIsoAnalysis/2,setLogs=(True,True),
            setUnits=("g/cm**3","Msun"),xlims=(1e-28,1e-21),ylims=(1e4,1e9),plotTitle="Gas Density PDF",saveFigPath=saveFigPath,
            xylabels=("$\mathrm{Density\ (g/cm^3)}$","$\mathrm{Mass,}\/\mathrm{d}M\mathrm{/dlog}\/\mathrm{\\rho}\/\mathrm{(M_{\odot})}$"))
    # Temp Density PDF
    plotBinned(simsNS,idxNS,("TemperatureG3log","Masses"),50,(1e1,1e7),message="Gas Temperature PDF"+extraText,spLim=widthIsoAnalysis/2,setLogs=(True,True),
            setUnits=("K","Msun"),xlims=(1e1,1e7),ylims=0,plotTitle="Gas Temperature PDF",saveFigPath=saveFigPath,
            xylabels=("$\mathrm{Temperature\ (K)}$","$\mathrm{Mass,}\/\mathrm{d}M\mathrm{/dlog}\/\mathrm{\\rho}\/\mathrm{(M_{\odot})}$"))
    # Cylindrically binned Smoothing Length
    plotBinned(simsNS,idxNS,("particle_position_cylindrical_radius","SmoothingLength"),50,(0,widthIsoAnalysis/2-1),saveFigPath=saveFigPath,
            wField=(gasPart,"Masses"),setLogs=(False,True),setUnits=("kpc","pc"),xlims=(0,widthIsoAnalysis/2-1), message="Cylindrically binned Smoothing Length"+extraText,
            ylims=0,plotTitle="Mass weighted Smoothing Length",xylabels=("Cylidrincal radius (Kpc)","Smoothing length (pc)"))
    # Density binned Smoothing Length
    plotBinned(simsNS,idxNS,("Density","SmoothingLength"),50,(1e-29,1e-20),saveFigPath=saveFigPath,
            wField=(gasPart,"ones"),setLogs=(True,True),setUnits=("g/cm**3","pc"),xlims=0, message="Density Binned Smoothing Length"+extraText,
            ylims=0,plotTitle="Mass weighted Smoothing Length in Density Bins",xylabels=("$\mathrm{Density\ (g/cm^3)}$","Smoothing length (pc)"))
    # Smoothing Length binned Mass
    plotBinned(simsNS,idxNS,("SmoothingLength","Masses"),50,(1e1,1e5),saveFigPath=saveFigPath,
            wField=(gasPart,"Masses"),setLogs=(True,False),setUnits=("pc","Msun"),xlims=0, message="Smoothing Length binned Mass"+extraText,
            ylims=0,plotTitle="Smoothing Length PDF",xylabels=("Smoothing length (pc)","Mass (Msun)"))

    ### Phase spaces
    # Gas Phase
    ytPhasePanel(simsNS,idxNS,blackLine=1,zFields=["Density", "TemperatureG3log", "Masses"],zFieldLim = (1e3,1e8,1e-29,1e-21,10,1e7),
                message="Gas Phase"+extraText,saveFigPath=saveFigPath,zWidth=widthIsoAnalysis/2)
    # Smoothing Length Phase
    ytPhasePanel(simsNS,idxNS,zLog=1,zFields = ["Density","TemperatureG3log","SmoothingLength"],wField=0, message="Smoothing Length Phase"+extraText,
                zFieldUnits = ["g/cm**3","K","pc"],zFieldLim = (1e1,1e5,1e-29,1e-21,10,1e7),saveFigPath=saveFigPath,zWidth=widthIsoAnalysis/2)
    # Radius-Temp Phase
    ytPhasePanel(simsNS,idxNS,blackLine=1,zFields=["particle_position_cylindrical_radius", "TemperatureG3log", "Masses"],saveFigPath=saveFigPath,
                zFieldLim = (1e3,1e8,1e-3,1e2,10,1e7),message="Gas Radius Temp Phase"+extraText,zFieldUnits = ["kpc","K","Msun"],zWidth=widthIsoAnalysis/2)
    # Radius-Density Phase
    ytPhasePanel(simsNS,idxNS,blackLine=1,zFields=["particle_position_cylindrical_radius", "Density", "Masses"],saveFigPath=saveFigPath,
                zFieldLim = (1e3,1e8,1e-3,1e2,1e-28,1e-18),message="Gas Radius Density Phase"+extraText,zFieldUnits = ["kpc","g/cm**3","Msun"],zWidth=widthIsoAnalysis/2)

    ### Projections
    # Density projection
    ytProjPanel(simsNS,idxNS,twoAxis=True,message="Density Proj"+extraText,zFieldLim=(0.00001, 0.1),zWidth=widthIsoAnalysis,saveFigPath=saveFigPath)
    # Temperature Proj
    ytProjPanel(simsNS,idxNS,zField="TemperatureG3log",twoAxis=True,wField="density_squared",zWidth=widthIsoAnalysis,zFieldUnit="K",
                zFieldLim=(1e1,1e6),message="Temperature Proj"+extraText,saveFigPath=saveFigPath)
    # Elevation Map
    ytProjPanel(simsNS,idxNS,twoAxis=True,zField="elevation",wField="density",zFieldUnit="kpc",zWidth=widthIsoAnalysis,zFieldLim=(-1,1),takeLog=0,
                message="Elevation Map"+extraText,saveFigPath=saveFigPath)
    # Resolution Map
    ytProjPanel(simsNS,idxNS,twoAxis=True,zField="resolution",wField="inv_volume_sq",zFieldUnit="pc",zWidth=widthIsoAnalysis,zFieldLim=(10,1e3),takeLog=1,
                message="Resolution Map"+extraText,saveFigPath=saveFigPath)
    
def SFFanalysis(sims2,idx2,saveFigPath,extraText=""):
    widthIsoAnalysis = 30 # Corresponds to two times the approximate disk size
    ### Bins
    # Clump Mass Cumulative
    #sffHalos = findHalos(sims2,idx2,starPart,hopThresh=4e9,overWrite=False)
    #plotClumpMassF(sims2,idx2,sffHalos,saveFigPath=saveFigPath,message="Clump Mass Cumulative")
    # Star Surface Density
    plotBinned(sims2,idx2,("particle_position_cylindrical_radius","Masses"),50,(0,widthIsoAnalysis/2-1),message="Star Surface Density",part=starPart,saveFigPath=saveFigPath,
            binFunction=binFunctionCilBins,setUnits=("kpc","Msun"),xlims=(0,widthIsoAnalysis/2-1),ylims=0,plotTitle="Cylindrically binned stellar surface density",
            xylabels=("Cylidrincal radius (Kpc)","Newly Formed Stars Surface density ($\\frac{\mathrm{M}_{\odot}}{pc^2}$)"))
    # Star Velocity Profile
    plotBinned(sims2,idx2,("particle_position_cylindrical_radius","particle_velocity_cylindrical_theta"),50,(0,widthIsoAnalysis/2-1),message="Star Velocity Profile",
            part=starPart,spLim=widthIsoAnalysis/2,wField=(starPart,"Masses"),setLogs=(False,False),setUnits=("kpc","km/s"),xlims=(0,widthIsoAnalysis/2-1),ylims=0,saveFigPath=saveFigPath,
            plotTitle="Newly Formed Stars Velocity Profile",xylabels=("Cylidrincal radius (Kpc)","Rotational velocity (km/s)"))
    # Star Velocity Dispersion
    plotRotDisp(sims2,idx2,50,widthIsoAnalysis/2,ylims=0,part=starPart,titlePlot="Newly Formed Stars Velocity Dispersion Profile",saveFigPath=saveFigPath,message="Star Velocity Dispersion")
    # Surface Density
    plotBinned(sims2,idx2,("particle_position_cylindrical_radius","particle_mass_young_stars"),50,(1e-3,widthIsoAnalysis/2),message="Surface Density",saveFigPath=saveFigPath,
            part=starPart,binFunction=binFunctionCilBinsSFR,setUnits=("kpc","Msun"),xlims=(0,widthIsoAnalysis/2-1),ylims=0,plotTitle="Surface Density",
            xylabels=("Cylidrincal radius (Kpc)","$\mathrm{Star\ Formation\ Rate\ Surface\ Density\ (M_{\odot}/yr/kpc^2)}$"))
    # Total SFR
    plotSFR(sims2,idx2,saveFigPath=saveFigPath,message="Total SFR",yLims=0)
    # KS Cil Binned
    plotKScil(sims2,idx2,saveFigPath=saveFigPath,message="KS Cil Binned",rLim=widthIsoAnalysis/2)
    # KS Mock Obs
    plotKSmock(sims2,idx2,saveFigPath=saveFigPath,message="KS Mock Obs")

    ### Phase spaces
    # Metal Gas Phase
    ytPhasePanel(sims2,idx2,blackLine=0,zLog=0,zFields = ["Density","TemperatureG3log","Metallicity"],wField="Masses",saveFigPath=saveFigPath,
                zFieldUnits = ["g/cm**3","K","1"],zFieldLim = (1e-2,4e-2,1e-29,1e-21,10,1e7),message="Metal Gas Phase",zWidth=widthIsoAnalysis/2)
    # Gas Obs
    nMockBins = int(widthIsoAnalysis*1e3/lowResMock)
    ytPhasePanel(sims2,idx2, cM=mColorMap2,zFields = ["x_centered","y_centered","den_low_res"], grid=False,wField=0,saveFigPath=saveFigPath,
                zFieldUnits=0, zFieldLim = (1e0,1e3,-widthIsoAnalysis/2,widthIsoAnalysis/2,-widthIsoAnalysis/2,widthIsoAnalysis/2),xb=nMockBins,yb=nMockBins,
                message="Gas Obs", depositionAlg="cic")
    # SFR Obs
    ytPhasePanel(sims2,idx2, cM=mColorMap2,part=starPart,zFields = ["x_centered","y_centered","sfr_den_low_res"], grid=False,wField=0,saveFigPath=saveFigPath,
                zFieldUnits=0, zFieldLim = (3e-4,3e-1,-widthIsoAnalysis/2,widthIsoAnalysis/2,-widthIsoAnalysis/2,widthIsoAnalysis/2),xb=nMockBins,yb=nMockBins,
                message="SFR Obs", depositionAlg="cic")

    ### Projections
    # Star Density Proj
    ytProjPanel(sims2,idx2,bSize=400,part="PartType4",zField="particle_mass",zFieldUnit="Msun",zFieldLim=0,saveFigPath=saveFigPath, #ovHalo=sffHalos,
                message="Star Density Proj",zWidth=widthIsoAnalysis)
    # Metallicity Proj
    ytProjPanel(sims2,idx2,part="PartType0",zField="Metallicity",wField="density_squared",zFieldUnit="1",takeLog=0,zFieldLim=0,saveFigPath=saveFigPath,
                cM=mColorMap,message="Metallicity Proj",zWidth=widthIsoAnalysis)
    
### CAL-1,2,3 Suite
def CalAnalysis(simsC,idxC,saveFigPath,extraText = ""):
    meanZ    = min([simsC[i].snap[idxC[i]].z for i in range(nSims)])
    meanRvir = np.mean([simsC[i].snap[idxC[i]].rvir for i in range(nSims)])
    # Gas Density PDF
    plotBinned(simsC,idxC,("Density","Masses"),50,(1.1e-29,1e-20),message="Gas Density PDF 100kpc",spLim=100,setLogs=(True,True),
            setUnits=("g/cm**3","Msun"),xlims=(1.1e-29,1e-20),ylims=(1.1e5,1e10),plotTitle="Gas Density PDF 100kpc",blLine=1.67e-24*1,saveFigPath=saveFigPath,
            xylabels=("$\mathrm{Density\ (g/cm^3)}$","$\mathrm{Mass,}\/\mathrm{d}M\mathrm{/dlog}\/\mathrm{\\rho}\/\mathrm{(M_{\odot})}$"))
    # Gas Density PDF rvir
    plotBinned(simsC,idxC,("Density","Masses"),50,(1.1e-29,1e-20),message="Gas Density PDF rvir",spLim=meanRvir,setLogs=(True,True),
            setUnits=("g/cm**3","Msun"),xlims=(1.1e-29,1e-20),ylims=(1.1e5,1e10),plotTitle="Gas Density PDF rvir",blLine=1.67e-24*1,saveFigPath=saveFigPath,
            xylabels=("$\mathrm{Density\ (g/cm^3)}$","$\mathrm{Mass,}\/\mathrm{d}M\mathrm{/dlog}\/\mathrm{\\rho}\/\mathrm{(M_{\odot})}$"))
    # Surface Density Spherically Binned
    plotBinned(simsC,idxC,("particle_position_spherical_radius","Masses"),50,(1e-3,15),binFunction=binFunctionSphBins,
            setUnits=("kpc","Msun"),xlims=(0,14),ylims=(1.1e-6,2e1),plotTitle="Spherically binned surface density",saveFigPath=saveFigPath,
            xylabels=("Cylidrincal radius (Kpc)","Surface density ($\\frac{\mathrm{M}_{\odot}}{pc^2}$)"),message="Surface Density")
    
    projFields = ["Density","TemperatureG3log"]
    projWs     = [0,"density_squared"]
    projParts  = [gasPart,gasPart]
    projUnits  = ["g/cm**2","K"]
    projLims   = [[1.5e-4, 9.9e-2],[5e3, 9.9e5]]
    projCms    = ["viridis","magma"]
    # Checks if all the simulations have star particles
    allSnapStars = all([starPart in simsC[i].snap[idxC[i]].pType for i in range(len(simsC))])
    if allSnapStars:
        projFields = ["stardensity","Density","TemperatureG3log"]
        projWs     = [0,0,"density_squared"]
        projParts  = [starPart,gasPart,gasPart]
        projUnits  = [0,"g/cm**2","K"]
        projLims   = [[1e-1,9.95e3],[1.5e-4, 9.9e-2],[5e3, 9.9e5]]
        projCms    = [starCmap,"viridis","magma"]
        # Stellar Mass
        if meanZ < 11.5:
            plotSFmass(simsC,idxC,xLims=[meanZ-0.25,11.75],zLim=[11.5,meanZ],yLims=[1e6,1e11], splimit = 40, message="Stellar Mass"+extraText,saveFigPath=saveFigPath)
        else:
            plotSFmass(simsC,idxC,xLims=[meanZ-0.25,20.75],zLim=[20.5,meanZ],yLims=[1e6,1e11], splimit = 40, message="Stellar Mass"+extraText,saveFigPath=saveFigPath)

        # Surface Density Spherically Binned Stars
        plotBinned(simsC,idxC,("particle_position_spherical_radius","Masses"),50,(1e-3,15),saveFigPath=saveFigPath,binFunction=binFunctionSphBins,
                setUnits=("kpc","Msun"),xlims=(0,14),ylims=(1.1e-6,2e1),plotTitle="Stars Spherically binned surface density",part=starPart,
                xylabels=("Cylidrincal radius (Kpc)","Surface density ($\\frac{\mathrm{M}_{\odot}}{pc^2}$)"),message="Surface Density Star"+extraText)
    

    # Phase diagrams
    ytPhasePanel(simsC,idxC,blackLine=1,zFields=["Density", "TemperatureG3log", "Masses"],zFieldLim = (1e3,9.5e7,2e-29,1e-20,10,1e7),saveFigPath=saveFigPath,message="Gas Phase",cM="inferno",axAspect=0.5,zWidth=50)

    # Projections
    ytMultiPanel(simsC,idxC,projFields,wField=projWs,saveFigPath=saveFigPath,
                part=projParts,zFieldUnit=projUnits,zFieldLim=projLims,
                saveFig=True,axisProj=0,zWidth=200,cM=projCms,message="MultiPanel 200kpc")
    ytMultiPanel(simsC,idxC,projFields,wField=projWs,saveFigPath=saveFigPath,
                part=projParts,zFieldUnit=projUnits,zFieldLim=projLims,
                saveFig=True,axisProj=0,zWidth=80,cM=projCms,message="MultiPanel 80kpc")
    ytMultiPanel(simsC,idxC,projFields,wField=projWs,saveFigPath=saveFigPath,
                part=projParts,zFieldUnit=projUnits,zFieldLim=projLims,
                saveFig=True,axisProj=0,zWidth=20,cM=projCms,message="MultiPanel 20kpc")
    #ytMultiPanel(simsC,idxC,projFields,wField=projWs,saveFigPath=saveFigPath,
    #            part=projParts,zFieldUnit=projUnits,zFieldLim=projLims,
    #            saveFig=True,axisProj=1,zWidth=200,cM=projCms,message="MultiPanel 200kpc ax1")
    #ytMultiPanel(simsC,idxC,projFields,wField=projWs,saveFigPath=saveFigPath,
    #            part=projParts,zFieldUnit=projUnits,zFieldLim=projLims,
    #            saveFig=True,axisProj=1,zWidth=80,cM=projCms,message="MultiPanel 80kpc ax1")
    #ytMultiPanel(simsC,idxC,projFields,wField=projWs,saveFigPath=saveFigPath,
    #            part=projParts,zFieldUnit=projUnits,zFieldLim=projLims,
    #            saveFig=True,axisProj=1,zWidth=20,cM=projCms,message="MultiPanel 20kpc ax1")
    #ytMultiPanel(simsC,idxC,projFields,wField=projWs,saveFigPath=saveFigPath,
    #            part=projParts,zFieldUnit=projUnits,zFieldLim=projLims,
    #            saveFig=True,axisProj=2,zWidth=200,cM=projCms,message="MultiPanel 200kpc ax2")
    #ytMultiPanel(simsC,idxC,projFields,wField=projWs,saveFigPath=saveFigPath,
    #            part=projParts,zFieldUnit=projUnits,zFieldLim=projLims,
    #            saveFig=True,axisProj=2,zWidth=80,cM=projCms,message="MultiPanel 80kpc ax2")
    #ytMultiPanel(simsC,idxC,projFields,wField=projWs,saveFigPath=saveFigPath,
    #            part=projParts,zFieldUnit=projUnits,zFieldLim=projLims,
    #            saveFig=True,axisProj=2,zWidth=20,cM=projCms,message="MultiPanel 20kpc ax2")
    #ytMultiPanel(simsC,idxC,["stardensity","Density","TemperatureG3log"],wField=[0,0,"density_squared"],saveFigPath=saveFigPath,
    #            part=[starPart,gasPart,gasPart],zFieldUnit=[0,"g/cm**2","K"],zFieldLim=[[1e-1,9.95e3],[1.5e-4, 9.9e-2],[5e3, 9.9e5]],
    #            saveFig=True,axisProj=0,zWidth=200,cM=[starCmap,"viridis","magma"],message="MultiPanel 200kpc")
    #ytMultiPanel(simsC,idxC,["stardensity","Density","TemperatureG3log"],wField=[0,0,"density_squared"],saveFigPath=saveFigPath,
    #            part=[starPart,gasPart,gasPart],zFieldUnit=[0,"g/cm**2","K"],zFieldLim=[[1e-1,9.95e3],[1.5e-4, 9.9e-2],[5e3, 9.9e5]],
    #            saveFig=True,axisProj=0,zWidth=30,cM=[starCmap,"viridis","magma"],message="MultiPanel 30kpc")

### CAL-4 Suite
def Cal4analysis(simsC4,idxC4,saveFigPath,extraText = ""):
    meanZ    = min([simsC4[i].snap[idxC4[i]].z for i in range(nSims)])
    meanRvir = np.mean([simsC4[i].snap[idxC4[i]].rvir for i in range(nSims)])
    # Stellar Mass
    if meanZ < 11.5:
        plotSFmass(simsC4,idxC4,xLims=[meanZ-0.25,11.75],zLim=[11.5,meanZ],yLims=[1e6,1e11], splimit = 40, message="Stellar Mass"+extraText,saveFigPath=saveFigPath)
    else:
        plotSFmass(simsC4,idxC4,xLims=[meanZ-0.25,20.75],zLim=[20.5,meanZ],yLims=[1e6,1e11], splimit = 40, message="Stellar Mass"+extraText,saveFigPath=saveFigPath)

    # Gas Density PDF
    #rvir = [5.77,7.52,8.43,11.43,25.2]
    plotBinned(simsC4,idxC4,("Density","Masses"),50,(1e-29, 1e-20),message="Gas Density PDF z"+str(round(meanZ,1)),spLim=meanRvir,setLogs=(True,True),
               setUnits=("g/cm**3","Msun"),xlims=(1.1e-29,1e-20),ylims=(1e5,1e10),plotTitle="Gas Density PDF",blLine=1.67e-24*1,saveFigPath=saveFigPath,
               xylabels=("$\mathrm{Density\ (g/cm^3)}$","$\mathrm{Mass,}\/\mathrm{d}M\mathrm{/dlog}\/\mathrm{\\rho}\/\mathrm{(M_{\odot})}$"),axAspect=0.75)
    # Metallicity PDF    
    plotBinned(simsC4,idxC4,("MetallicityG3","Masses"),50,(1e-4,1e1),message="Metallicity PDF"+extraText,spLim=meanRvir,setLogs=(True,True),
                setUnits=(0,"Msun"),xlims=(1e-4,1e1),ylims=(9.9e1,9e9),plotTitle="Metallicity PDF",saveFigPath=saveFigPath,
                xylabels=("Z/Z0","Mass (Msun)"),axAspect=0.75)
    # Metallicity Star PDF    
    plotBinned(simsC4,idxC4,("MetallicityG3","Masses"),50,(1e-4,1e1),message="Metallicity Star PDF"+extraText,spLim=meanRvir,setLogs=(True,True),
                setUnits=(0,"Msun"),xlims=(1e-4,1e1),ylims=(9.9e1,9e9),plotTitle="Metallicity Star PDF",part=starPart,
                xylabels=("Z/Z0","Mass (Msun)"),axAspect=0.75,saveFigPath=saveFigPath)

    # Surface Density Spherically Binned
    plotBinned(simsC4,idxC4,("particle_position_spherical_radius","Masses"),50,(1e-3,15),saveFigPath=saveFigPath,binFunction=binFunctionSphBins,
            setUnits=("kpc","Msun"),xlims=(0,14),ylims=(1.1e-6,2e1),plotTitle="Spherically binned surface density",
            xylabels=("Cylidrincal radius (Kpc)","Surface density ($\\frac{\mathrm{M}_{\odot}}{pc^2}$)"),message="Surface Density"+extraText)

    # Surface Density Spherically Binned Stars
    plotBinned(simsC4,idxC4,("particle_position_spherical_radius","Masses"),50,(1e-3,15),saveFigPath=saveFigPath,binFunction=binFunctionSphBins,
            setUnits=("kpc","Msun"),xlims=(0,14),ylims=(1.1e-6,2e1),plotTitle="Stars Spherically binned surface density",part=starPart,
            xylabels=("Cylidrincal radius (Kpc)","Surface density ($\\frac{\mathrm{M}_{\odot}}{pc^2}$)"),message="Surface Density Star"+extraText)

    # Phase diagrams
    # Temp Den PDF
    ytPhasePanel(simsC4,idxC4,blackLine=0,zFields=["Density", "TemperatureG3log", "Masses"],saveFigPath=saveFigPath,
                 zFieldLim = (1e3,9.5e7,2e-29,1e-20,10,1e7),message="Gas Phase"+extraText,cM="inferno",axAspect=0.75,zWidth=25.4)
    # Temp Den Metal PDF
    ytPhasePanel(simsC4,idxC4,blackLine=0,zFields=["Density", "TemperatureG3log", "MetallicityMassG3"],saveFigPath=saveFigPath
                 ,zFieldLim = (1e-2,9.5e3,2e-29,1e-20,10,1e7),message="Metal Phase"+extraText,cM="turbo",axAspect=0.75,zWidth=25.4)
    # Radius Gas Phase
    ytPhasePanel(simsC4,idxC4,blackLine=0,zFields=["Density", "TemperatureG3log", "particle_position_spherical_radius"],
                saveFigPath=saveFigPath,zFieldLim = (1e-1,8e1,2e-29,1e-20,10,1e7),
                message="Distance Phase 80kpc"+extraText,cM="algae",axAspect=0.75,zWidth=80,zFieldUnits = ["g/cm**3","K","kpc"],wField="Masses")
    # Radius-Temp Phase
    ytPhasePanel(simsC4,idxC4,blackLine=1,zFields=["particle_position_cylindrical_radius", "TemperatureG3log", "Masses"],saveFigPath=saveFigPath,
                zFieldLim = (1e3,9.5e7,1e-3,1e2,10,1e7),message="Gas Radius Temp Phase"+extraText,zFieldUnits = ["kpc","K","Msun"],zWidth=100)
    # Radius-Density Phase
    ytPhasePanel(simsC4,idxC4,blackLine=1,zFields=["particle_position_cylindrical_radius", "Density", "Masses"],saveFigPath=saveFigPath,
                zFieldLim = (1e3,9.5e7,1e-3,1e2,1e-28,1e-18),message="Gas Radius Density Phase"+extraText,zFieldUnits = ["kpc","g/cm**3","Msun"],zWidth=100)

    # Projections
    ytMultiPanel(simsC4,idxC4,["stardensity","Density","TemperatureG3log","MetallicityG3"],wField=[0,0,"density_squared","density"],saveFigPath=saveFigPath,
                part=[starPart,gasPart,gasPart,gasPart],zFieldUnit=[0,"g/cm**2","K",0],zFieldLim=[[1e-1,9.95e3],[1.5e-4, 9.9e-2],[5e3, 5e6],[5e-5,1e0]],
                saveFig=True,axisProj=0,zWidth=200,cM=[starCmap,"viridis","magma","turbo"],message="MultiPanel 200kpc"+extraText)
    ytMultiPanel(simsC4,idxC4,["stardensity","Density","TemperatureG3log","MetallicityG3"],wField=[0,0,"density_squared","density"],saveFigPath=saveFigPath,
                part=[starPart,gasPart,gasPart,gasPart],zFieldUnit=[0,"g/cm**2","K",0],zFieldLim=[[1e-1,9.95e3],[1.5e-4, 9.9e-2],[5e3, 5e6],[5e-5,1e0]],
                saveFig=True,axisProj=0,zWidth=80,cM=[starCmap,"viridis","magma","turbo"],message="MultiPanel 80"+extraText)
    ytMultiPanel(simsC4,idxC4,["stardensity","Density","TemperatureG3log","MetallicityG3"],wField=[0,0,"density_squared","density"],saveFigPath=saveFigPath,
                part=[starPart,gasPart,gasPart,gasPart],zFieldUnit=[0,"g/cm**2","K",0],zFieldLim=[[1e-1,9.95e3],[1.5e-4, 9.9e-2],[5e3, 5e6],[5e-5,1e0]],
                saveFig=True,axisProj=0,zWidth=20,cM=[starCmap,"viridis","magma","turbo"],message="MultiPanel 20kpc"+extraText)

### CAL-4 Suite
def Cal4analysisProj(simsC4,idxC4,saveFigPath,extraText = ""):
    # Projections
    ytMultiPanel(simsC4,idxC4,["stardensity","Density","TemperatureG3log","MetallicityG3"],wField=[0,0,"density_squared","density"],saveFigPath=saveFigPath,
                part=[starPart,gasPart,gasPart,gasPart],zFieldUnit=[0,"g/cm**2","K",0],zFieldLim=[[1e-1,9.95e3],[1.5e-4, 9.9e-2],[5e3, 5e6],[5e-5,1e0]],
                saveFig=True,axisProj=1,zWidth=200,cM=[starCmap,"viridis","magma","turbo"],message="MultiPanel 200kpc ax1"+extraText)
    ytMultiPanel(simsC4,idxC4,["stardensity","Density","TemperatureG3log","MetallicityG3"],wField=[0,0,"density_squared","density"],saveFigPath=saveFigPath,
                part=[starPart,gasPart,gasPart,gasPart],zFieldUnit=[0,"g/cm**2","K",0],zFieldLim=[[1e-1,9.95e3],[1.5e-4, 9.9e-2],[5e3, 5e6],[5e-5,1e0]],
                saveFig=True,axisProj=1,zWidth=80,cM=[starCmap,"viridis","magma","turbo"],message="MultiPanel 80 ax1"+extraText)
    ytMultiPanel(simsC4,idxC4,["stardensity","Density","TemperatureG3log","MetallicityG3"],wField=[0,0,"density_squared","density"],saveFigPath=saveFigPath,
                part=[starPart,gasPart,gasPart,gasPart],zFieldUnit=[0,"g/cm**2","K",0],zFieldLim=[[1e-1,9.95e3],[1.5e-4, 9.9e-2],[5e3, 5e6],[5e-5,1e0]],
                saveFig=True,axisProj=1,zWidth=20,cM=[starCmap,"viridis","magma","turbo"],message="MultiPanel 20kpc ax1"+extraText)
    
    ytMultiPanel(simsC4,idxC4,["stardensity","Density","TemperatureG3log","MetallicityG3"],wField=[0,0,"density_squared","density"],saveFigPath=saveFigPath,
                part=[starPart,gasPart,gasPart,gasPart],zFieldUnit=[0,"g/cm**2","K",0],zFieldLim=[[1e-1,9.95e3],[1.5e-4, 9.9e-2],[5e3, 5e6],[5e-5,1e0]],
                saveFig=True,axisProj=2,zWidth=200,cM=[starCmap,"viridis","magma","turbo"],message="MultiPanel 200kpc ax2"+extraText)
    ytMultiPanel(simsC4,idxC4,["stardensity","Density","TemperatureG3log","MetallicityG3"],wField=[0,0,"density_squared","density"],saveFigPath=saveFigPath,
                part=[starPart,gasPart,gasPart,gasPart],zFieldUnit=[0,"g/cm**2","K",0],zFieldLim=[[1e-1,9.95e3],[1.5e-4, 9.9e-2],[5e3, 5e6],[5e-5,1e0]],
                saveFig=True,axisProj=2,zWidth=80,cM=[starCmap,"viridis","magma","turbo"],message="MultiPanel 80 ax2"+extraText)
    ytMultiPanel(simsC4,idxC4,["stardensity","Density","TemperatureG3log","MetallicityG3"],wField=[0,0,"density_squared","density"],saveFigPath=saveFigPath,
                part=[starPart,gasPart,gasPart,gasPart],zFieldUnit=[0,"g/cm**2","K",0],zFieldLim=[[1e-1,9.95e3],[1.5e-4, 9.9e-2],[5e3, 5e6],[5e-5,1e0]],
                saveFig=True,axisProj=2,zWidth=20,cM=[starCmap,"viridis","magma","turbo"],message="MultiPanel 20kpc ax2"+extraText)

### CAL-4 Suite
def Cal4analysisOT(simsC4,idxC4,saveFigPath,extraText = ""):
    # Stellar Ratio             
    plotMsMh(simsC4,idxC4,xLims=0,message="Stellar Ratio"+extraText,yLims=[1e-5,2.5e-1],saveFigPath=saveFigPath)


# Get the simulation with the minimum amount of snapshots
idxMinSnaps = np.argmin([len(sims[i].ytFull) for i in range(nSims)])
godsChosen  = sims[idxMinSnaps]
numCompari  = len(godsChosen.ytFull)

# Returns an idx list, with the closest snapshots idxs of the first simulation to the times of the second simulation
def getIdxList(simToCompare,simReference):
    lenList = len(simReference.ytFull)
    times = [simReference.snap[i].time for i in range(lenList)]
    return [tToIdx(simToCompare,times[i]) for i in range(lenList)]

# Snapshot index matrix (indices at same time for all snapshots), where [simulation,time]
minIdxMatrix = np.array([getIdxList(sims[i],godsChosen) for i in range(nSims)])

# Find the indices for the list sent if that option is used
if ztList:
    minIdxMatrix = []
    numCompari   = len(ztList)
    for i in range(nSims):
        if sims[i].cosmo:
            minIdxMatrix.append([zToIdx(sims[i],ztList[j]) for j in range(len(ztList))])
        else:
            minIdxMatrix.append([tToIdx(sims[i],ztList[j]) for j in range(len(ztList))])
    minIdxMatrix = np.array(minIdxMatrix)
    f.write(f"\nFound snapshots appropiate to the times/redshift set. Using (simulation,time): {minIdxMatrix}\n")

f.write(f"\n---------------------- \n")
f.write(f" -- SETUP COMPLETE -- \n")
f.write(f"---------------------- \n")

# Connects analysis title to function
analysisDict = {"NSFF":NSFFanalysis,"SFF":SFFanalysis,"Cal4":Cal4analysis,"Cal4Proj":Cal4analysisProj,"Cal":CalAnalysis}
# Same but for analysis that happen over time
analysisDictTime = {"Cal4OT":Cal4analysisOT}

# Does every analysis requested
for analysisRoutine in analysisList:
    f.write(f"\n-------------------------------------------------------\n")
    f.write(f"Running analysis routine: {analysisRoutine}\n")
    f.write(f"-------------------------------------------------------\n")

    routinePath = os.path.join(savePath,analysisRoutine)
    os.makedirs(routinePath, exist_ok=True)

    # Runs the part of the analysis that compares between different times
    f.write(f"\n---- ANALYSIS OVER TIME ----\n")
    if analysisRoutine in analysisDictTime.keys():
        analysisDictTime[analysisRoutine](sims,minIdxMatrix.tolist(),saveFigPath=routinePath)

    # Runs the analysis for each set of near-time snapshots
    if analysisRoutine in analysisDict.keys():
        for i in range(numCompari):
            meanTime = np.mean([sims[j].snap[minIdxMatrix[j,i]].time for j in range(nSims)])
            f.write(f"\n---- TIME: {meanTime:.2f} ----\n")
            # Makes the final path for saving figures
            figurePath = os.path.join(routinePath,f"time_{meanTime:.2f}")
            os.makedirs(figurePath, exist_ok=True)
            analysisDict[analysisRoutine](sims,minIdxMatrix[:,i],saveFigPath=figurePath)
    
    
f.write(f"\n=====================================================================================================================\n")
f.write(f"All analysis done! Copying log file to {savePath} and finishing \n")
f.close()
# Copies log file to output dir so it's not easily rewritten
shutil.copyfile(logPath,os.path.join(savePath,"log.txt"))
