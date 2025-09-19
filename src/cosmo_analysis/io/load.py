# TODO: Description of file for documentation

# TODO: Description of functions for documentation

from ..core import utils
from ..core.sim_objs import *
from ..core import sim_prop
from ..core import fields
import numpy as np
import yt
import h5py
import os

# Gets data for multiple particles
def getData(selector,field,parts,units = 0):
    results = []
    for part in parts:
        if units == 0:
            results = np.concatenate((results,selector[(part,field)]))
        else:
            results = np.concatenate((results,selector[(part,field)].to(units)))
    return results

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
    methodDict = {"1":sim_prop.findCenter,"2":sim_prop.findCenter2,"3":sim_prop.findCenter3,"4":sim_prop.findCenter4,"5":sim_prop.findCenter5,"6":sim_prop.findCenter6,"7":sim_prop.findCenter7}
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
    
    # TODO: Fix logging
    #if verboseLvl > 0: f.write(f"  - Loading snapshot {idx} true index {sim.snap[idx].ytIdx} at {sim.snap[idx].time} Myr \n")
    #if cosmological:
    #    f.write(f"    With Redshit: {sim.snap[idx].z} Scale: {sim.snap[idx].a} \n")
    if sameCenter != 0:
        sim.snap[idx].center = np.array(sameCenter)
    else:
        loadCenters(sim,idx,overrideCenter=overrideCenter,fun=centerFun)
    #if verboseLvl > 1: f.write(f"    Center initialized at {sim.snap[idx].center} pc \n")
    #if verboseLvl > 2: f.write(f"    YT Center initialized at {sim.snap[idx].ytcen} \n")

    # Getting virial radius after calculating center
    if cosmological: sim.snap[idx].rvir = sim_prop.getRvir(sim,idx)
    # Now calculate the face-on and edge-on axis
    fOn,eOn = sim_prop.getAxes(sim,idx)
    sim.snap[idx].face_on = fOn
    sim.snap[idx].edge_on = eOn
    
    # Load each type of particle
    if loadAllP:
        nParts = sum(np.array(sim.ytFull[idx].parameters["NumPart_ThisFile"]) > 0)
        typeP = ["PartType"+str(i) for i in range(nParts)]
        sim.snap[idx].p = [None]*len(typeP)
        for i in range(len(typeP)):
            #f.write(f"    - Loading {typeP[i]} \n")
            loadParticles(sim,idx,i,typeP[i])

# Load a full simulation
def loadSim(sim,ytData,allowedSnaps=0,overrideCenter=0,loadAllP=0,cosmological=0,
            verboseLvl=1,sameCenter=0,centerDefs=0):
    #f.write(f"Loading from yt file: {ytData} \n")
    #f.write(f"  Loading YT snapshots into list \n")

    sim.ytFile = ytData

    # Load only some snapshots if desired
    if allowedSnaps != 0:
        sim.ytFull = [None]*len(allowedSnaps)
        #f.write(f"  Loading only snapshots: {allowedSnaps} \n")
        for i in range(len(allowedSnaps)):
            sim.ytFull[i] = ytData[allowedSnaps[i]]
            #if verboseLvl > 1: f.write(f"    Loaded yt snapshot {i} from file: {ytData[allowedSnaps[i]]} \n")
    else:
        sim.ytFull = [None]*len(ytData)
        ytStr = [None]*len(ytData)
        for i in range(len(ytData)):
            #ytStr[i] = str(ytData[i].filename)[:]
            ytStr[i] = 0
        #f.write(f"  Loading all snapshots found {ytStr} \n")    
        for i in range(len(ytData)):
            sim.ytFull[i] = ytData[i]
            #if verboseLvl > 1: f.write(f"    Loaded yt snapshot {i} from file: {ytData[i]} \n")

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
    #f.write(f"\n- Started loading simulation {name} \n")
    #f.write(f"  With cosmology {cosmological}\n")
            
    if not boundbox:
        ytDataset = yt.load(pathPattern,unit_base=unitBase)
    else:
        ytDataset = yt.load(pathPattern,unit_base=unitBase,bbox = boundbox)
    # Load simulation
    loadSim(simulation, ytDataset, verboseLvl=verboseLvl, overrideCenter=overrideCenter, cosmological = cosmological,
            allowedSnaps=allowedSnaps, centerDefs=centerDefs, sameCenter=sameCenter, loadAllP = loadP)

    # Collect snapshot and centers for adding custom fields
    snapArr = []
    snhtArr = []
    centArr = []
    for j in range(len(simulation.ytFull)):
        snapArr.append(simulation.ytFull[j])
        snhtArr.append(simulation.snap[j])
        centArr.append(simulation.snap[j].ytcen)

    fields.add_field_to_snaps(snapArr,snhtArr,centArr)

    return simulation

