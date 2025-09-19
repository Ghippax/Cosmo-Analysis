# TODO: Description of file for documentation

# TODO: Description of functions for documentation

import numpy as np

import yt

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