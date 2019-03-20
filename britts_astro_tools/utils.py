import pandas as pd
import numpy as np
from astropy.cosmology import WMAP9

def update_progress(progress, barLength):
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Pause...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent completed: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def check_for_nans(data):
    # True if one or more nan is in data (an array), False otherwise
    return(data.isnull().values.any())


def find_nearest(array, value):
    # find the index of the element of array which is closest to value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def add_row(df, values):
    # Add a new row to a pandas dataframe. Values should be a list of values to fill each column.
    df.loc[len(df)] = values
    return(df)


def angular_size(r, z, degrees=False):
    # z: average redshift in bin; can be a np array
    # r: physical size of aperture radius in Mpc; can be a np array
    # degrees: bool, if True results in in degrees; otherwise, they're in arcmin
    # returns: corresponding angular radius of aperture in degrees

    # convert to kpc
    r = r*1000.
    # Separation in transverse comoving kpc corresponding to an arcminute at redshift z
    foo = WMAP9.kpc_comoving_per_arcmin(z)
    # theta in arcmin
    theta = (r / foo).value
    # theta in degrees
    if degrees:
        theta = theta * 0.0166667

    return(theta)
