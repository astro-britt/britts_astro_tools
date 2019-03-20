import numpy as np
import pandas as pd

from astropy.table import Table


def table_to_df(table):
    ''' For turning an astropy table which may or may not have some 2-dimensional columns into a pandas dataframe'''
    df = pd.DataFrame()
    for i in range(len(table.colnames)):
        shape = table.columns[i].shape
        if len(shape) is 1:
            # need byteswap().newbyteorder() for fits files
            df[table.colnames[i]] = table.columns[i].data.byteswap().newbyteorder()
        else:
            # for now this is all 2D data- generalize this later for higher-dimensional data
            col_data = table.columns[i].data
            for j in range(col_data.shape[1]):
                df[table.colnames[i]+str(j)] = col_data[:,j].byteswap().newbyteorder()
    return(df)


def fits_to_df(filename):
    ''' For turning a fits file which may or may not have some 2-dimensional columns into a pandas dataframe'''
    table = Table.read(filename, format='fits')
    return(table_to_df(table))
