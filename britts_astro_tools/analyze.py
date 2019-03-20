import pandas as pd
import numpy as np
from astropy.cosmology import WMAP9


def get_bkg_idxs(bkg, size):
    # for background, choose a number of random bkg apertures equal to the number of clusters in this group
    # Return random integers from the “discrete uniform” distribution of the specified dtype in the “half-open” interval [low, high)
    # ensure there are no repeats
    bkg_idxs = np.random.randint(high=bkg.bkg_aperture_ID.max(),
                                 low=bkg.bkg_aperture_ID.min(),
                                 size=size)
    # save the ones that are unique
    keep = np.unique(bkg_idxs)
    while len(keep) < size:
        new = np.random.randint(high=bkg.bkg_aperture_ID.max(),
                                low=bkg.bkg_aperture_ID.min(),
                                size=size-len(keep))
        new = np.unique(new)
        keep = np.unique(np.append(keep, new))
    return(keep)


def make_snr_map(bkg_counts, cluster_counts, n_clusters, nbins=100):
    # make an snr map to go with each richness bin
    # note: cluster counts is before background subtraction

    # usage example
    '''
    foo = make_snr_map(bkg_counts=results.norm_counts_bkg[0],
                         cluster_counts=results.norm_counts_cluster[0],
                         n_clusters=results.n_clusters[0], nbins=100)
    plt.imshow(foo.transpose())
    '''

    bkg_counts = np.asarray(bkg_counts).reshape((nbins, nbins))
    cluster_counts = np.asarray(cluster_counts).reshape((nbins, nbins))
    # first make sure the shapes are compatible
    assert(bkg_counts.shape == cluster_counts.shape)
    # undo the normalization used to make CMDs
    bkg_counts = np.multiply(bkg_counts, n_clusters)
    cluster_counts = np.multiply(cluster_counts, n_clusters)
    # make an empty array to hold the snr map
    # snr = np.zeros_like(bkg_counts)
    # calculate poissonian noise
    # noise = sqrt((c+b)+b)
    noise = np.sqrt(np.add(cluster_counts, np.multiply(bkg_counts, 2)))
    signal = cluster_counts - bkg_counts
    # set snr to 0 where noise is 0 -- does this make sense? or should snr be infinity here?
    snr = np.divide(signal, noise, where=noise != 0)
    assert(snr.shape == cluster_counts.shape)
    return(snr)


def make_uncer_map(bkg_counts, cluster_counts, n_clusters, nbins):
    # make an uncertainty map
    bkg_counts = np.asarray(bkg_counts).reshape((nbins, nbins))
    cluster_counts = np.asarray(cluster_counts).reshape((nbins, nbins))
    # first make sure the shapes are compatible
    assert(bkg_counts.shape == cluster_counts.shape)
    # undo the normalization used to make CMDs
    bkg_counts = np.multiply(bkg_counts, n_clusters)
    cluster_counts = np.multiply(cluster_counts, n_clusters)
    # noise = sqrt((c+b)**2+b**2)
    noise = np.sqrt(np.add(np.sqrt(cluster_counts), np.sqrt(np.multiply(bkg_counts, 2))))
    return(noise)


def gauss2(x, A1, mu1, sigma1, A2, mu2, sigma2):
    # define a 2 component gaussian mixture
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))


def gauss(x, A, mu, sigma):
    # define a gaussian
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def eval_RS_prob_proxy(color,
                       gauss_a_RS, gauss_mu_RS, gauss_sigma_RS,
                       gauss_a_BC, gauss_mu_BC, gauss_sigma_BC):
    # if this proxy is >0, this is probably a member of the red sequence
    # if it's <0, its probably a member of the blue cloud
    # can do something more sophisticated later

    # value of red sequence function at this color
    RS = gauss(color, gauss_a_RS, gauss_mu_RS, gauss_sigma_RS)
    # value of blue cloud function at this color
    BC = gauss(color, gauss_a_BC, gauss_mu_BC, gauss_sigma_BC)

    X = RS-BC
    # make sure any galaxies redder than the median of the red sequence
    # are selected as red sequence members
    X[color >= gauss_mu_RS] = 1.
    return X


def get_GDR(cutoff_mag, rs_members, uncertainty_map):
    dwarves = rs_members.query('MODEL_MAG_R > {}'.format(cutoff_mag))
    giants = rs_members.query('MODEL_MAG_R < {}'.format(cutoff_mag))
    print("Num giants: {}     Num dwarves: {}".format(len(giants), len(dwarves)))
    GDR = len(giants)/len(dwarves)
    print("Num giants: {}     Num dwarves: {}     GDR: {}".format(len(giants), len(dwarves), GDR))
    # also calculate uncertainty
    delta_giants = dwarves['uncertainty'].sum()
    delta_dwarves = giants['uncertainty'].sum()

    delta_gdr = GDR * np.sqrt((delta_giants/len(giants))**2 + (delta_dwarves/len(dwarves))**2)

    return(GDR, delta_gdr)


def setup_results_df(richness_cutoffs, rm_clusters, n_zbin):
    results = pd.DataFrame(columns=['group_num',
                                    'z_bin',
                                    'n_clusters',
                                    'bkg_idxs',
                                    'num_cluster_gals',
                                    'num_bkg_gals',
                                    'num_bkg_idxs',
                                    'norm_counts_bkg',
                                    'norm_counts_cluster',
                                    'norm_counts_subtracted',
                                    'color_hist_bkg',
                                    'color_hist_cluster',
                                    'color_hist_subtracted',
                                    'GDR',
                                    'RF',
                                    'snr_map',
                                    'uncertainty_map',
                                    'lambda_hi',
                                    'lambda_lo',
                                    'popt_RS',
                                    'pcov_RS',
                                    'popt_BC',
                                    'pcov_BC',
                                    'total_signal_plot',
                                    'blue_cloud_plot',
                                    'red_seq_plot'
                                    ])
    # start to fill in this dataframe with the easy things
    results['group_num'] = [1, 2, 3, 4, 5]
    results['z_bin'] = n_zbin
    richness_lo = []
    richness_hi = []
    for idx in results.group_num - 1:
        richness_lo.append(richness_cutoffs[idx])
        richness_hi.append(richness_cutoffs[idx+1])

    results['richness_lo'] = richness_lo
    results['richness_hi'] = richness_hi

    for idx, row in results.iterrows():
        clusters_in_group = rm_clusters[(rm_clusters.LAMBDA.between(row.richness_lo, row.richness_hi))]
        results.loc[idx, 'n_clusters'] = len(clusters_in_group)

    # set up the columns which will hold arrays
    results['color_hist_bkg'] = results['color_hist_bkg'].astype('object')
    results['color_hist_cluster'] = results['color_hist_cluster'].astype('object')
    results['color_hist_subtracted'] = results['color_hist_subtracted'].astype('object')
    results['norm_counts_bkg'] = results['norm_counts_bkg'].astype('object')
    results['total_signal_plot'] = results['total_signal_plot'].astype('object')
    results['blue_cloud_plot'] = results['blue_cloud_plot'].astype('object')
    results['red_seq_plot'] = results['red_seq_plot'].astype('object')
    results['uncertainty_map'] = results['uncertainty_map'].astype('object')
    results['snr_map'] = results['snr_map'].astype('object')
    results['pcov_RS'] = results['pcov_RS'].astype('object')
    results['pcov_BC'] = results['pcov_BC'].astype('object')
    results['popt_RS'] = results['popt_RS'].astype('object')
    results['popt_BC'] = results['popt_BC'].astype('object')

    return(results)
