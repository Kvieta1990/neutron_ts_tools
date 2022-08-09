from mantid.simpleapi import \
    CreateWorkspace, \
    FitPeaks, \
    Fit, \
    mtd, \
    DeleteWorkspace
import numpy as np
from scipy.signal import argrelextrema
import os
import matplotlib.pyplot as plt


def bkg_finder_TOF(all_data: list,
                   all_range: list,
                   fudge_factor: list):
    """This routine is for extracting background from input bank-by-bank data.

    :param all_data: Input bank-by-bank data
    :param all_range: Range used for each bank for final merging
    :param fudge_factor: An extra factor applied to the estimated background
    :param out_dir: Output directory
    :param out_stem: Output file stem name
    """
    for bank, bank_data in enumerate(all_data):
        if all_range[bank][1] != 0:
            x_bank = bank_data[0]
            y_bank = bank_data[1]

            ws_tmp = CreateWorkspace(x_bank, y_bank)
            x_bkg_pt = []
            y_bkg_pt = []
            naughty_region_x = x_bank
            naughty_region_y = y_bank
            bottom_tmp = argrelextrema(np.asarray(naughty_region_y), np.less, order=1)
            for item in bottom_tmp[0]:
                x_bkg_pt.append(naughty_region_x[item])
                y_bkg_pt.append(naughty_region_y[item])

            x_min = argrelextrema(np.asarray(y_bkg_pt), np.less, order=1)
            y_min = [y_bkg_pt[item] for item in x_min[0]]
            x_min = [x_bkg_pt[item] for item in x_min[0]]

            min_min = argrelextrema(np.asarray(y_min), np.less, order=1)
            y_min_min = [y_min[item] for item in min_min[0]]
            x_min_min = [x_min[item] for item in min_min[0]]

            ws_real_bkg = CreateWorkspace(x_min_min, y_min_min)

    # Remove all intermediate workspaces
    DeleteWorkspace(ws_tmp)

    return
