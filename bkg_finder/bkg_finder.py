from mantid.simpleapi import \
    CreateWorkspace, \
    FitPeaks, \
    Fit, \
    mtd, \
    DeleteWorkspace
import numpy as np
from scipy.signal import argrelextrema
import os


def bkg_finder(all_data: list,
               all_range: list,
               fudge_factor: list,
               out_dir: str,
               out_stem: str):
    """This routine is for extracting background from input bank-by-bank data.

    :param all_data: Input bank-by-bank data
    :param all_range: Range used for each bank for final merging
    :param fudge_factor: An extra factor applied to the estimated background
    :param out_dir: Output directory
    :param out_stem: Output file stem name
    """
    x_out = []
    y_out = []
    y_bkg_out = []
    for bank, bank_data in enumerate(all_data):
        if all_range[bank][1] != 0:
            x_bank = bank_data[0]
            y_bank = bank_data[1]

            y_bank = np.asarray(y_bank)
            if bank == 4:
                x_left = []
                x_max_init = np.arange(x_bank[0], x_bank[-1], 0.1)
                for item in x_max_init:
                    x_left.append(str(item - 0.1))
                    x_left.append(str(item + 0.1))
                x_max = [str(item) for item in x_max_init]
            else:
                x_max = argrelextrema(y_bank, np.greater, order=1)
                x_left = []
                for item in x_max[0]:
                    x_left.append(str(x_bank[item] - 0.1))
                    x_left.append(str(x_bank[item] + 0.1))
                x_max = [str(x_bank[item]) for item in x_max[0]]

            centers = ','.join(x_max)
            boundary = ','.join(x_left)

            ws_tmp = CreateWorkspace(x_bank, y_bank)

            FitPeaks(InputWorkspace=ws_tmp,
                     StartWorkspaceIndex=0,
                     StopWorkspaceIndex=0,
                     PeakCenters=centers,
                     FitWindowBoundaryList=boundary,
                     PeakFunction="Gaussian",
                     BackgroundType="Quadratic",
                     FitFromRight=True,
                     HighBackground=False,
                     OutputWorkspace="ws_tmp_out",
                     OutputPeakParametersWorkspace="ws_tmp_param_out",
                     FittedPeaksWorkspace="ws_tmp_fit")

            x_bkg_pt = []
            y_bkg_pt = []
            for i in range(mtd['ws_tmp_param_out'].rowCount()):
                a0_tmp = mtd['ws_tmp_param_out'].row(i)["A0"]
                a1_tmp = mtd['ws_tmp_param_out'].row(i)["A1"]
                a2_tmp = mtd['ws_tmp_param_out'].row(i)["A2"]

                x_tmp = mtd['ws_tmp_param_out'].row(i)["PeakCentre"]
                y_tmp = a0_tmp + a1_tmp * x_tmp + a2_tmp * x_tmp**2.

                x_bkg_pt.append(x_tmp)
                y_bkg_pt.append(y_tmp)

            naughty_region_x = []
            naughty_region_y = []
            for count, x_e_tmp in enumerate(x_bank):
                if x_e_tmp > float(x_max[-1]):
                    naughty_region_x.append(x_e_tmp)
                    naughty_region_y.append(y_bank[count])
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

            if bank == 4:
                print("Debugging -> ", x_min_min)
                print("Debugging -> ", y_min_min)

            c_factor = [1., 10., 0.1, 0.05, 0.01, 0.001]
            for c_factor_try in c_factor:
                c_init = c_factor_try * 0.1
                Fit(f"name=UserFunction, Formula=a-b*exp(-c*x*x), a=1, b=0.1, c={c_init}",
                    ws_real_bkg,
                    Output='ws_real_bkg_fitted')
                c_err = mtd['ws_real_bkg_fitted_Parameters'].row(2)["Error"]
                print("Debugging ->", bank, c_err)
                if c_err != 0.:
                    break

            a_init = mtd['ws_real_bkg_fitted_Parameters'].row(0)["Value"]
            b_init = mtd['ws_real_bkg_fitted_Parameters'].row(1)["Value"]
            c_init = mtd['ws_real_bkg_fitted_Parameters'].row(2)["Value"]
            c_used = fudge_factor[bank] * c_init

            y_bkg = [a_init - b_init * np.exp(-c_used * item**2.) for item in x_bank]
            for i in range(len(x_bank)):
                if all_range[bank][0] <= x_bank[i] < all_range[bank][1]:
                    x_out.append(x_bank[i])
                    y_out.append(y_bank[i] - y_bkg[i])
                    y_bkg_out.append(y_bkg[i])

            ws_bkg_final = CreateWorkspace(x_out, y_bkg_out)

    # Output merged data with background subtracted.
    file_out = open(os.path.join(out_dir, out_stem + ".sq"), "w")
    num_pts = len(x_out)
    file_out.write(f"{num_pts}\n")
    file_out.write("# Merged S(Q)\n")
    for i in range(len(x_out)):
        file_out.write("{0:10.5F}{1:20.5F}\n".format(x_out[i], y_out[i]))
    file_out.close()

    # Remove all workspaces
    # DeleteWorkspace(ws_tmp)
    # DeleteWorkspace("ws_tmp_param_out")
    # DeleteWorkspace("ws_tmp_fit")
    # DeleteWorkspace("ws_real_bkg")
    # DeleteWorkspace("ws_bkg_final")
    # DeleteWorkspace("ws_tmp_out")
    # DeleteWorkspace("ws_real_bkg_fitted_Workspace")
    # DeleteWorkspace("ws_real_bkg_fitted_Parameters")
    # DeleteWorkspace("ws_real_bkg_fitted_NormalisedCovarianceMatrix")

    return

