import os
import sys
import subprocess
from h5py import File
from pystog.stog import StoG
import json
from mantid.simpleapi import \
    CreateWorkspace, \
    FitPeaks, \
    Fit, \
    mtd, \
    DeleteWorkspace
import numpy as np
from scipy.signal import argrelextrema

def extractor(nexus_file: str, wks_name: str, out_dir: str, dir_name=None):
    '''Method for extracting workspace from nexus file.
    '''

    def extract_from_input_file(input_file):
        wks_list = list()
        data = File(input_file, "r")
        for name, _ in data.items():
            index = os.path.join(name, "title").replace("\\", "/")
            wks_list.append(str(data[index][()][0]).split("'")[1].split("'")[0])

        title = wks_list[0]

        for name, _ in data.items():
            index = os.path.join(name, "title").replace("\\", "/")
            if data[index][(0)].decode("UTF-8") == title:
                ypath = os.path.join("/", name,
                                     "workspace", "values").replace("\\", "/")
                break

        return wks_list, len(data[ypath][()])

    if dir_name:
        nexus_file = os.path.join(dir_name, nexus_file)

    wks_list, num_banks = extract_from_input_file(nexus_file)

    if wks_name not in wks_list:
        return False

    _, tail = os.path.split(nexus_file)
    stog = StoG(**{"Outputs": {"StemName": out_dir + "/"}})
    all_files = list()

    for i in range(num_banks):
        stog.read_nexus_file_by_bank(nexus_file, i, wks_name)
        output_file = "{}_bank{}".format(tail.split(".")[0], i + 1)
        os.rename(os.path.join(out_dir, wks_name + "_bank" + str(i) + ".dat"),
                  os.path.join(out_dir, output_file + ".dat"))
        all_files.append(output_file)

    return (all_files, num_banks)

def load_merge_config(num_banks: int, input_file: str, out_dir: str):
    '''Method for loading in the merge config file.
    '''

    f = open(input_file, "r")
    merge_config_loaded = json.load(f)
    f.close()

    merge_config = dict()
    if "RemoveBkg" not in merge_config_loaded.keys():
        merge_config["RemoveBkg"] = True
    else:
        merge_config["RemoveBkg"] = merge_config_loaded["RemoveBkg"]
    for i in range(num_banks):
        bank = str(i + 1)
        merge_config[bank] = {}
        if bank in merge_config_loaded.keys():
            if "Qmin" in merge_config_loaded[bank].keys():
                merge_config[bank]["Qmin"] = merge_config_loaded[bank]["Qmin"]
            else:
                merge_config[bank]["Qmin"] = "0.0"
            if "Qmax" in merge_config_loaded[bank].keys():
                merge_config[bank]["Qmax"] = merge_config_loaded[bank]["Qmax"]
            else:
                merge_config[bank]["Qmax"] = "0.0"
            if "Yoffset" in merge_config_loaded[bank].keys():
                merge_config[bank]["Yoffset"] = merge_config_loaded[bank]["Yoffset"]
            else:
                merge_config[bank]["Yoffset"] = "0.0"
            if "Yscale" in merge_config_loaded[bank].keys():
                merge_config[bank]["Yscale"] = merge_config_loaded[bank]["Yscale"]
            else:
                merge_config[bank]["Yscale"] = "1.0"
        else:
            merge_config[bank]["Qmin"] = "0.0"
            merge_config[bank]["Qmax"] = "0.0"
            merge_config[bank]["Yoffset"] = "0.0"
            merge_config[bank]["Yscale"] = "1.0"

    out_file = os.path.join(out_dir, os.path.split(input_file)[1])
    out_f = open(out_file, "w")
    json.dump(merge_config, out_f, indent=2)
    out_f.close()

    return merge_config

def load_pystog_config(input_file: str, out_dir: str, nexus_file: str):
    '''Method for loading in the pystog config file.
    '''

    f = open(input_file, "r")
    pystog_config = json.load(f)
    f.close()

    stem_name = os.path.split(nexus_file)[1].split(".")[0]

    merged_file = os.path.join(out_dir, stem_name + "_merged.sq")
    list_tmp = pystog_config["Files"]
    list_tmp[0]["Filename"] = merged_file
    pystog_config["Files"] = list_tmp

    pystog_config["Outputs"]["StemName"] = os.path.join(out_dir, stem_name + "_merged")

    out_file = os.path.join(out_dir, os.path.split(input_file)[1])
    out_f = open(out_file, "w")
    json.dump(pystog_config, out_f, indent=2)
    out_f.close()

    return out_file

def bkg_finder(all_data: list,
               all_range: list,
               fudge_factor: list):
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

            c_factor = [1., 10., 0.1, 0.05, 0.01, 0.001]
            for c_factor_try in c_factor:
                c_init = c_factor_try * 0.1
                Fit(f"name=UserFunction, Formula=a-b*exp(-c*x*x), a=1, b=0.1, c={c_init}",
                    ws_real_bkg,
                    Output='ws_real_bkg_fitted')
                c_err = mtd['ws_real_bkg_fitted_Parameters'].row(2)["Error"]
                if c_err != 0. and c_err != float("inf") and c_err != float("-inf"):
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

    # Remove all workspaces
    DeleteWorkspace(ws_tmp)
    DeleteWorkspace("ws_tmp_param_out")
    DeleteWorkspace("ws_tmp_fit")
    DeleteWorkspace("ws_real_bkg")
    DeleteWorkspace("ws_tmp_out")
    DeleteWorkspace("ws_real_bkg_fitted_Workspace")
    DeleteWorkspace("ws_real_bkg_fitted_Parameters")
    DeleteWorkspace("ws_real_bkg_fitted_NormalisedCovarianceMatrix")

    return x_out, y_out, y_bkg_out

def merge_banks(num_banks: int, out_dir: str, nexus_file: str, merge_config: dict):

    banks_x = []
    banks_y = []
    stem_name = os.path.split(nexus_file)[1].split(".")[0]

    for bank in range(num_banks):
        banks_x.append([])
        banks_y.append([])

        file_name = os.path.join(out_dir, stem_name + "_bank" + str(bank + 1) + ".dat")
        with open(file_name, "r") as f:
            lines = f.readlines()[2:]
        for line in lines:
            banks_x[bank].append(float(line.split()[0]))
            banks_y[bank].append(float(line.split()[1]))

    qmin_list = list()
    qmax_list = list()
    qmax_max = 0.
    qmax_max_bank = 0
    valid_region = False
    for bank in range(num_banks):
        qmin_tmp = merge_config[str(bank + 1)]['Qmin']
        qmax_tmp = merge_config[str(bank + 1)]['Qmax']
        if qmin_tmp.strip() == "" or qmax_tmp.strip() == "":
            qmin_list.append(0.)
            qmax_list.append(0.)
        else:
            qmin_tmp = float(qmin_tmp)
            qmax_tmp = float(qmax_tmp)
            if qmin_tmp == qmax_tmp:
                qmin_list.append(0.)
                qmax_list.append(0.)
            elif qmin_tmp > qmax_tmp:
                msg_p1 = f"[Error] Qmax smaller than Qmin for bank-{bank+1}. "
                msg_p2 = "Please input valid values and try again."
                print(msg_p1 + msg_p2)
                return
            else:
                valid_region = True
                qmin_list.append(qmin_tmp)
                qmax_list.append(qmax_tmp)
                if qmax_tmp > qmax_max:
                    qmax_max = qmax_tmp
                    qmax_max_bank = bank

    if not valid_region:
        print("[Error] Qmin and Qmax values are all zero for all banks.")
        print("[Error] Please input valid values and try again.")
        return

    remove_bkg = merge_config["RemoveBkg"]
    if not remove_bkg:
        x_merged = list()
        y_merged = list()

        for bank in range(num_banks):
            yoffset_tmp = merge_config[str(bank + 1)]['Yoffset']
            yscale_tmp = merge_config[str(bank + 1)]['Yscale']
            if yoffset_tmp.strip() == "":
                yoffset_tmp = 0.0
            if yscale_tmp.strip() == "":
                yscale_tmp = 1.0
            yoffset_tmp = float(yoffset_tmp)
            yscale_tmp = float(yscale_tmp)
            if bank == qmax_max_bank:
                for i, x_val in enumerate(banks_x[bank]):
                    if qmin_list[bank] <= x_val <= qmax_list[bank]:
                        x_merged.append(x_val)
                        y_merged.append(banks_y[bank][i] / yscale_tmp + yoffset_tmp)
            else:
                for i, x_val in enumerate(banks_x[bank]):
                    if qmin_list[bank] <= x_val < qmax_list[bank]:
                        x_merged.append(x_val)
                        y_merged.append(banks_y[bank][i] / yscale_tmp + yoffset_tmp)
    else:
        bank_range = list()
        yscale_list = list()
        yoffset_list = list()
        all_data = list()
        x_merged_raw = list()
        y_merged_raw = list()
        # TODO: The hard coded `qmax_bkg_est` and `fudge_factor` needs to be
        # updated to adapt to general way of grouping detectors into banks.
        if num_banks == 6:
            qmax_bkg_est = [25., 25., 25., 25., 40., 0.]
            fudge_factor = [1., 1., 1., 0.7, 0.7, 1.]
        elif num_banks == 1:
            qmax_bkg_est = [40.]
            fudge_factor = [0.7]
        else:
            qmax_bkg_est = [25. for _ in range(num_banks)]
            qmax_bkg_est[-1] = 0.
            qmax_bkg_est[-2] = 40.
            fudge_factor = [1. for _ in range(num_banks)]
        for bank in range(num_banks):
            bank_range.append([qmin_list[bank], qmax_list[bank]])
            yoffset_tmp = merge_config[str(bank + 1)]['Yoffset']
            yscale_tmp = merge_config[str(bank + 1)]['Yscale']
            if yoffset_tmp.strip() == "":
                yoffset_tmp = 0.0
            if yscale_tmp.strip() == "":
                yscale_tmp = 1.0
            yscale_list.append(float(yscale_tmp))
            yoffset_list.append(float(yoffset_tmp))
            x_tmp = list()
            y_tmp = list()
            if bank == qmax_max_bank:
                for i, x_val in enumerate(banks_x[bank]):
                    if qmin_list[bank] <= x_val <= qmax_bkg_est[bank]:
                        x_tmp.append(x_val)
                        y_tmp.append(banks_y[bank][i])
                    if qmin_list[bank] <= x_val <= qmax_list[bank]:
                        x_merged_raw.append(x_val)
                        y_merged_raw.append(banks_y[bank][i])

            else:
                for i, x_val in enumerate(banks_x[bank]):
                    if qmin_list[bank] <= x_val < qmax_bkg_est[bank]:
                        x_tmp.append(x_val)
                        y_tmp.append(banks_y[bank][i])
                    if qmin_list[bank] <= x_val < qmax_list[bank]:
                        x_merged_raw.append(x_val)
                        y_merged_raw.append(banks_y[bank][i])
            all_data.append([x_tmp, y_tmp])

        x_merged_init, y_merged_init, y_bkg_out = bkg_finder(all_data, bank_range, fudge_factor)
        x_merged = x_merged_init
        y_merged = list()
        for i, x_val in enumerate(x_merged):
            if i == len(x_merged) - 1:
                for j, b_r in enumerate(bank_range):
                    if x_val == b_r[1]:
                        y_merged.append(y_merged_init[i] / yscale_list[j] + yoffset_list[j])
            for j, b_r in enumerate(bank_range):
                if b_r[0] <= x_val < b_r[1]:
                    y_merged.append(y_merged_init[i] / yscale_list[j] + yoffset_list[j])

    if len(x_merged) == 0:
        print("[Error] Qmin and Qmax values are all zero for all banks.")
        print("[Error] Please input valid values and try again.")
        return

    merged_data_ref = stem_name + '_merged.sq'

    merge_data_out = os.path.join(out_dir, merged_data_ref)
    merge_f = open(merge_data_out, "w")
    merge_f.write("{0:10d}\n\n".format(len(x_merged)))
    for i, item in enumerate(x_merged):
        merge_f.write("{0:10.3F}{1:20.6F}\n".format(item, y_merged[i]))
    merge_f.close()

if __name__ == "__main__":
    num_args = len(sys.argv)
    if num_args == 6:
        nexus_file = sys.argv[1]
        wks_name = sys.argv[3]
        out_dir = sys.argv[2]
        merge_input = sys.argv[4]
        pystog_input = sys.argv[5]
        extractor_out = extractor(nexus_file=nexus_file,
                                  wks_name=wks_name,
                                  out_dir=out_dir)
    elif num_args == 7:
        nexus_file = sys.argv[2]
        wks_name = sys.argv[4]
        out_dir = sys.argv[3]
        dir_name = sys.argv[1]
        merge_input = sys.argv[5]
        pystog_input = sys.argv[6]
        extractor_out = extractor(nexus_file=nexus_file,
                                  wks_name=wks_name,
                                  out_dir=out_dir,
                                  dir_name=dir_name)

    if extractor_out:
        merge_config = load_merge_config(input_file=merge_input,
                                         num_banks=extractor_out[1],
                                         out_dir=out_dir)
        pystog_file = load_pystog_config(input_file=pystog_input,
                                         out_dir=out_dir,
                                         nexus_file=nexus_file)
        merge_banks(num_banks=extractor_out[1],
                    out_dir=out_dir,
                    nexus_file=nexus_file,
                    merge_config=merge_config)
        print("[Info] pystog in progress...")
        subprocess.run(["pystog_cli", "--json", pystog_file])
        print("[Info] pystog job done!")
    else:
        sys.exit("[Error] NeXus file extraction failed.")
