from bkg_finder import bkg_finder
import os

# Testing `bkg_finder` with Si data
bank_range = [[0., 0.], [1., 1.75], [1.75, 3.5], [3.5, 5.3], [5.3, 40.], [0., 0.]]
qmax_bkg_est = [0., 25., 25., 25., 40., 0.]
fudge_factor = [1., 1., 0.1, 0.7, 0.7, 1.]
all_data = []
for i in range(len(bank_range)):
    x_tmp = []
    y_tmp = []
    with open(os.path.join("./test_data", "Si_test_bank" + str(i + 1) + ".dat"), "r") as f:
        print("[Info] Reading in file-" + str(i + 1))
        line = f.readline()
        line = f.readline()
        while line:
            line = f.readline()
            if line:
                if line.split()[1] != "nan":
                    if bank_range[i][0] <= float(line.split()[0]) < qmax_bkg_est[i]:
                        x_tmp.append(float(line.split()[0]))
                        y_tmp.append(float(line.split()[1]))
    all_data.append([x_tmp, y_tmp])

print("[Info] Extracting background for all banks and merging all banks...")
bkg_finder(all_data, bank_range, fudge_factor, out_dir="./", out_stem="Si_test_merged")
