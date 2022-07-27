from bkg_finder_TOF import bkg_finder_TOF
import os

# Testing `bkg_finder` with Si data
bank_range = [[0., 0.], [900., 12000.], [1400., 20000.], [2000., 18000.], [2000., 14000.], [0., 0.]]
qmax_bkg_est = [0., 12000., 20000., 18000., 14000., 0.]
fudge_factor = [1., 1., 0.1, 0.7, 0.7, 1.]
all_data = []
for i in range(len(bank_range)):
    x_tmp = []
    y_tmp = []
    with open(os.path.join("./test_data", "Si_test-" + str(i) + ".xye"), "r") as f:
        print("[Info] Reading in file-" + str(i + 1))
        for j in range(6):
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
bkg_finder_TOF(all_data, bank_range, fudge_factor)
