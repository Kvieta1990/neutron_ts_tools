import sys
import os
import json
import numpy as np
from sklearn.cluster import KMeans


def abs_grouping(sam_abs_ws,
                 con_abs_ws,
                 abs_file,
                 abs_c_file,
                 num_clusters,
                 group_out_file,
                 group_ref_det_out_file):
    Rebin(InputWorkspace=sam_abs_ws,
          OutputWorkspace="abs_pbp",
          Params="0.1,0.05,3.0")
    
    num_spectra = mtd['abs_pbp'].getNumberHistograms()
    num_monitors = int(np.sum(mtd['abs_pbp'].detectorInfo().detectorIDs() < 0))
    
    print("[Info] Clustering the absorption spectra for all detectors...")
    all_spectra = list()
    all_spectra_id = dict()
    for i in range(num_monitors, num_spectra):
        y_tmp = mtd['abs_pbp'].readY(i)
        all_spectra.append(y_tmp)
        all_spectra_id[i] = mtd['abs_pbp'].getDetector(i).getID()
    
    X = np.array(all_spectra)
    clustering = KMeans(n_clusters=num_clusters).fit(X)
    
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("[Info] Done with the clustering of absorption spectra.")
    
    CreateGroupingWorkspace(InputWorkspace="abs_pbp",
                            OutputWorkspace="grouping")
    grouping = mtd["grouping"]
    for i, label in enumerate(labels):
        if label == -1:
            continue
        det_id = all_spectra_id[i + num_monitors]
        det_id = mtd['abs_pbp'].detectorInfo().indexOf(det_id) - num_monitors
        grouping.setY(det_id, [int(label + 1)])
    
    SaveDetectorsGrouping(InputWorkspace=grouping,
                          OutputFile=group_out_file)
    GroupDetectors(InputWorkspace=sam_abs_ws,
                   OutputWorkspace="sam_abs_grouped", Behaviour="Average",
                   CopyGroupingFromWorkspace=grouping)
    SaveNexus(InputWorkspace="sam_abs_grouped",
              Filename=abs_file)
    if con_abs_ws != "":
        GroupDetectors(InputWorkspace=con_abs_ws,
                       OutputWorkspace="con_abs_grouped", Behaviour="Average",
                       CopyGroupingFromWorkspace=grouping)
        SaveNexus(InputWorkspace="con_abs_grouped",
                  Filename=abs_c_file)
    
    print("[Info] Check the similarity of absorption spectra within groups.")
    ref_id = list()
    for i in range(num_clusters):
        ref_id.append(list(labels).index(i))
    with open(group_ref_det_out_file, "w") as f:
        for i, item in enumerate(ref_id):
            if i == len(ref_id) - 1:
                f.write("{0:d}".format(item))
            else:
                f.write("{0:d}\n".format(item))
    
    perct_max_tmp = list()
    for i, group in enumerate(labels):
        perct_tmp = list()
        for j in range(len(all_spectra[i])):
            top = abs(all_spectra[i][j] - all_spectra[ref_id[group]][j])
            bottom = all_spectra[ref_id[group]][j]
            perct_tmp.append(top / bottom * 100.)
        perct_max_tmp.append(max(perct_tmp))
    perct_max = max(perct_max_tmp)
    
    print(f"[Info] # of clusters: {n_clusters_}")
    print(f"[Info] # of noise: {n_noise_}")
    print("[Info] Maximum percentage difference: {0:3.1F}".format(perct_max))


cache_dir = sys.argv[1]
ipts = sys.argv[2]
cfg_file = sys.argv[3]

with open(cfg_file, "r") as f:
    cfg_dict = json.load(f)
env_name = cfg_dict["EnvironmentName"]
try:
    gen_group_num = int(cfg_dict["ReGenerateGrouping"])
except Exception as e:
    print(e)
    sys.exit()
tmin = cfg_dict["TMIN"]
tmax = cfg_dict["TMAX"]
inst_ref_rn = cfg_dict["InstrumentGeometryRun"]
sam_ele_size = cfg_dict["SampleElementSize"]
con_ele_size = cfg_dict["ContainerElementSize"]

abs_group_gen_file = None
abs_ms_methods = {
    "SO": "SampleOnly",
    "SC": "SampleAndContainer",
    "FPP": "FullPaalmanPings",
    "None": None
}

info_file = os.path.join(cache_dir, f"IPTS-{ipts}", "abs_calc_sample_info.csv")
with open(info_file, "r") as f:
    lines = f.readlines()[1:]
    for i, line in enumerate(lines):
        abs_cache_fn = line.split(",")[2].strip().replace(" ", "_").replace(".", "p")
        tmp_fn = "_md_" + "{0:7.5F}".format(float(line.split(",")[3].strip())).replace(".", "p")
        abs_cache_fn += tmp_fn
        abs_cache_fn += ("_pf_" + "{0:5.3F}".format(float(line.split(",")[4].strip())).replace(".", "p"))
        abs_cache_fn += ("_sp_" + line.split(",")[6].strip())
        abs_cache_fn += ("_r_" + "{0:6.4F}".format(float(line.split(",")[7].strip())).replace(".", "p"))
        abs_cache_fn += ("_h_" + "{0:3.1F}".format(float(line.split(",")[8].strip())).replace(".", "p"))
        abs_cache_fn += ("_env_" + env_name)
        abs_cache_fn += ("_cont_" + line.split(",")[5].strip())
        abs_cache_fn += ("_" + line.split(",")[9].strip() + "_" + line.split(",")[10].strip())

        if gen_group_num > 0 and i == 0:
            abs_group_gen_file = abs_cache_fn + "_s.nxs"

        abs_method = line.split(",")[9].strip()
        central_cache_f_s = os.path.join(cache_dir, f"IPTS-{ipts}", abs_cache_fn + "_s.nxs")
        central_cache_f_s_g = os.path.join(cache_dir, f"IPTS-{ipts}", abs_cache_fn + "_s_g.nxs")
        central_cache_f_c = os.path.join(cache_dir, f"IPTS-{ipts}", abs_cache_fn + "_c.nxs")
        central_cache_f_c_g = os.path.join(cache_dir, f"IPTS-{ipts}", abs_cache_fn + "_c_g.nxs")

        f_s_exists = os.path.exists(central_cache_f_s)
        f_c_exists = os.path.exists(central_cache_f_c)

        redo_cond_1 = not f_s_exists
        so = abs_method == "SO"
        redo_cond_2 = f_s_exists and (not f_c_exists) and (not so)

        if redo_cond_1 or redo_cond_2:
            from mantid.simpleapi import SaveNexus
            from total_scattering.file_handling.load import create_absorption_wksp

            sam_scans = inst_ref_rn
            sam_abs_corr_type = abs_ms_methods[abs_method]
            sam_geo_dict = {
                "Shape": line.split(",")[6].strip(),
                "Radius": float(line.split(",")[7].strip()),
                "Height": float(line.split(",")[8].strip())
            }
            sam_mat_dict = {
                "ChemicalFormula": line.split(",")[2].strip(),
                "SampleMassDensity": float(line.split(",")[3].strip())
            }
            sam_env_dict = {
                "Name": env_name,
                "Container": line.split(",")[5].strip()
            }
            sam_ms_method = abs_ms_methods[line.split(",")[10].strip()]
            sam_elementsize = sam_ele_size
            con_elementsize = con_ele_size
            config = {
                "AlignAndFocusArgs": {
                    "TMin": tmin,
                    "TMax": tmax
                }
            }

            sam_abs_ws, con_abs_ws = create_absorption_wksp(
                sam_scans,
                sam_abs_corr_type,
                sam_geo_dict,
                sam_mat_dict,
                sam_env_dict,
                ms_method=sam_ms_method,
                elementsize=sam_elementsize,
                con_elementsize=con_elementsize,
                **config)
            SaveNexus(InputWorkspace=sam_abs_ws,
                      Filename=central_cache_f_s)
            if con_abs_ws != "":
                SaveNexus(InputWorkspace=con_abs_ws,
                          Filename=central_cache_f_c)
        else:
            print("[Info] Pre-calculated absorption spectra already exists.")

    print("[Info] Done processing all the samples in current IPTS.")

if gen_group_num > 0:
    msg_1 = "[Info] To generate grouping scheme according to the similarity "
    msg_2 = "in the calculated absorption spectra across all detectors."
    print(msg_1 + msg_2)
    from mantid.simpleapi import \
        Load, \
        mtd, \
        Rebin, \
        CreateGroupingWorkspace, \
        SaveDetectorsGrouping, \
        GroupDetectors, \
        LoadNexus, \
        SaveNexus

    group_out_f = os.path.join(os.path.dirname(cfg_file), "abs_grouping.xml")
    group_out_ref_det_f = os.path.join(os.path.dirname(cfg_file), "abs_grouping_ref_dets.txt")
    if not (redo_cond_1 or redo_cond_2):
        sam_abs_ws = LoadNexus(Filename=central_cache_f_s)
        if os.path.isfile(central_cache_f_c):
            con_abs_ws = LoadNexus(Filename=central_cache_f_c)
        else:
            con_abs_ws = ""

    abs_grouping(sam_abs_ws,
                 con_abs_ws,
                 central_cache_f_s_g,
                 central_cache_f_c_g,
                 gen_group_num,
                 group_out_f,
                 group_out_ref_det_f)

    print("[Info] Group file saved to,")
    print("[Info] ", group_out_f)
else:
    print("[Info] No new grouping scheme will be generated.")
    print("[Warning] Please check the existence of the following file,")
    group_out_f = os.path.join(os.path.dirname(cfg_file), "abs_grouping.xml")
    print("[Warning] ", group_out_f)
