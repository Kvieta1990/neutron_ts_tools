import sys
import os
import json
import numpy as np

# ================== Hard coded params ==================
# --> Container definition
container_dict = { "QUARTZ03": { "Shape": "Cylinder",
                                 "Radius": 0.14
                               },
                   "PAC03": { "Shape": "Cylinder",
                              "Radius": 0.135
                            },
                   "PAC06": { "Shape": "Cylinder",
                              "Radius": 0.295
                            },
                   "PAC08": { "Shape": "Cylinder",
                              "Radius": 0.385
                            },
                   "PAC10": { "Shape": "Cylinder",
                              "Radius": 0.46
                            }
                 }
# <-- Container definition
#
# --> Environment name
env_name_def = "InAir"
# <-- Environment name
#
# --> Cache directory
ipts = sys.argv[2]
config="/SNS/NOM/shared/autoreduce/configs/mts_abs_pre_calc.json"
with open(config, "r") as f:
    c_json = json.load(f)
    cache_dir = c_json["CacheDir"] + "/IPTS-" + ipts
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
# <-- Cache directory
# ================== Hard coded params ==================

sample_info_file = sys.argv[1]
with open(sample_info_file, "r") as f:
    sample_dict = json.load(f)


def mass_desnity_calc(mass, shape, radius, **kwargs):
    height = kwargs.get('height', None)
    if shape == "Cylinder":
        volume = np.pi * radius**2. * height
    elif shape == "Spherical":
        volume = 4. / 3. * np.pi * radius**3.
    else:
        return

    return mass / volume


def unit_converter_length(in_unit):
    if in_unit == "cm":
        return 1.
    elif in_unit == "mm":
        return 0.1
    elif in_unit == "m":
        return 100.
    elif in_unit == "µm":
        return 1E-4
    else:
        return


def unit_converter_weight(in_unit):
    if in_unit == "g":
        return 1.
    elif in_unit == "kg":
        return 1E3
    elif in_unit == "µg":
        return 1E-6
    elif in_unit == "mg":
        return 1E-3
    else:
        return


def main():
    sample_dict_final = dict()
    for item in sample_dict["items"]:
        sample_id = item["sample id"]
        sample_name = item["sample name"]
        sample_fl = item["molecular formula"]
        mass = item["sample mass"]
        mass_unit = item["mass units"]
        if not (mass is None or mass_unit is None):
            mass = float(mass) * unit_converter_weight(mass_unit)
        mass_density = None
        container_name = item["container name"].lower()
        if "quartz" in container_name and "03" in container_name:
            container_id = "QUARTZ03"
        elif "pac" in container_name and "03" in container_name:
            container_id = "PAC03"
        elif "pac" in container_name and "06" in container_name:
            container_id = "PAC06"
        elif "pac" in container_name and "08" in container_name:
            container_id = "PAC08"
        elif "pac" in container_name and "10" in container_name:
            container_id = "PAC10"
        else:
            container_id = None
        if not container_id is None:
            shape = container_dict[container_id]["Shape"]
            radius = container_dict[container_id]["Radius"]
        else:
            shape = None
            radius = None
        if shape == "Cylinder":
            height = item["sample height in container"]
            height_unit = item["height units"]
            if not (height is None or height_unit is None):
                height = float(height) * unit_converter_length(height_unit)
            else:
                height = None
            if not (mass is None or radius is None or height is None):
                mass_density = mass_desnity_calc(mass, shape, radius, height=height)
            else:
                mass_density = None
        else:
            height = None
            if not (mass is None or radius is None):
                mass_density = mass_desnity_calc(mass, shape, radius)
            else:
                mass_density = None
        env_name = env_name_def
        sample_dict_final[sample_id] = { "SampleName": sample_name,
                                         "ChemicalFormula": sample_fl,
                                         "MassDensity": mass_density,
                                         "ContainerID": container_id,
                                         "Shape": shape,
                                         "Radius": radius,
                                         "Height": height }
    with open(os.path.join(cache_dir, "abs_calc_sample_info.csv"), "w") as f:
        f.write("ITEM ID,Sample Name,Chemical Formula,Mass Density,")
        f.write("Packing Fraction,Container Type,Shape,Radius,Height,")
        f.write("Abs Method (SO or SC or FPP),MS Method (None or SO or SC)\n")
        for key, item in sample_dict_final.items():
            if item["SampleName"] is None:
                sn_tmp = "Null"
            else:
                sn_tmp = item["SampleName"]
            if item["ChemicalFormula"] is None:
                cf_tmp = "Null"
            else:
                cf_tmp = item["ChemicalFormula"]
            if item["MassDensity"] is None:
                md_tmp = "Null"
            else:
                md_tmp = "{0:7.5F}".format(item["MassDensity"])
            if item["ContainerID"] is None:
                cid_tmp = "Null"
            else:
                cid_tmp = item["ContainerID"]
            if item["Shape"] is None:
                sp_tmp = "Null"
            else:
                sp_tmp = item["Shape"]
            if item["Radius"] is None:
                rd_tmp = "Null"
            else:
                rd_tmp = "{0:6.4F}".format(item["Radius"])
            if item["Height"] is None:
                ht_tmp = "Null"
            else:
                ht_tmp = "{0:3.1F}".format(item["Height"])
            
            f.write("{0:d},{1:s},{2:s},{3:s},{4:s},{5:s},".format(key,
                                                                  sn_tmp,
                                                                  cf_tmp,
                                                                  md_tmp,
                                                                  "1.0",
                                                                  cid_tmp))
            f.write("{0:s},{1:s},{2:s},{3:s},{4:s}\n".format(sp_tmp,
                                                             rd_tmp,
                                                             ht_tmp,
                                                             "SO",
                                                             "None"))


if __name__ == "__main__":
    main()
