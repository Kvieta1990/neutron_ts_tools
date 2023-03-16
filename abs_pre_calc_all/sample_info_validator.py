import sys

si_file = sys.argv[1]

file_in = open(si_file, "r")
lines = file_in.readlines()
file_in.close()

column = {
    0: "ITEM ID",
    1: "Sample Name",
    2: "Chemical Formula",
    3: "Mass Density",
    4: "Packing Fraction",
    5: "Container Type",
    6: "Shape",
    7: "Radius",
    8: "Height",
    9: "Abs Method (SO or SC or FPP)",
    10: "MS Method (None or SO or SC)"
}
valid_cid = ["PAC03", "PAC06", "PAC08", "PAC10", "QuartzTube03"]
valid_shape = ["Cylinder", "Sphere", "Hollow Cylinder"]
missing_entry = dict()
for line in lines[1:]:
    if len(line) > 0:
        sample_id = line.split(",")[0]
        missing_entry[sample_id] = []
        for i, item in enumerate(line.split(",")):
            if item == "Null":
                missing_entry[sample_id].append(column[i]) 
            elif i in [3, 4, 7, 8]:
                try:
                    _ = float(item.strip())
                except Exception as e:
                    print(e)
                    missing_entry[sample_id].append(column[i]) 
            elif i == 5:
                if not item in valid_cid:
                    missing_entry[sample_id].append(column[i])
            elif i == 6:
                if not item in valid_shape:
                    missing_entry[sample_id].append(column[i])
            elif i == 9:
                if not item in ["SO", "SC", "FPP"]:
                    missing_entry[sample_id].append(column[i])
            elif i == 10:
                if not item.strip() in ["None", "SO", "SC"]:
                    missing_entry[sample_id].append(column[i])

incomplete_num = 0
for key,item in missing_entry.items():
    if len(item) > 0:
        incomplete_num += 1

if incomplete_num > 0:
    print("\n==============Missing entries==============")
    for key, item in missing_entry.items():
        if len(item) > 0:
            print(f"{key}", item)
    print("==============Missing entries==============")
    sys.exit(1)
