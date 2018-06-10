from pathlib import Path
import platform

def getDir(client, typ):

    filename = "data_dir.csv"
    sep_linux = "/"
    sep_windows = "\\"

    home = str(Path.home())
    os   = platform.system()

    if os == "Linux":
        filename = sep_linux+filename
    else:
        filename = sep_windows+filename

    f = open(home+filename, "r")
    f.readline()

    d = {}
    for x in f:
        x = x.rstrip()
        fields = x.split("|")
        key = (fields[0],fields[1],fields[2])
        d[key] = fields[3]

    key = (client,platform.node(),typ)
    if key in d:
        return d[key]
    else:
        return None