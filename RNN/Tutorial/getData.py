def getData(config):
    loc = config["getData"]
    file = config["fileName"]
    with open(loc+file, 'r') as f:
        text = f.read()
    return text