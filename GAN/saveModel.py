def saveModel(config, model, typ, epoch):
    loc = config["modelDir"]
    filename = typ + "_" + str(epoch) + ".h5"
    model.save(loc+filename)