def getTrainData(train_name,label_name):
    root = "C:\DataSet\Code\FaceRecognition\Train Face Crop RGB"
    data_path = os.path.join(root,train_name)
    label_path = os.path.join(root,label_name)
    traindata = np.load(data_path)                      # @mobeen Load numpy array 
    trainlabel = np.load(label_path)
    
    idx = np.random.permutation(len(traindata))         # generate random index array
    x,y = traindata[idx], trainlabel[idx]               # use randon index array to shuffle labels and images (apply to both vis and the images)

    # showData(x, y)

    labels = to_categorical(y)
    # print(labels[0])
    

    

    return x, labels