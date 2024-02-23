import os
import numpy as np
import pandas as pd

"""
Assuming the dataset folder is in the relative path ../datasets/processedDataset
Output: Four arrays containing the data location and its label
"""
def getFiles(path = "../datasets/processedDataset"):
    #get all the testing data
    testSet = []
    testLabel = []
    for dir_path in walk_directories(path + "/test"):
        currLabel = os.path.basename(dir_path)
        for _, _, files in os.walk(dir_path):
            for file in files:
                testSet.append(dir_path + "/" + file)
                testLabel.append(currLabel)

    #get all the training data
    trainSet = []
    trainLabel = []
    for dir_path in walk_directories(path + "/train"):
        currLabel = os.path.basename(dir_path)
        for _, _, files in os.walk(dir_path):
            for file in files:
                trainSet.append(dir_path + "/" + file)
                trainLabel.append(currLabel)

    return trainSet, trainLabel, testSet, testLabel


"""
Walk the system directories looking for images
Output: Two arrays containing the data location and its label
"""
def walk_directories(directory, image_extensions=('jpg', 'jpeg', 'png', 'gif')):
    # Walk through the directory tree
    for root, directories, files in os.walk(directory):
        # Check if any image files are present in the current directory
        if any(file.lower().endswith(image_extensions) for file in files):
            # Yield the current directory path
            yield root

"""
Converts the data into csv file
Output: Two CSV files containg all the information
"""
def convertDataframe(train, trainlabel, test, testlabel,path = "../datasets/"):
    #convert into dataframe
    train_df = pd.DataFrame({
            'trainSet': train,
            'trainLabel': trainlabel,
        })
    test_df = pd.DataFrame({
            'testSet': test,
            'testabel': testlabel,
        })
    train_df.to_csv(path + 'train_data.csv', index=False)
    test_df.to_csv(path + 'test_data.csv', index=False)



a,b,c,d = getFiles()
convertDataframe(a,b,c,d)
