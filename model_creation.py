from ast import Import
from flask import Flask, render_template, request
from keras.models import save_model
import os
import numpy as np
from keras.models import Model, load_model
import matplotlib.pyplot as mp
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as om
import torchvision as tv
from MedNetClass import MedNet

if os.path.exists('saved_model.pt'):
    model: MedNet = torch.jit.load("saved_model.pt")
    model.eval()

else:

    if torch.cuda.is_available():  # Make sure GPU is available
        dev = torch.device("cuda:0")
        kwar = {'num_workers': 8, 'pin_memory': True}
        cpu = torch.device("cpu")
    else:
        print("Warning: CUDA not found, CPU only.")
        dev = torch.device("cpu")
        kwar = {}
        cpu = torch.device("cpu")

    np.random.seed(551)
    dataDir = 'resized'  # The main data directory
    classNames = os.listdir(dataDir)  # Each type of image can be found in its own subdirectory
    numClass = len(classNames)  # Number of types = number of subdirectories
    imageFiles = [[os.path.join(dataDir, classNames[i], x) for x in os.listdir(os.path.join(dataDir, classNames[i]))]
                  for i in range(numClass)]  # A nested list of filenames
    numEach = [len(imageFiles[i]) for i in range(numClass)]  # A count of each type of image
    imageFilesList = []  # Created an un-nested list of filenames
    imageClass = []  # The labels -- the type of each individual image in the list
    for i in range(numClass):
        imageFilesList.extend(imageFiles[i])
        imageClass.extend([i] * numEach[i])
    numTotal = len(imageClass)  # Total number of images
    imageWidth, imageHeight = Image.open(imageFilesList[0]).size  # The dimensions of each image

    print("There are", numTotal, "images in", numClass, "distinct categories")
    print("Label names:", classNames)
    print("Label counts:", numEach)
    print("Image dimensions:", imageWidth, "x", imageHeight)

    mp.subplots(3, 3, figsize=(8, 8))

    for i, k in enumerate(np.random.randint(numTotal, size=9)):  # Take a random sample of 9 images and
        im = Image.open(imageFilesList[k])  # plot and label them
        arr = np.array(im)
        mp.subplot(3, 3, i + 1)
        mp.xlabel(classNames[imageClass[k]])
        mp.imshow(arr, cmap='gray', vmin=0, vmax=255)
    mp.tight_layout()
    mp.show()

    toTensor = tv.transforms.ToTensor()


    def scaleImage(x):  # Pass a PIL image, return a tensor
        y = toTensor(x)
        if (y.min() < y.max()):  # Assuming the image isn't empty, rescale so its values run from 0 to 1
            y = (y - y.min()) / (y.max() - y.min())
        z = y - y.mean()  # Subtract the mean value of the image
        return z


    imageTensor = torch.stack(
        [scaleImage(Image.open(x)) for x in imageFilesList])  # Load, scale, and stack image (X) tensor
    print("test")
    classTensor = torch.tensor(imageClass)  # Create label (Y) tensor
    print("Rescaled min pixel value = {:1.3}; Max = {:1.3}; Mean = {:1.3}"
          .format(imageTensor.min().item(), imageTensor.max().item(), imageTensor.mean().item()))

    validFrac = 0.1  # Define the fraction of images to move to validation dataset
    testFrac = 0.1  # Define the fraction of images to move to test dataset
    validList = []
    testList = []
    trainList = []

    for i in range(numTotal):
        rann = np.random.random()  # Randomly reassign images
        if rann < validFrac:
            validList.append(i)
        elif rann < testFrac + validFrac:
            testList.append(i)
        else:
            trainList.append(i)

    nTrain = len(trainList)  # Count the number in each set
    nValid = len(validList)
    nTest = len(testList)
    print("Training images =", nTrain, "Validation =", nValid, "Testing =", nTest)
    trainIds = torch.tensor(trainList)  # Slice the big image and label tensors up into
    validIds = torch.tensor(validList)  # training, validation, and testing tensors
    testIds = torch.tensor(testList)
    trainX = imageTensor[trainIds, :, :, :]
    trainY = classTensor[trainIds]
    validX = imageTensor[validIds, :, :, :]
    validY = classTensor[validIds]
    testX = imageTensor[testIds, :, :, :]
    testY = classTensor[testIds]

    model = MedNet(imageWidth, imageHeight, numClass).to(dev)

    learnRate = 0.07  # Define a learning rate.
    maxEpochs = 70  # Maximum training epochs
    t2vRatio = 3  # Maximum allowed ratio of validation to training loss
    t2vEpochs = 20  # Number of consecutive epochs before halting if validation loss exceeds above limit
    batchSize = 1450  # Batch size. Going too large will cause an out-of-memory error.
    trainBats = nTrain // batchSize  # Number of training batches per epoch. Round down to simplify last batch
    validBats = nValid // batchSize  # Validation batches. Round down
    testBats = -(-nTest // batchSize)  # Testing batches. Round up to include all
    CEweights = torch.zeros(numClass)  # This takes into account the imbalanced dataset.
    for i in trainY.tolist():  # By making rarer images count more to the loss,
        CEweights[i].add_(1)  # we prevent the model from ignoring them.
    CEweights = 1. / CEweights.clamp_(min=1.)  # Weights should be inversely related to count
    CEweights = (CEweights * numClass / CEweights.sum()).to(dev)  # The weights average to 1
    opti = om.SGD(model.parameters(), lr=learnRate)  # Initialize an optimizer

    for i in range(maxEpochs):
        model.train()  # Set model to training mode
        epochLoss = 0.
        permute = torch.randperm(nTrain)  # Shuffle data to randomize batches
        trainX = trainX[permute, :, :, :]
        trainY = trainY[permute]
        for j in range(trainBats):  # Iterate over batches
            opti.zero_grad()  # Zero out gradient accumulated in optimizer
            batX = trainX[j * batchSize:(j + 1) * batchSize, :, :, :].to(dev)  # Slice shuffled data into batches
            batY = trainY[j * batchSize:(j + 1) * batchSize].to(dev)  # .to(dev) moves these batches to the GPU
            yOut = model(batX)  # Evalute predictions
            loss = F.cross_entropy(yOut, batY, weight=CEweights)  # Compute loss
            epochLoss += loss.item()  # Add loss
            loss.backward()  # Backpropagate loss
            opti.step()  # Update model weights using optimizer
        validLoss = 0.
        permute = torch.randperm(nValid)  # We go through the exact same steps, without backprop / optimization
        validX = validX[permute, :, :, :]  # in order to evaluate the validation loss
        validY = validY[permute]
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Temporarily turn off gradient descent
            for j in range(validBats):
                opti.zero_grad()
                batX = validX[j * batchSize:(j + 1) * batchSize, :, :, :].to(dev)
                batY = validY[j * batchSize:(j + 1) * batchSize].to(dev)
                yOut = model(batX)
                validLoss += F.cross_entropy(yOut, batY, weight=CEweights).item()
        epochLoss /= trainBats  # Average loss over batches and print
        validLoss /= validBats
        print("Epoch = {:-3}; Training loss = {:.4f}; Validation loss = {:.4f}".format(i, epochLoss, validLoss))
        if validLoss > t2vRatio * epochLoss:
            t2vEpochs -= 1  # Test if validation loss exceeds halting threshold
            if t2vEpochs < 1:
                print("Validation loss too high; halting to prevent overfitting")
                break

    confuseMtx = np.zeros((numClass, numClass), dtype=int)  # Create empty confusion matrix
    model.eval()
    with torch.no_grad():
        permute = torch.randperm(nTest)  # Shuffle test data
        testX = testX[permute, :, :, :]
        testY = testY[permute]
        for j in range(testBats):  # Iterate over test batches
            batX = testX[j * batchSize:(j + 1) * batchSize, :, :, :].to(dev)
            batY = testY[j * batchSize:(j + 1) * batchSize].to(dev)
            yOut = model(batX)  # Pass test batch through model
            pred = yOut.max(1, keepdim=True)[1]  # Generate predictions by finding the max Y values
            for j in torch.cat((batY.view_as(pred), pred), dim=1).tolist():  # Glue together Actual and Predicted to
                confuseMtx[j[0], j[1]] += 1  # make (row, col) pairs, and increment confusion matrix
    correct = sum(
        [confuseMtx[i, i] for i in range(numClass)])  # Sum over diagonal elements to count correct predictions
    print("Correct predictions: ", correct, "of", nTest)
    print("Confusion Matrix:")
    print(confuseMtx)
    print(classNames)

    print(correct / nTest * 100)


    def scaleBack(x):  # Pass a tensor, return a numpy array from 0 to 1
        if (x.min() < x.max()):  # Assuming the image isn't empty, rescale so its values run from 0 to 1
            x = (x - x.min()) / (x.max() - x.min())
        return x[0].to(cpu).numpy()  # Remove channel (grayscale anyway)


    model.eval()
    mp.subplots(3, 3, figsize=(8, 8))
    imagesLeft = 9
    permute = torch.randperm(nTest)  # Shuffle test data
    testX = testX[permute, :, :, :]
    testY = testY[permute]
    for j in range(testBats):  # Iterate over test batches
        batX = testX[j * batchSize:(j + 1) * batchSize, :, :, :].to(dev)
        batY = testY[j * batchSize:(j + 1) * batchSize].to(dev)
        yOut = model(batX)  # Pass test batch through model
        pred = yOut.max(1)[1].tolist()  # Generate predictions by finding the max Y values
        for i, y in enumerate(batY.tolist()):
            if imagesLeft and y != pred[i]:  # Compare the actual y value to the prediction
                imagesLeft -= 1
                mp.subplot(3, 3, 9 - imagesLeft)
                mp.xlabel(classNames[pred[i]])  # Label image with what the model thinks it is
                mp.imshow(scaleBack(batX[i]), cmap='gray', vmin=0, vmax=1)
    mp.tight_layout()
    mp.show()

    # Enregistrer le mod�le sur le disque
    scripted_model = torch.jit.script(model)
    scripted_model.save("saved_model.pt")


# app = Flask(__name__)

## D�finir les classes de sortie
# classes = {0: 'class1', 1: 'class2', 2: 'class3', 3: 'class4', 4: 'class5', 5: 'class6'}

## D�finir la page d'accueil de l'application
# @app.route('/', methods=['GET', 'POST'])
# def home():
#    if request.method == 'POST':
#        # R�cup�rer l'image t�l�charg�e par l'utilisateur
#        file = request.files['file']
#        img = Image.open(file)
#        img = img.resize((224, 224))
#        img = np.array(img) / 255.0

#        # Faire une pr�diction de classe avec le mod�le de classification
#        pred = model.predict(np.array([img]))
#        class_idx = np.argmax(pred)

#        # Afficher la classe pr�dite sur la page r�sultante
#        return render_template('result.html', class_name=classes[class_idx])
#    else:
#        return render_template('home.html')


