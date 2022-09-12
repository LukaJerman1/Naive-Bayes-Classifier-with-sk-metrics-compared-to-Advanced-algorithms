
import numpy as np
from random import randrange
import csv
import math
from sklearn import metrics

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pb
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve,auc
from sklearn.svm import SVC



def mean(number):
    # Vrne povprečje številk

    return np.mean(number)


def stdev(number):
    # Vrne standardni odklon oz. deviacijo

    return np.std(number)



def load_csv(filename): #naloži csv file
    
    line=csv.reader(open(filename, "r"))

    next(line) # preskoži prvo vrstico z imeni
    dataset=list(line)

    for i in range(len(dataset)):

        dataset[i] = [float(x) for x in dataset[i]]  # V primeru, da so številke tipa String jih spremeni v float

    return dataset


def cross_validation_split(dataset, k_folds):
    # Podatke razdeli na k-folde, npr. 5 k_fold. Ustvari list() razdeljenih podatkov in jih vrne. Navzkrižna validacija.
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / k_folds) # Določi fold size - deli vse dolžino vseh podatkov z število foldov
    for i in range(k_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy)) # Naključen podatek
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)

    # print(dataset_split)
    return dataset_split # Vrne [[[35.0, 72000.0, 0.0], [49.0, 28000.0, 1.0], ...]] za vsak fold je en list() podatkov


def accuracy_metric(actual,predicted): # Metrika točnosti
    
    correct = 0

    for i in range(len(actual)):

        if actual[i] == predicted[i]: # Primerjava dejanske vrednosti in predvidene
            correct += 1 # če sta enaki se števec pravilnih poveča 

    return correct / float(len(actual)) * 100.0 # izračun procenta pravilnosti



def get_others (dataset,algorithm,n_folds, ):
    # Vzame podatke(.csv) , algoritem in število foldov in izračuna željene metrike
    folds = cross_validation_split(dataset, n_folds)
    # -------- Vse željene metrike
    recall = list()
    f1 = list()
    precision = list()
    confusion=list()
    roc_auc=list()
    roc_curve_fpr = list()
    roc_curve_tpr = list()
    scores = list()
    # ---------------
    for fold in folds:
        train_set = list(folds)
        # Vse folde shrani v train_set
        train_set.remove(fold)
        # Izbriše trenutnega
        train_set = sum(train_set, [])
        # Sešteje vse vrednosti v listu z začetkom pri []
        test_set = list()
        # test_set je prazen list()
        for row in fold:
            row_copy = list(row)
            # Kopija vrste
            test_set.append(row_copy)
            # Vrsto doda v test_set
            row_copy[-1] = None
            # Na indexu -1 == None
        predicted = algorithm(train_set, test_set, ) # Algoritem vrne BAYES
        actual = [row[-1] for row in fold] # izpis dejanskih vrednosti podatkov

        recall_metric = metrics.recall_score(actual,predicted) 
        recall.append(recall_metric) # Izračun recall

        f1_metric = metrics.f1_score(actual,predicted)
        f1.append(f1_metric) # izračun F1

        precision_metric = metrics.precision_score(actual,predicted)
        precision.append(precision_metric) # izračun precision

        confusion_matric = metrics.confusion_matrix(actual,predicted)
        confusion.append(confusion_matric) # izračun confusion matrix

        roc_auc_metric = metrics.roc_auc_score(actual,predicted)
        roc_auc.append(roc_auc_metric) # izračun roc auc score

        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy) # izračun natančnosti

        """
        nb_fpr,nb_tpr,= metrics.roc_curve(actual,predicted)
        roc_curve_fpr.append(nb_fpr)
        roc_curve_tpr.append(nb_tpr)
        """

    return recall,f1,precision,confusion,roc_auc,scores,actual,predicted # Vrne vse metrike

# Implementacija NB ---> Algoritem omogoči način kalkulacije možnosti(dogodka) glede na del podatka, 
# ki pripada k določenemu razredu na osnovi predhodnega znanja

# Formula : P(class|data) = (P(data|class) * P(class)) / P(data)

def separate_by_class(dataset):
    
    # Razdruži učen set po vrednosti razreda
    separated = {}

    for i in range(len(dataset)):
        row = dataset[i]

        if row[-1] not in separated: # če podatka ni v vrsti
            separated[row[-1]] = []
        separated[row[-1]].append(row)

    # print("separate_by_class: ")
    # print(separated)
    # vrne razdeljene podatke:
    # vse podatke, ki se končajo z vrednostjo 1.0 - torej True
    # vse podatke, ki se končajo z vrednostjo 0.0 - torej False
    # Npr. {0.0: [[25.0, 22000.0, 0.0], [40.0, 75000.0, 0.0], ...}

    return separated # vrne vse podatke po razredih


def model(dataset):
    # Za vsak atribut najde povprečje in standardni odklon
    models = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    # Zip vzame dataset, ki je list listov in iterira skozi vsak element v vsaki vrsti in vrne stolpec kot list številk
    models.pop() 
    # Odstrani zadnjo, ker je vrednost razreda

    # print("Models: ")
    # print(models)
    # Npr:  [(46.24390243902439, 8.573300835150606), (87024.39024390244, 41669.13099989888)]

    # Vrnil bo glede na k-fold toliko 1.0 kot 0.0 modelov
    return models


def model_by_class(dataset):
    
    # Uporabi funkcijo separate_by_class, da razdruži podatke po razredih
    # Uporabi funkcijo model, da pridobi povprečje in SD vsakega atributa
    separated = separate_by_class(dataset)
    class_models = {}
    
    for (classValue, instances) in separated.items():
        class_models[classValue] = model(instances)

    # Vrne statistiko za vsako vrstico, torej za 1.0 ali 0.0
    # print("class models: ")
    # print(class_models)

    # Npr. {0.0: [(32.9377990430622, 8.11112750572704), (60406.6985645933, 23875.116735806158)], 1.0: [(46.126126126126124, 8.385011420662034), (88864.86486486487, 42317.22467578678)]}

    return class_models


def calculate_pdf(x,mean,stdev):
    # Probability Density Function
    # Izračun možnosti z gaussovo funkcijo
    # x je nek element
    if stdev == 0.0:
        if x == mean:
            return 1.0
        else:
            return 0.0
    # Formula za PDF
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    # Gausova distribucija
    res = 1/(math.sqrt(2*math.pi)*stdev)*exponent
    # print("Res: ")
    # print(res)
    # Npr. Res: 0.01877764154881396

    # Za vsako vrednost izračuna možnost
    return res 


def calculate_class_probabilities(models, input):
    # Calculate the class probability for input sample. Combine probability of each feature
    # Izračuna možnost glede na razred. Združi možnost vsakega atributa.
    probabilities = {}
    for (classValue, classModels) in models.items():
        probabilities[classValue] = 1
        for i in range(len(classModels)):
            (mean, stdev) = classModels[i]
            x = input[i]
            # Združuje možnosti glede na razred
            probabilities[classValue] *= calculate_pdf(x, mean, stdev)

    # print("Probabilities: ")
    # print(probabilities)
    # Npr. {1.0: 5.979676179745182e-08, 0.0: 5.99115868627663e-07}

    # Dictionary v katerem je shranjen
    return probabilities


def predict(models, inputVector):
    # Compare probability for each class. Return the class label which has max probability.
    # Primerja možnost za vsak razred. Vrne razred z najvišjo možnostjo
    probabilities = calculate_class_probabilities(models, inputVector)
    (topLabel, bestProb) = (None, -1)
    for (classValue, proba) in probabilities.items():
        if topLabel is None or proba > bestProb:
            bestProb = proba
            topLabel = classValue

    # print("Top label: ")
    # print(topLabel)
    # Npr. Top label: 1.0

    return topLabel


def getPredictions(models, testSet):
    # """Get class label for each value in test set."""
    # Label razreda za vsako vrednost v setu
    predictions = []
    for i in range(len(testSet)):
        result = predict(models, testSet[i])
        predictions.append(result)

    # print("Predictions: ")
    # print(predictions)
    # Npr. Predictions: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,...]

    # Za vsak fold naredi list, ki vsebuje napoved za element
    return predictions



def bayes_classifier(train, test, ):
    
    # 
    summaries = model_by_class(train) 
    # Natrenira model
    predictions = getPredictions(summaries, test)

    # print("NB: ")
    # print(predictions)
    # Izpis prejšne funkcije
    # Npr. Predictions: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,...]

    return predictions

def Average(lst):
    # Izračuna povprečje v list()

    return sum(lst) / len(lst)

def main():
    # load and prepare data
    filename = 'Used_Data.csv'
    dataset = load_csv(filename)

    n_folds = 5
    
    others=get_others(dataset, bayes_classifier, n_folds)
    
    for i in range(len(others)):
        if(i==0):
            print("-------------------------------------------")
            print("Recall for each fold:")
            print(others[i])
            print("Average Recall:")
            print(Average(others[i]))
            print("-------------------------------------------")
        elif(i==1):
            print("-------------------------------------------")
            print("F1 for each fold:")
            print(others[i])
            print("Average F1:")
            print(Average(others[i]))
            print("-------------------------------------------")
        elif(i==2):
            print("-------------------------------------------")
            print("Precision for each fold:")
            print(others[i])
            print("Average Precision:")
            print(Average(others[i]))
            print("-------------------------------------------")
        elif(i==3):
            print("-------------------------------------------")
            print("Confusion for each fold:")
            print(others[i])
            print("Average Confusion:")
            print(Average(others[i]))
            print("-------------------------------------------")
        elif(i==4):
            print("-------------------------------------------")
            print("ROC AUC for each fold:")
            print(others[i])
            print("Average ROC AUC:")
            print(Average(others[i]))
            print("-------------------------------------------")
        elif(i==5):
            print("-------------------------------------------")
            print("Accuracy for each fold:")
            print(others[i])
            print("Average Accuracy:")
            print(Average(others[i]))
            print("-------------------------------------------")
    
    

    data=pb.read_csv("Used_Data.csv")
    data.head(10)

    X = data.iloc[:,0:-1].values
    y = data.iloc[:,-1].values

    X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=4)
    
    nb=GaussianNB()
    nb.fit(X_train, y_train)
    
    nb_probs=nb.predict_proba(X_test)
    nb_probs=nb_probs[:, 1]

    nb_auc = roc_auc_score(y_test, nb_probs)
    nb_fpr,nb_tpr,__ =roc_curve(y_test, nb_probs)

    plt.title("AUC ROC Curve")
    plt.plot(nb_fpr, nb_tpr, marker=".", label= "Naive Bayes (AUROC = %0.3f)" % nb_auc)
    plt.legend(loc="lower right")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    

main()
