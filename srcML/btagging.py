#!/usr/bin/env python
from sklearn.model_selection import train_test_split
import btagging_helpers, btagging_models
import pandas as pd

#### Settings
recreateData            = False
checkData               = False
trainModels             = False
evalModel               = False
predictData             = True
testSize                = 0.2
trainSize               = 0.8
btagging_helpers.gTarget = 5 # 5 = bjets, 4 =cjets

data_LHC18b8 = pd.DataFrame()
####### Create filtered dataset, create features etc.
if trainModels or evalModel:
    data_LHC18b8 = btagging_helpers.LoadInputData_LHC18b8(recreate=recreateData)
    print("Finished loading MC")

#  data_LHC18b8 = btagging_helpers.LoadInputData_LHC18b8(recreate=recreateData)

if checkData:
    print("CheckData requested")
    btagging_helpers.CheckData(data_LHC18b8, 'allJets')
    print(data_LHC18b8.corr()['Jet_MC_MotherHadron'].sort_values(ascending=False))

####### Split dataset into training and validation
if trainModels or evalModel:
    toy_train, toy_test = train_test_split(data_LHC18b8, test_size=testSize, train_size =trainSize, random_state=42)
    print('### Dataset split done. Training samples: {}, testing samples: {}'.format(len(toy_train), len(toy_test)))

####### Train & evaluate the models
if trainModels:
    btagging_models.Fit_RandomForest(toy_train)
    btagging_models.FitKerasModel(toy_train, toy_test)


if evalModel:
    btagging_models.EvaluateModelPerformance('Keras_Default', toy_test)
    btagging_models.EvaluateModelPerformance('RandomForest', toy_test)

####### Export scores
#if predictData:
#  models = ['Keras_Default', 'RandomForest']
#  for i in range(1,9+1):
#    btagging_helpers.AddScoresToFiles(models, i)
#
#  for i in range(1,20+1):
#    btagging_helpers.AddScoresToFiles_MC(models, i)

if predictData:
    models = ['Keras_Default', 'RandomForest']
    btagging_helpers.AddScoresToFiles_MC(models, 1)
