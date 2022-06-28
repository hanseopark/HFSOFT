from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plot
import pandas as pd
import btagging_keras, btagging_helpers
import numpy

plot.rc('axes', linewidth=3)

#########################################################
def FitModel(label, data, clas, sample_weight=None):
  y_train = (data['Jet_MC_MotherHadron'] == btagging_helpers.gTarget).values

  X_train = pd.DataFrame(data)
  X_train.drop('Jet_MC_MotherHadron', axis=1, inplace=True)
  X_train.drop('Event_PtHardBin', axis=1, inplace=True)

  print('Input parameters: {}'.format(X_train.columns))

  if sample_weight is None:
    clas.fit(X_train, y_train)
  else:
    clas.fit(X_train, y_train, sample_weight = sample_weight)

  # Save the model
  import joblib
  joblib.dump(clas, './Models/Model_{}.pkl'.format(label))

  print("... done")
  print("")


#########################################################
def Fit_MLP(data):
  print("")
  print("### Training neural network classifier...")
  clas = MLPClassifier(hidden_layer_sizes=(100,100,50), learning_rate='adaptive', verbose=0)
  FitModel('NeuralNetwork_Target{}'.format(btagging_helpers.gTarget), data, clas)

def Fit_RandomForest(data):
  print("")
  print("### Training random forest classifier...")
  clas = RandomForestClassifier(n_estimators=150, max_depth=8)
  FitModel('RandomForest_Target{}'.format(btagging_helpers.gTarget), data, clas)

def Fit_Ridge(data):
  print("")
  print("### Training ridge classifier...")
  clas = RidgeClassifier()
  FitModel('Ridge_Target{}'.format(btagging_helpers.gTarget), data, clas)

def Fit_Keras(data):
  print("")
  print("### Training Keras classifier...")
  FitKerasModel(data)


###############################################
def GenerateROCCurve(label, truth, score, suffix = ''):
  """ROC curve & AUC are generated"""
  from sklearn.metrics import roc_curve, roc_auc_score

  currentROCx, currentROCy, _ = roc_curve(truth, score)
  currentAUC = roc_auc_score(truth, score)
  print('AUC={:f}'.format(currentAUC))

  plot.plot(currentROCy, currentROCy, label='Guess ROC', linewidth=3.0)
  plot.plot(currentROCx, currentROCy, label='b-jets', linewidth=3.0)
  plot.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
  plot.legend(loc='lower right', prop={'size': 30})
  plot.tick_params(labelsize=30, length=10, width=3)
  plot.xlim(0, 1)
  plot.ylim(0, 1)
  plot.title('ROC curves', fontsize=35)
  plot.xlabel('False positive rate', fontsize=30)
  plot.ylabel('True positive rate', fontsize=30)
  plot.savefig('./{:s}-ROC{}.png'.format(label, suffix), dpi=480)
  plot.clf()


#########################################################
def GetScoreThreshold(scores, efficiency):
  import numpy
  if len(scores.shape) == 2:
    scores = numpy.sort(scores[:, 0])[::-1]
  else:
    scores = numpy.sort(scores)[::-1]
  for i, score in enumerate(scores):
    eff = float(i)/len(scores)
    if eff >= efficiency:
      return score

#########################################################
def GetThresholdEfficiency(scores, threshold):
  import numpy
  if len(scores.shape) == 2:
    scores = numpy.sort(scores[:, 0])[::-1]
  else:
    scores = numpy.sort(scores)[::-1]
  selected = 0
  for score in scores:
    if score >= threshold:
      selected += 1
  return float(selected)/len(scores)


#########################################################
def EvaluateModelPerformance(model, data_in):
  print('\n###### Evaluating {}'.format(model))

  ### Filter + split datasets
  data = pd.DataFrame(data_in.loc[(data_in['Jet_Pt'] >= 30) & (data_in['Jet_Pt'] < 40)])
  data_test_B  = pd.DataFrame(data.loc[(data['Jet_MC_MotherHadron'] == 5)])
  data_test_C  = pd.DataFrame(data.loc[(data['Jet_MC_MotherHadron'] == 4)])
  data_test_LF = pd.DataFrame(data.loc[(data['Jet_MC_MotherHadron'] != 4) & (data['Jet_MC_MotherHadron'] != 5) ])
  #smallestLen = min(min(len(data_test_B), len(data_test_C)), len(data_test_LF))
  #data_test_B = data_test_B[0:smallestLen] # Shorten to have same length
  #data_test_C = data_test_C[0:smallestLen] # Shorten to have same length
  #data_test_LF = data_test_LF[0:smallestLen] # Shorten to have same length

  ### Combined dataset (with equal class weights)
  data_combined = pd.concat([data_test_B, data_test_C, data_test_LF])
  truth_test_combined = (data_combined['Jet_MC_MotherHadron'] == btagging_helpers.gTarget).values
  data_test_combined = pd.DataFrame(data_combined)

  data_test_B.drop('Jet_MC_MotherHadron', axis=1, inplace=True)
  data_test_C.drop('Jet_MC_MotherHadron', axis=1, inplace=True)
  data_test_LF.drop('Jet_MC_MotherHadron', axis=1, inplace=True)
  data_test_B.drop('Event_PtHardBin', axis=1, inplace=True)
  data_test_C.drop('Event_PtHardBin', axis=1, inplace=True)
  data_test_LF.drop('Event_PtHardBin', axis=1, inplace=True)
  data_test_combined.drop('Jet_MC_MotherHadron', axis=1, inplace=True)
  data_test_combined.drop('Event_PtHardBin', axis=1, inplace=True)

  ### Extract scores for sklearn classifier
  if not 'keras' in model.lower():
    import joblib
    clas = joblib.load('./Models/Model_{}_Target{}.pkl'.format(model, btagging_helpers.gTarget))

    score_combined = clas.predict_proba(data_test_combined)[:,1]
    score_B  = clas.predict_proba(data_test_B)[:,1]
    score_C  = clas.predict_proba(data_test_C)[:,1]
    score_LF = clas.predict_proba(data_test_LF)[:,1]
  ### Extract scores for keras classifier
  else:
    X_test_combined  = GetDataForKeras(data_test_combined)
    X_test_B  = GetDataForKeras(data_test_B)
    X_test_C  = GetDataForKeras(data_test_C)
    X_test_LF = GetDataForKeras(data_test_LF)
    myModel = btagging_keras.AliMLKerasModel(2)
    myModel.LoadModel('Model_{}_Target{}'.format(model, btagging_helpers.gTarget))#_Default')
    score_combined  = myModel.fModel.predict(X_test_combined, batch_size=512, verbose=0)
    score_B  = myModel.fModel.predict(X_test_B, batch_size=512, verbose=0)
    score_C  = myModel.fModel.predict(X_test_C, batch_size=512, verbose=0)
    score_LF = myModel.fModel.predict(X_test_LF, batch_size=512, verbose=0)

  threshold = GetScoreThreshold(score_B, 0.2)
  print('\n##########################')
  print('\nThresold score is: {} \n'.format(threshold))
  print('b efficiency: {}'.format(GetThresholdEfficiency(score_B, threshold)))
  print('c efficiency: {}'.format(GetThresholdEfficiency(score_C, threshold)))
  print('udsg efficiency: {}'.format(GetThresholdEfficiency(score_LF, threshold)))
  print('##########################')

  # Evaluate model
  GenerateROCCurve(model, truth_test_combined, score_combined)
  SaveHistogram('./{:s}-Scores.png'.format(model), 'Scores on validation data', tuple([score_B, score_C, score_LF]), ('B','C', "light flavor"), rangex=(0,1), logY=True)


#########################################################
def SaveHistogram(fname, title, y, functionlabels, rangex=(0,1), legendloc='upper right', logY=False, nbins=100):
  import math
  """Simple histogram helper function"""
  # Check input data
  for i in range(len(y)):
    for item in y[i]:
      if math.isnan(item):
        print('Histogram {:s} was not saved due to invalid values.'.format(fname))
        return

  for i in range(len(y)):
    plot.hist(y[i], label=functionlabels[i], bins=nbins, histtype='step', density=True, linewidth=3.0)
  plot.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
  plot.tick_params(labelsize=30, length=10, width=3)
  plot.legend(loc=legendloc, prop={'size': 30})
  plot.title(title, fontsize=35, pad=10)
  plot.xlabel('Scores', fontsize=30)
  plot.xlim(rangex[0], rangex[1])
#  plot.yscale('log', nonposy='clip')

  if logY:
    plot.yscale('log', nonposy='clip')

  plot.savefig("{:s}".format(fname), dpi=480)
  #plot.show()
  plot.clf()



#########################################################
def FitKerasModel(data, val_data, verbose=True):

  ### Prepare data
  X_train = GetDataForKeras(data)
  X_test = GetDataForKeras(val_data)
  y_train = (data['Jet_MC_MotherHadron'] == btagging_helpers.gTarget).values
  y_test = (val_data['Jet_MC_MotherHadron'] == btagging_helpers.gTarget).values

  ### Prepare model
  myModel = btagging_keras.AliMLKerasModel(2)
  myModel.AddBranchCNN1D([128,64,32], [2,0,0], 1, [4,2,2], 0.2, inputShape=(10, 3))
  myModel.AddBranchCNN1D([16,32,64,96,64], [2,0,0,0,0], 1, [2,2,2,2,2], 0.2, inputShape=(10, 2))
  myModel.SetFinalLayer(4, 128,0.2, 'ridge')
  myModel.fInit = 'he_uniform'
  myModel.fOptimizer = 'adam'
  myModel.fLossFunction = 'binary_crossentropy'
  myModel.fBatchSize = 1000
  myModel.CreateModel('Model_Keras_Default_Target{}'.format(btagging_helpers.gTarget))
  myModel.fLearningRate = 0.01
  myModel.PrintProperties()

  ### Train model
  myModel.TrainModel(X_train, y_train, X_test, y_test, numEpochs = 300)
  myModel.SaveModel()


#########################################################
def GetDataForKeras(data):
  arr_Lxy        = pd.concat([data['Jet_SecVtx_Lxy_{}'.format(i)] for i in range(10)], axis=1).values
  arr_SigmaLxy   = pd.concat([data['Jet_SecVtx_SigmaLxy_{}'.format(i)] for i in range(10)], axis=1).values
  arr_Dispersion = pd.concat([data['Jet_SecVtx_Dispersion_{}'.format(i)] for i in range(10)], axis=1).values
  arr_branch1 = numpy.array([arr_Lxy, arr_SigmaLxy, arr_Dispersion])
  arr_branch1 = numpy.swapaxes(arr_branch1, 0, 1)
  arr_branch1 = numpy.swapaxes(arr_branch1, 1, 2)

  arr_IPd        = pd.concat([data['Jet_Track_IPd_{}'.format(i)] for i in range(10)], axis=1).values
  arr_CovIPd     = pd.concat([data['Jet_Track_CovIPd_{}'.format(i)] for i in range(10)], axis=1).values
  arr_branch2 = numpy.array([arr_IPd, arr_CovIPd])
  arr_branch2 = numpy.swapaxes(arr_branch2, 0, 1)
  arr_branch2 = numpy.swapaxes(arr_branch2, 1, 2)

  return [arr_branch1,arr_branch2]
