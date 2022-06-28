from __future__ import print_function
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd
import root_pandas
import os.path, math
import matplotlib.pyplot as plt
import ROOT
from functools import partial

kNumMaxConstituents = 50
kNumMaxSecVertices  = 50
gTarget = 5

#########################################################
def ReadRootFile_LHC18b8(jetType, ptHardBin, truncate=True):
    import progressbar, math, time
    columns = ['Jet_Pt', 'Jet_Area', 'Jet_NumTracks', 'Jet_Track_IPd', 'Jet_Track_IPz', 'Jet_Track_CovIPd', 'Jet_Track_CovIPz', 'Jet_SecVtx_Lxy', 'Jet_NumSecVertices', 'Jet_SecVtx_SigmaLxy', 'Jet_SecVtx_Chi2', 'Jet_SecVtx_Dispersion', 'Jet_MC_MotherHadron']

    print('### Starting extraction for pT-hard bin {}.'.format(ptHardBin))

    fname = '/Users/hadi/ML_Rudiger/LHC18b8_pThardBin_{}.root'.format(ptHardBin)

    data_raw = root_pandas.read_root(fname, 'JetTree_AliAnalysisTaskJetExtractor_Jet_AKTChargedR040_tracks_pT0150_E_scheme_{}'.format(jetType), columns=columns)
    if truncate:
        data_raw = data_raw[data_raw['Jet_Pt']> 5]
    data_raw = data_raw[:50]


    # Preprocess data
    data = pd.DataFrame(data=data_raw['Jet_MC_MotherHadron'])
    data['Jet_Pt']              = data_raw['Jet_Pt']
    data['Jet_Area']            = data_raw['Jet_Area']
    data['Jet_NumTracks']       = data_raw['Jet_NumTracks']
    data['Jet_NumSecVertices']  = data_raw['Jet_NumSecVertices']
    data['Jet_Area']            = data_raw['Jet_Area']
    data['Event_PtHardBin']     = pd.Series()

    for i in range(kNumMaxConstituents):
        data['Jet_Track_IPd_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))
    data['Jet_Track_IPz_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))
    data['Jet_Track_CovIPd_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))
    data['Jet_Track_CovIPz_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))

    for i in range(kNumMaxSecVertices):
        data['Jet_SecVtx_Lxy_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))
    data['Jet_SecVtx_SigmaLxy_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))
    data['Jet_SecVtx_Chi2_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))
    data['Jet_SecVtx_Dispersion_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))

    jetID = 0
    #####  Loop over all jets
    for constIPd, constIPz, constCovIPd, constCovIPz, secVtxLxy, secVtxSigmaLxy, secVtxChi2, secVtxDispersion in zip(data_raw['Jet_Track_IPd'], data_raw['Jet_Track_IPz'],
                                                                                                                     data_raw['Jet_Track_CovIPd'], data_raw['Jet_Track_CovIPz'], data_raw['Jet_SecVtx_Lxy'], data_raw['Jet_SecVtx_SigmaLxy'], data_raw['Jet_SecVtx_Chi2'], data_raw['Jet_SecVtx_Dispersion']):

        #startTime = time.time()
    data['Event_PtHardBin'] = ptHardBin

    # Sort tracks for descending sig. impact parameters d
    sigIPd = constIPd/np.sqrt(constCovIPd)
    indices = sigIPd.argsort()[::-1]
    indices = indices[:kNumMaxConstituents]
    constIPd = constIPd[indices]
    constIPz = constIPz[indices]
    constCovIPd = constCovIPd[indices]
    constCovIPz = constCovIPz[indices]

    # Loop over constituents
    for i in range(len(indices)):
        data['Jet_Track_IPd_{}'.format(i)].iat[jetID] = constIPd[i]
        data['Jet_Track_IPz_{}'.format(i)].iat[jetID] = constIPz[i]
        data['Jet_Track_CovIPd_{}'.format(i)].iat[jetID] = constCovIPd[i]
        data['Jet_Track_CovIPz_{}'.format(i)].iat[jetID] = constCovIPz[i]

    # Sort sec. vertices for descending sig. Lxy
    sigLxy = secVtxLxy/(secVtxSigmaLxy)
    indices = sigLxy.argsort()[::-1]
    indices = indices[:kNumMaxSecVertices]
    secVtxLxy = secVtxLxy[indices]
    secVtxSigmaLxy = secVtxSigmaLxy[indices]
    secVtxChi2 = secVtxChi2[indices]
    secVtxDispersion = secVtxDispersion[indices]

    # Loop over sec. vertices
    for i in range(len(indices)):
        data['Jet_SecVtx_Lxy_{}'.format(i)].iat[jetID] = secVtxLxy[i]
        data['Jet_SecVtx_SigmaLxy_{}'.format(i)].iat[jetID] = secVtxSigmaLxy[i]
        data['Jet_SecVtx_Chi2_{}'.format(i)].iat[jetID] = secVtxChi2[i]
        data['Jet_SecVtx_Dispersion_{}'.format(i)].iat[jetID] = secVtxDispersion[i]

    #print('Loop time: {}'.format(time.time()-startTime))
    jetID += 1

    print('... pT-hard bin {} -- {} samples.'.format(ptHardBin, len(data)))
    data = data.fillna(0.0)
    return data


#########################################################
def LoadInputData_LHC18b8(recreate):

    # Load data from HFD5 store if already created
    data = None
    if os.path.isfile('./dataset.h5') and not recreate:
        store = pd.HDFStore('./dataset.h5')
    if 'LHC18b8' in store:
        print("")
        print("### Loading input dataset ({}) from HDF5 store... ####".format('LHC18b8'))
        data = store['LHC18b8']

        # If data could not be loaded -> create the HDF5 store
        if data is None:
            print("")
    print("#### Creating input dataset for HDF5 store... this will take some time ####")

    import multiprocessing
    maxParts = 3
    try:
        pool = multiprocessing.Pool(8)
        data_binned_b  = pool.map(partial(ReadRootFile_LHC18b8, 'bJets'), range(1,maxParts+1))
        data_binned_c  = pool.map(partial(ReadRootFile_LHC18b8, 'cJets'), range(1,maxParts+1))
        data_binned_lf = pool.map(partial(ReadRootFile_LHC18b8, 'udsgJets'), range(1,maxParts+1))
    finally:
        pool.close()
        pool.join()

    data_b  = pd.concat(data_binned_b, copy=False)
    data_c  = pd.concat(data_binned_c, copy=False)
    data_lf = pd.concat(data_binned_lf, copy=False)
    data    = pd.concat([data_b, data_c, data_lf], copy=False)

    # Save the data to the HDF5 store
    store = pd.HDFStore('./dataset.h5')
    store['LHC18b8'] = data

    print("... done")
    print("")
    data = data.fillna(0.0)
    return data

#########################################################
def ReadRootFile_LHC17pq(jetType, part):
    import progressbar, math
    columns = ['Jet_Pt', 'Jet_Area', 'Jet_NumTracks', 'Jet_Track_IPd', 'Jet_Track_IPz', 'Jet_Track_CovIPd', 'Jet_Track_CovIPz', 'Jet_SecVtx_Lxy', 'Jet_NumSecVertices', 'Jet_SecVtx_SigmaLxy', 'Jet_SecVtx_Chi2', 'Jet_SecVtx_Dispersion']

    fname = '/eos/user/h/hahassan/Files_For_ML/LHC17pq/LHC17pq_Part_{}.root'.format(part)
    data_raw = root_pandas.read_root(fname, 'JetTree_AliAnalysisTaskJetExtractor_Jet_AKTChargedR040_tracks_pT0150_E_scheme_{}'.format(jetType), columns=columns)

    # Preprocess data
    data = pd.DataFrame(data=data_raw['Jet_Pt'])
    data['Jet_Area']            = data_raw['Jet_Area']
    data['Jet_NumTracks']       = data_raw['Jet_NumTracks']
    data['Jet_NumSecVertices']  = data_raw['Jet_NumSecVertices']
    data['Jet_Area']            = data_raw['Jet_Area']

    for i in range(kNumMaxConstituents):
        data['Jet_Track_IPd_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))
    data['Jet_Track_IPz_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))
    data['Jet_Track_CovIPd_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))
    data['Jet_Track_CovIPz_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))

    for i in range(kNumMaxSecVertices):
        data['Jet_SecVtx_Lxy_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))
    data['Jet_SecVtx_SigmaLxy_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))
    data['Jet_SecVtx_Chi2_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))
    data['Jet_SecVtx_Dispersion_{}'.format(i)] = pd.Series(data=np.zeros_like(data['Jet_Pt']))

    jetID = 0
    #####  Loop over all jets
    for constIPd, constIPz, constCovIPd, constCovIPz, secVtxLxy, secVtxSigmaLxy, secVtxChi2, secVtxDispersion in zip(data_raw['Jet_Track_IPd'], data_raw['Jet_Track_IPz'],
                                                                                                                     data_raw['Jet_Track_CovIPd'], data_raw['Jet_Track_CovIPz'], data_raw['Jet_SecVtx_Lxy'], data_raw['Jet_SecVtx_SigmaLxy'], data_raw['Jet_SecVtx_Chi2'], data_raw['Jet_SecVtx_Dispersion']):

        #startTime = time.time()

    # Sort tracks for descending sig. impact parameters d
    sigIPd = constIPd/np.sqrt(constCovIPd)
    indices = sigIPd.argsort()[::-1]
    indices = indices[:kNumMaxConstituents]
    constIPd = constIPd[indices]
    constIPz = constIPz[indices]
    constCovIPd = constCovIPd[indices]
    constCovIPz = constCovIPz[indices]

    # Loop over constituents
    for i in range(len(indices)):
        data['Jet_Track_IPd_{}'.format(i)].iat[jetID] = constIPd[i]
        data['Jet_Track_IPz_{}'.format(i)].iat[jetID] = constIPz[i]
        data['Jet_Track_CovIPd_{}'.format(i)].iat[jetID] = constCovIPd[i]
        data['Jet_Track_CovIPz_{}'.format(i)].iat[jetID] = constCovIPz[i]

    # Sort sec. vertices for descending sig. Lxy
    sigLxy = secVtxLxy/(secVtxSigmaLxy)
    indices = sigLxy.argsort()[::-1]
    indices = indices[:kNumMaxSecVertices]
    secVtxLxy = secVtxLxy[indices]
    secVtxSigmaLxy = secVtxSigmaLxy[indices]
    secVtxChi2 = secVtxChi2[indices]
    secVtxDispersion = secVtxDispersion[indices]

    # Loop over sec. vertices
    for i in range(len(indices)):
        data['Jet_SecVtx_Lxy_{}'.format(i)].iat[jetID] = secVtxLxy[i]
        data['Jet_SecVtx_SigmaLxy_{}'.format(i)].iat[jetID] = secVtxSigmaLxy[i]
        data['Jet_SecVtx_Chi2_{}'.format(i)].iat[jetID] = secVtxChi2[i]
        data['Jet_SecVtx_Dispersion_{}'.format(i)].iat[jetID] = secVtxDispersion[i]

    #print('Loop time: {}'.format(time.time()-startTime))
    jetID += 1

    print('... -- {} samples.'.format(len(data)))
    data = data.fillna(0.0)
    return data


#########################################################
def LoadInputData_LHC17pq(recreate):

    # Load data from HFD5 store if already created
    data = None
    if os.path.isfile('./dataset.h5') and not recreate:
        store = pd.HDFStore('./dataset.h5')
    if 'LHC17pq' in store:
        print("")
        print("### Loading input dataset ({}) from HDF5 store... ####".format('LHC17pq'))
        data = store['LHC17pq']

        # If data could not be loaded -> create the HDF5 store
        if data is None:
            print("")
    print("#### Creating input dataset for HDF5 store... this will take some time ####")

    if 1:
        import multiprocessing
        maxParts = 11
        try:
            pool = multiprocessing.Pool(8)
        data_binned = pool.map(partial(ReadRootFile_LHC17pq, 'allJets'), [1,2,3,4,5,6,7,8,9,10,11])
        finally:
            pool.close()
        pool.join()
    else:
        data_binned =  []
        for i in range(1, 10):
            data_binned += [ReadRootFile_LHC17pq('allJets', i)]

    data  = pd.concat(data_binned, copy=False)

    # Save the data to the HDF5 store
    store = pd.HDFStore('./dataset.h5')
    store['LHC17pq'] = data

    print("... done")
    print("")
    data = data.fillna(0.0)
    return data

#########################################################
def CheckData(data, name):
    """
    Data exploration
    """

    print("Checking Data/MC")

    import seaborn
    dataC = pd.DataFrame(data)
    dataC.drop(['Jet_Track_IPd_{}'.format(i) for i in range(2, kNumMaxConstituents)], inplace=True,axis=1)
    dataC.drop(['Jet_Track_IPz_{}'.format(i) for i in range(2, kNumMaxConstituents)], inplace=True,axis=1)
    dataC.drop(['Jet_Track_CovIPd_{}'.format(i) for i in range(2, kNumMaxConstituents)], inplace=True,axis=1)
    dataC.drop(['Jet_Track_CovIPz_{}'.format(i) for i in range(2, kNumMaxConstituents)], inplace=True,axis=1)

    dataC.drop(['Jet_SecVtx_Lxy_{}'.format(i) for i in range(2, kNumMaxSecVertices)], inplace=True,axis=1)
    dataC.drop(['Jet_SecVtx_SigmaLxy_{}'.format(i) for i in range(2, kNumMaxSecVertices)], inplace=True,axis=1)
    dataC.drop(['Jet_SecVtx_Chi2_{}'.format(i) for i in range(2, kNumMaxSecVertices)], inplace=True,axis=1)
    dataC.drop(['Jet_SecVtx_Dispersion_{}'.format(i) for i in range(2, kNumMaxSecVertices)], inplace=True,axis=1)

    dataC.hist(bins=50, figsize=(20,15))
    plt.savefig('./DataOverview_{}.png'.format(name), dpi=120)
    plt.clf()

    #correlation matrix
    seaborn.heatmap( dataC.corr(), square=False, vmin=-1, vmax=1, cmap="PiYG")
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.10)
    plt.savefig('./DataCorrelation_{}.png'.format(name), dpi=120)
    plt.clf()

    # Pair correlations
    seaborn.pairplot(data, hue='Jet_MC_MotherHadron', vars=['Jet_Track_IPd_0', 'Jet_SecVtx_Lxy_0', 'Jet_SecVtx_Lxy_1', 'Jet_SecVtx_Lxy_2', 'Jet_SecVtx_Lxy_3', 'Jet_SecVtx_Lxy_4', 'Jet_SecVtx_Dispersion_0'])
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.10)
    plt.savefig('./PairCorrelations_{}.png'.format(name), dpi=120)
    plt.clf()

#########################################################
def AddScoresToFiles_MC(models, part):
    """ Add scores to file
    Can be used in multiprocessing
    """
    import btagging_keras, btagging_models
    fname = '/Users/hadi/ML_Rudiger/LHC18b8_pThardBin_{}.root'.format(part)
    data_b    = ReadRootFile_LHC18b8('bJets', part, truncate=False)
    data_c    = ReadRootFile_LHC18b8('cJets', part, truncate=False)
    data_udsg = ReadRootFile_LHC18b8('udsgJets', part, truncate=False)

    for dat in [data_b, data_c, data_udsg]:
        dat.drop(['Jet_MC_MotherHadron', 'Event_PtHardBin'], inplace=True, axis=1)

        # Load model and apply it
        print("### Adding model predictions ...")
        predictions_b = []
        predictions_c = []
        predictions_udsg = []
        for modelName in models:
            if not 'keras' in modelName.lower():
                import joblib
                model = joblib.load('./Models/Model_{}_Target{}.pkl'.format(modelName, gTarget))
                predictions_b     += [model.predict_proba(data_b)[:,1]]
                predictions_c     += [model.predict_proba(data_c)[:,1]]
                predictions_udsg  += [model.predict_proba(data_udsg)[:,1]]
    else:
        model = btagging_keras.AliMLKerasModel(2)
        model.LoadModel('Model_{}_Target{}'.format(modelName, gTarget))
        data_keras_b    = btagging_models.GetDataForKeras(data_b)
        data_keras_c    = btagging_models.GetDataForKeras(data_c)
        data_keras_udsg = btagging_models.GetDataForKeras(data_udsg)
        predictions_b     += [model.fModel.predict(data_keras_b, batch_size=512, verbose=0)[:,0]]
        predictions_c     += [model.fModel.predict(data_keras_c, batch_size=512, verbose=0)[:,0]]
        predictions_udsg  += [model.fModel.predict(data_keras_udsg, batch_size=512, verbose=0)[:,0]]
        print("... done")

        # Write results as new Tree to file
        AddBranchesToTreeInFile(fname, 'Scores_{}'.format('bJets'), models, predictions_b, overwrite=True)
        AddBranchesToTreeInFile(fname, 'Scores_{}'.format('cJets'), models, predictions_c, overwrite=True)
        AddBranchesToTreeInFile(fname, 'Scores_{}'.format('udsgJets'), models, predictions_udsg, overwrite=True)
        print("... Added predictions to Trees")


#########################################################
def AddScoresToFiles(models, part):
    """ Add scores to file
    Can be used in multiprocessing
    """
    import btagging_keras, btagging_models
    fname = '/opt/Data/bJets/LHC17pq_Part{}.root'.format(part)
    data = ReadRootFile_LHC17pq('allJets', part)

    # Load model and apply it
    print("### Adding model predictions ...")
    predictions = []
    for modelName in models:
        if not 'keras' in modelName.lower():
            import joblib
            model = joblib.load('./Models/Model_{}_Target{}.pkl'.format(modelName, gTarget))
            predictions += [model.predict_proba(data)[:,1]]
    else:
        model = btagging_keras.AliMLKerasModel(2)
        model.LoadModel('Model_{}_Target{}'.format(modelName, gTarget))
        data_keras = btagging_models.GetDataForKeras(data)
        predictions += [model.fModel.predict(data_keras, batch_size=512, verbose=0)[:,0]]
        print("... done")

        # Write results as new Tree to file
        AddBranchesToTreeInFile(fname, 'Scores_{}'.format('allJets'), models, predictions, overwrite=True)
        print("... Added predictions to Trees")


#########################################################
def AddBranchesToTreeInFile(fname, tname, bNames, bData, offset=0, overwrite=False):
    """Add custom branches to tree in file"""

    ##### Try to read file & tree
    ofile   = ROOT.TFile(fname, 'UPDATE')
    outTree = ofile.Get(tname)
    if not outTree or overwrite == True:
        outTree = ROOT.TTree(tname, tname)

        if overwrite == False:
            offset = outTree.GetEntries()

            ##### Create branches, if they do not exist
            outBranches = [None,] * len(bNames)
            outBuffer   = [None, ] * len(bNames)
            for i, bname in enumerate(bNames):
                outBranches[i] = outTree.GetBranch(bname)
    outBuffer[i]   = np.zeros(1, dtype=float)
    if not outBranches[i]:
        outBranches[i] = outTree.Branch(bname, outBuffer[i], '{:s}/D'.format(bname))
    else:
        outTree.SetBranchAddress(bname, outBuffer[i])

        ##### Loop through the chain and add raw samples to output tree
        for iData in range(len(bData[0])):

            for iType in range(len(bData)):
                try:
                    outBuffer[iType][0] = bData[iType][iData][0]
                except:
                    outBuffer[iType][0] = bData[iType][iData]

    outTree.GetEntry(offset+iData)
    outTree.Fill()

    outTree.Write('', ROOT.TObject.kOverwrite)
    ofile.Close()
