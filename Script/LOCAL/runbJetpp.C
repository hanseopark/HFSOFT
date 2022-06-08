#if !defined(__CINT__) || defined(__CLING__)
#include "AliMCEventHandler.h"
#include "AliESDInputHandler.h"
#include "AliAODInputHandler.h"
#include "AliAnalysisAlien.h"
#include "AliAnalysisManager.h"
#include <TStopwatch.h>

#include "AliAnalysisTaskMyTask.h"

R__ADD_INCLUDE_PATH($ALICE_PHYSICS)
// #include <PWG/EMCAL/macros/AddTaskAodSkim.C>
// #include <PWG/EMCAL/macros/AddTaskEsdSkim.C>

#include <PWGHF/vertexingHF/macros/AddTaskImproveITSCVMFS.C>

#include <PWG/EMCAL/macros/AddTaskMCTrackSelector.C>
#include <PWGJE/EMCALJetTasks/macros/AddTaskEmcalJet.C>
#include <PWGGA/GammaConv/macros/AddTask_GammaOutlierRemoval.C>

//#include <PWGPP/EMCAL/macros/AddTaskEMCALPi0CalibrationV2.C>

#endif
#include "localRunningChain.h"


//______________________________________________________________________________
void runbJetpp(
    Int_t           intMCrunning                = 0,
    Int_t           collsys                     = 0, //0 pp, 1 pPb, 2 PbPb
    TString         runPeriod                   = "LHC15n",
    TString         runPeriodData               = "LHC15n",
    TString         dataType                    = "AOD",
    TString         runMode                     = "PQ2HC",//P:PCM, 2:PCM+Tree, Q:PhotonQA, H:hybrid PCMEMC, C: EMC
    Int_t           recoPassData                = 4,
    TString         tenderPassData              = "pass4",
    Bool_t          useCorrTask                 = kFALSE,
    TString         aodConversionCutnumber      = "10000003_06000008400000001000000000",
    Bool_t          isRun2                      = kFALSE,
    UInt_t          numLocalFiles               = 50,
	Bool_t          isLxplus                    = kFALSE,
    Int_t           chunk                       = -1
)
{
    // since we will compile a class, tell root where to look for headers
    #if !defined (__CINT__) || defined (__CLING__)
        gInterpreter->ProcessLine(".include $ROOTSYS/include");
        gInterpreter->ProcessLine(".include $ALICE_ROOT/include");
    #else
        gROOT->ProcessLine(".include $ROOTSYS/include");
        gROOT->ProcessLine(".include $ALICE_ROOT/include");
    #endif

    // Create analysis manager
    AliAnalysisManager* mgr                     = new AliAnalysisManager("LocalAnalysisTaskRunning");

    // change this objects to strings
    TString usedData(dataType);
    cout << dataType.Data() << " analysis chosen" << endl;
    // Check type of input and create handler for it
    TString localFiles("-1");
	if(isLxplus) localFiles                       = Form("../test16g%s_lx.txt",dataType.Data());
	else localFiles                               = Form("../test16g%s.txt",dataType.Data());
    if(chunk != -1)
      localFiles                                  = Form("../testSample%s_%d.txt",dataType.Data(),chunk);

    if(dataType.Contains("AOD")){
        AliAODInputHandler* aodH                = new AliAODInputHandler();
        aodH->AddFriend((char*)"AliAODGammaConversion.root");
        mgr->SetInputEventHandler(aodH);
    } else if(dataType.Contains("ESD")){
        AliESDInputHandler* esdH                = new AliESDInputHandler();
        mgr->SetInputEventHandler(esdH);
    } else {
        cout << "Data type not recognized! You have to specify ESD, AOD, or sESD!\n";
    }

    cout << "Using " << localFiles.Data() << " as input file list.\n";

    // Create MC handler, if MC is demanded
    if (intMCrunning && (!dataType.Contains("AOD")))
    {
        AliMCEventHandler* mcH                  = new AliMCEventHandler();
        mcH->SetPreReadMode(AliMCEventHandler::kLmPreRead);
        mcH->SetReadTR(kTRUE);
        mgr->SetMCtruthEventHandler(mcH);
    }
    // -----------------------------------------
    //                CDB CONNECT
    // -----------------------------------------
    #if !defined (__CINT__) || defined (__CLING__)
        AliTaskCDBconnect *taskCDB=reinterpret_cast<AliTaskCDBconnect*>(
        gInterpreter->ExecuteMacro("$ALICE_PHYSICS/PWGPP/PilotTrain/AddTaskCDBconnect.C()"));
        taskCDB->SetFallBackToRaw(kTRUE);
    #else
        gROOT->LoadMacro("$ALICE_PHYSICS/PWGPP/PilotTrain/AddTaskCDBconnect.C");
        AliTaskCDBconnect *taskCDB = AddTaskCDBconnect();
        taskCDB->SetFallBackToRaw(kTRUE);
    #endif

    // -----------------------------------------
    //            ImproverTask_CVMF
    // -----------------------------------------
    #if !defined (__CINT__) || defined (__CLING__)
		AliAnalysisTaskSEImproveITSCVMFS *taskSEI=reinbterpret_cast<AliAnalysisTaskSEImproveITSCVMFS*>(
		gInterpreter->ExecuteMacro("$ALICE_PHYSICS/PWGHF/vertexingHF/macros/AddTaskImproveITSCVMFS.C"));
    #else
		gROOT->LoadMacro("ALICE_PHYSICS/PWGHF/vertexingHF/macros/AddTaskImproveITSCVMFS.C");
		AliAnalysisTaskSEImproveITSCVMFS *taskSEI = AddTaskImproveITSCVMFS();
	#endif

    mgr->SetUseProgressBar(1, 1);
    if (!mgr->InitAnalysis()) return;

    mgr->PrintStatus();

    // LOCAL CALCULATION
    TChain* chain = 0;
    //TChain* chain = new TChain("aodTree");
    if (usedData == "AOD") {
        chain = CreateAODChain(localFiles.Data(), numLocalFiles);
        // chain = CreateAODChain(localFiles.Data(), numLocalFiles,0,kFALSE,"AliAODGammaConversion.root");
        //chain->Add("/Users/hanseopark/alice/work/Data/LocalFiles/TEMP/LHC16d/pass1/0/AliAOD.root");
    } else {  // ESD
        chain = CreateESDChain(localFiles.Data(), numLocalFiles);
    }

    cout << endl << endl;
    cout << "****************************************" << endl;
    cout << "*                                      *" << endl;
    cout << "*            start analysis            *" << endl;
    cout << "*                                      *" << endl;
    cout << "****************************************" << endl;
    cout << endl << endl;

    TStopwatch watch;
	  watch.Start();

    // start analysis
    cout << "Starting LOCAL Analysis...";
    mgr->SetDebugLevel(5);
    mgr->StartAnalysis("local", chain); //test, local
    // mgr->StartAnalysis("test", chain); //test, local


//	// Grid analysis
//    cout << "Starting GRID Analysis...";
//	// create an instance of the plugin
//    AliAnalysisAlien *alienHandler = new AliAnalysisAlien();
// // where the headers can be found
//	alienHandler->AddIncludePath("-I. -I$ROOTSYS/include -I$ALICE_ROOT -I$ALICE_ROOT/include -I$ALICE_PHYSICS/include");
//
//	mgr->SetGridHandler(alienHandler);
//	alienHandler->SetRunMode("test");


    cout << endl << endl;
    cout << "****************************************" << endl;
    cout << "*                                      *" << endl;
    cout << "*             end analysis             *" << endl;
    cout << "*                                      *" << endl;
    cout << "****************************************" << endl;
    cout << endl << endl;

    watch.Stop();
	  watch.Print();
}
