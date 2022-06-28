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
    Int_t           intMCrunning                = 0, //0: data, 1: MC, 2: JJ MC
    Int_t           collsys                     = 0, //0 pp, 1 pPb, 2 PbPb
    TString         runPeriod                   = "LHC15n",
    TString         dataType                    = "AOD",
    TString         runMode                     = "PQ2HC",//P:PCM, 2:PCM+Tree, Q:PhotonQA, H:hybrid PCMEMC, C: EMC
    Int_t           recoPassData                = 4,
    TString         tenderPassData              = "pass4",
    Bool_t          useCorrTask                 = kFALSE,
    TString         aodConversionCutnumber      = "10000003_06000008400000001000000000",
    Bool_t          isRun2                      = kFALSE,
    UInt_t          numLocalFiles               = 50,
	Bool_t			isPileup					= kFALSE,
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
	if(isLxplus) localFiles                       = Form("../test%s%s_lx.txt",runPeriod.Data(),dataType.Data());
	else localFiles                               = Form("../test%s%s.txt",runPeriod.Data(),dataType.Data());
    if(chunk != -1)
      localFiles                                  = Form("../testSample%s_%d.txt",dataType.Data(),chunk);

    if(dataType.Contains("AOD")){
        AliAODInputHandler* aodH                = new AliAODInputHandler();
        //aodH->AddFriend((char*)"AliAODGammaConversion.root");
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
    //            PHYSICS SELECTION
    // -----------------------------------------
    #if !defined (__CINT__) || defined (__CLING__)
        AliPhysicsSelectionTask *physSelTask=reinterpret_cast<AliPhysicsSelectionTask*>(
        //gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/OADB/macros/AddTaskPhysicsSelection.C(%i, %i)",intMCrunning ? 1 : 0, kTRUE)));
        gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/OADB/macros/AddTaskPhysicsSelection.C(%i, %i)",intMCrunning,kTRUE)));
    #else
        gROOT->LoadMacro("$ALICE_PHYSICS/OADB/macros/AddTaskPhysicsSelection.C");
        AliPhysicsSelectionTask* physSelTask = AddTaskPhysicsSelection(intMCrunning);
    #endif


    // -----------------------------------------
    //               PID RESPONSE
    // -----------------------------------------
    #if !defined (__CINT__) || defined (__CLING__)
        AliAnalysisTaskPIDResponse *pidRespTask=reinterpret_cast<AliAnalysisTaskPIDResponse*>(
        //gInterpreter->ExecuteMacro(Form("$ALICE_ROOT/ANALYSIS/macros/AddTaskPIDResponse.C(%i, %i, %i,\"%i\")",intMCrunning,kTRUE,0,recoPassData)));
        gInterpreter->ExecuteMacro(Form("$ALICE_ROOT/ANALYSIS/macros/AddTaskPIDResponse.C(%i, %i, %i,\"%i\")",intMCrunning,kFALSE,kFALSE,recoPassData)));
    #else
        gROOT->LoadMacro("$ALICE_ROOT/ANALYSIS/macros/AddTaskPIDResponse.C");
        AddTaskPIDResponse(intMCrunning,kFALSE,kFALSE);
    #endif


    // -----------------------------------------
    //            ImproverTask_CVMF
    // -----------------------------------------
    #if !defined (__CINT__) || defined (__CLING__)
		AliAnalysisTaskSEImproveITSCVMFS *taskImp=reinterpret_cast<AliAnalysisTaskSEImproveITSCVMFS*>(
		gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/PWGHF/vertexingHF/macros/AddTaskImproveITSCVMFS.C(%i, \"%s\", \"%s\", %i)",kTRUE,"","",0)));
		//gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/PWGHF/vertexingHF/macros/AddTaskImproveITSCVMFS.C(%i, \"%s\", \"%s\", %i)",kFALSE,"","",0)));
    #else
		gROOT->LoadMacro("ALICE_PHYSICS/PWGHF/vertexingHF/macros/AddTaskImproveITSCVMFS.C");
		AliAnalysisTaskSEImproveITSCVMFS *taskImp = AddTaskImproveITSCVMFS();
	#endif

    // -----------------------------------------
    //               Jet
    // -----------------------------------------
	
    #if !defined (__CINT__) || defined (__CLING__)
		//AliEmcalJetTask *taskjetReader = reinterpret_cast<AliEmcalJetTask*>(
		//gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskEmcalJet.C(\"%s\",\"%s\",%s,%f,AliJetContainer::kChargedJet,%f,%f,%f,AliJetContainer::pt_scheme,\"%s\",%f,kFALSE,kFALSE)","mcparticles","",AliJetContainer::antikt_algorithm,0.4,0.15,0.,0.01,"Jet",10.)));
		AddTaskEmcalJet("mcparticles", "", AliJetContainer::antikt_algorithm, 0.4, AliJetContainer::kChargedJet, 0.15, 0., 0.01, AliJetContainer::pt_scheme, "Jet", 10., kFALSE, kFALSE);
		AddTaskEmcalJet("tracks", "", AliJetContainer::antikt_algorithm, 0.4, AliJetContainer::kChargedJet, 0.15, 0., 0.01, AliJetContainer::pt_scheme, "Jet", 10., kFALSE, kFALSE);

		//AliEmcalJetTask* EMCJetTask=reinterpret_cast<AliEmcalJetTask*>(
		//gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskEmcalJet.C(\"mcparticles\", \"\", AliJetContainer::antikt_algorithm, 0.4, AliJetContainer::kChargedJet, 0.15, 0., 0.01, AliJetContainer::pt_scheme, \"Jet\", 10., kFALSE, kFALSE))")));

		//EMCJetTask->GetTrackContainer(0)->SetTrackFilterType(AliEmcalTrackSelection::kCustomTrackFilter);
		//EMCJetTask->GetTrackContainer(0)->SetAODFilterBits((1<<4)|(1<<9));
		//EMCJetTask->SelectCollisionCandidates(AliVEvent::kAny);
		//EMCJetTask->SetUseNewCentralityEstimation(kTRUE);
    #else
		gROOT->LoadMacro("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskEmcalJet.C");
    #endif

	Int_t arr[22]  = {0, 5, 7, 9, 12, 16, 21, 28, 36, 45, 57, 70, 85, 99, 115, 132, 150, 169, 190, 212, 235, 10000000};//
	TArrayI bins(22, arr);
    #if !defined (__CINT__) || defined (__CLING__)
		AliAnalysisTaskJetExtractor *taskJet = reinterpret_cast<AliAnalysisTaskJetExtractor*>(
		gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskJetExtractor.C(\"%s\",\"%s\",\"%s\",\"%s\",%f,\"%s\")","tracks","","Jet_AKTChargedR040_tracks_pT0150_pt_scheme","",0.4,"bJets")));
		//gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskJetExtractor.C(\"%s\",\"%s\",\"%s\",\"%s\",%f,\"%s\")","tracks","","Jet_AKTChargedR040_mcparticles_pT0150_pt_scheme","",0.4,"bJets")));
//		AliAnalysisTaskJetExtractor *taskJet = reinterpret_cast<AliAnalysisTaskJetExtractor*>(
//		gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskJetExtractor.C(\"%s\",\"%s\",\"%s\",\"%s\",%f,\"%s\")","tracks","","Jet_AKTChargedR040_tracks_pT0150_pt_scheme","",0.4,"bJets")));

//		taskJet->SetIsPythia(kTRUE);
//		taskJet->SelectCollisionCandidates(AliVEvent::kINT7);
//		taskJet->SetForceBeamType(AliAnalysisTaskEmcal::kpp);
//		taskJet->SetVzRange(-10,10);
//		taskJet->SetSaveConstituents(1);
//		taskJet->SetSaveConstituentsIP(1);
//		taskJet->SetSaveConstituentPID(0);
//		taskJet->SetSaveJetShapes(1);
//		taskJet->SetSaveJetSplittings(1);
//		taskJet->SetSaveSecondaryVertices(1);
//		taskJet->SetSaveTriggerTracks(0);
//		taskJet->SetSaveMCInformation(1);
//		taskJet->GetJetTree()->AddExtractionPercentage(0,10, 0.3);
//		taskJet->GetJetTree()->AddExtractionPercentage(10,20, 0.6);
//		taskJet->GetJetTree()->AddExtractionPercentage(20,40, 0.8);
//		taskJet->GetJetTree()->AddExtractionPercentage(40,200, 1.0);
//		taskJet->GetJetTree()->AddExtractionJetTypeHM(5);
//		taskJet->GetJetContainer(0)->SetJetRadius(0.4);
//		taskJet->GetJetContainer(0)->SetPercAreaCut(0.6);
//		taskJet->GetJetContainer(0)->SetJetEtaLimits(-0.5, 0.5);
//		taskJet->GetJetContainer(0)->SetMaxTrackPt(1000);
//		taskJet->GetTrackContainer(0)->SetTrackFilterType(AliEmcalTrackSelection::kCustomTrackFilter);
//		taskJet->GetTrackContainer(0)->SetAODFilterBits((1<<4)|(1<<9));
//		taskJet->SetNumberOfPtHardBins(bins.GetSize()-1);
//		taskJet->SetUserPtHardBinning(bins);
    #else
		gROOT->LoadMacro("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskJetExtractor.C");
    #endif
	

        //gInterpreter->ExecuteMacro(Form("$ALICE_ROOT/ANALYSIS/macros/AddTaskPIDResponse.C(%i, %i, %i,\"%i\")",intMCrunning,kFALSE,kFALSE,recoPassData)));
//////////////////////////////////////////////////////////////////////////////

//	////////// FOR MY TASK /////////
//	#if !defined (__CINT__) || defined (__CLING__)
//		gInterpreter->LoadMacro("$HFDIR/Script/LOCAL/AliAnalysisTaskMyTask.cxx++g");
////		AliAnalysisTaskMyTask *task = reinterpret_cast<AliAnalysisTaskMyTask*>(gInterpreter->ExecuteMacro("AddMyTask.C"));
//		AliAnalysisTaskMyTask *taskMe=reinterpret_cast<AliAnalysisTaskMyTask*>(
//		gInterpreter->ExecuteMacro("$HFDIR/Script/Local/AddMyTask.C"));
//	#else
//		gROOT->LoadMacro("$HFDIR/Script/LOCAL/AliAnalysisTaskMyTask.cxx++g");
//		gROOT->LoadMacro("$HFDIR/Script/LOCAL/AddMyTask.C");
//		AliAnalysisTaskMyTask *taskMe = AddMyTask();
//	#endif



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
    //mgr->SetDebugLevel(1);
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
