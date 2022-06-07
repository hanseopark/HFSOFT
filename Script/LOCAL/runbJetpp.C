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
#include <PWGGA/GammaConv/macros/AddTask_V0Reader.C>
//#include <PWGGA/GammaConv/macros/AddTask_ClusterQA.C>
//#include <PWGGA/GammaConv/macros/AddTask_PhotonQA.C>
#include <PWGGA/GammaConv/macros/AddTask_GammaConvV1_pp.C>
//#include <PWGGA/GammaConv/macros/AddTask_GammaConvV1_pPb.C>
//#include <PWGGA/GammaConv/macros/AddTask_GammaConvV1_PbPb.C>
#include <PWGGA/GammaConv/macros/AddTask_GammaCalo_pp.C>
//#include <PWGGA/GammaConv/macros/AddTask_GammaCalo_pPb.C>
#include <PWGGA/GammaConv/macros/AddTask_GammaCalo_PbPb.C>
#include <PWGGA/GammaConv/macros/AddTask_GammaConvCalo_pp.C>
#include <PWGGA/GammaConv/macros/AddTask_GammaHeavyMeson_ConvMode_pp.C>
#include <PWGGA/GammaConv/macros/AddTask_OmegaToPiZeroGamma_pp.C>
//#include <PWGGA/GammaConv/macros/AddTask_GammaCaloMerged_pp.C>
//#include <PWGGA/GammaConv/macros/AddTask_GammaCaloMerged_pPb.C>
//#include <PWGGA/GammaConv/macros/AddTask_GammaCaloMerged_PbPb.C>
#include <PWGGA/GammaConv/macros/AddTask_GammaConvNeutralMesonPiPlPiMiNeutralMeson_CaloMode_pp.C>
#include <PWGGA/GammaConv/macros/AddTask_GammaConvNeutralMesonPiPlPiMiNeutralMeson_MixedMode_pp.C>
#include <PWGGA/GammaConv/macros/AddTask_GammaConvCaloCalibration_MixedMode_pp.C>
//#include <PWGGA/GammaConv/macros/AddTask_GammaConvNeutralMesonPiPlPiMiNeutralMeson_CaloMode_pPb.C>

#include <PWGGA/GammaConv/macros/AddTask_ConvCaloCalibration_CaloMode_pPb.C>
#include <PWGGA/GammaConv/macros/AddTask_ConvCaloCalibration_CaloMode_pp.C>
#include <PWGGA/GammaConv/macros/AddTask_GammaConvCaloCalibration_MixedMode_pPb.C>
#include <PWGGA/GammaConv/macros/AddTask_GammaHeavyMeson_MixedMode_pp.C>
#include <PWGGA/GammaConv/macros/AddTask_GammaHeavyMeson_CaloMode_pp.C>
#include <PWGGA/GammaConv/macros/AddTask_ClusterQA.C>
//#include <PWGGA/GammaConv/macros/AddTask_ConvCaloTree.C>
#include <PWGGA/GammaConv/macros/AddTask_GammaCaloMerged_pp.C>

#include <PWG/EMCAL/macros/AddTaskMCTrackSelector.C>
#include <PWGJE/EMCALJetTasks/macros/AddTaskEmcalJet.C>
#include <PWGGA/GammaConv/macros/AddTask_GammaOutlierRemoval.C>
//#include <PWGPP/EMCAL/macros/AddTaskEMCALPi0CalibrationV2.C>

#endif
#include "localRunningChain.h"


//______________________________________________________________________________
void runLocalAnalysisROOT6(
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
    //            PHYSICS SELECTION
    // -----------------------------------------
    #if !defined (__CINT__) || defined (__CLING__)
        AliPhysicsSelectionTask *physSelTask=reinterpret_cast<AliPhysicsSelectionTask*>(
        gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/OADB/macros/AddTaskPhysicsSelection.C(%i)",intMCrunning ? 1 : 0)));
    #else
        gROOT->LoadMacro("$ALICE_PHYSICS/OADB/macros/AddTaskPhysicsSelection.C");
        AliPhysicsSelectionTask* physSelTask = AddTaskPhysicsSelection(intMCrunning);
    #endif

    // -----------------------------------------
    //               PID RESPONSE
    // -----------------------------------------
    #if !defined (__CINT__) || defined (__CLING__)
        AliAnalysisTaskPIDResponse *pidRespTask=reinterpret_cast<AliAnalysisTaskPIDResponse*>(
        gInterpreter->ExecuteMacro(Form("$ALICE_ROOT/ANALYSIS/macros/AddTaskPIDResponse.C(%i, %i, %i, \"%s\")",intMCrunning,kFALSE,kFALSE,tenderPassData.Data())));
    #else
        gROOT->LoadMacro("$ALICE_ROOT/ANALYSIS/macros/AddTaskPIDResponse.C");
        AddTaskPIDResponse(intMCrunning,kFALSE,kTRUE,tenderPassData);
    #endif

//    // -----------------------------------------
//    //               MULT SELECTION
//    // -----------------------------------------
//	#if !defined (__CINT__) || defined (__CLING__)
//		if(isRun2){
//			AliMultSelectionTask *multSelTask=reinterpret_cast<AliMultSelectionTask*>(
//					gInterpreter->ExecuteMacro("$ALICE_PHYSICS/OADB/COMMON/MULTIPLICITY/macros/AddTaskMultSelection.C(kFALSE)"));
//			// multSelTask->SetAlternateOADBforEstimators("LHC18c12");
//		} else {
//			AliCentralitySelectionTask *multSelTask=reinterpret_cast<AliCentralitySelectionTask*>(
//					gInterpreter->ExecuteMacro("$ALICE_PHYSICS/OADB/macros/AddTaskCentrality.C(kFALSE,kTRUE)"));
//		}
//	#else
//		if(isRun2){
//			gROOT->LoadMacro("$ALICE_PHYSICS/OADB/COMMON/MULTIPLICITY/macros/AddTaskMultSelection.C");
//			AddTaskMultSelection();
//		} else {
//			gROOT->LoadMacro("$ALICE_PHYSICS/OADB/macros/AddTaskCentrality.C");
//			AddTaskCentrality(kFALSE,kTRUE);
//		}
//	#endif
//
		// -----------------------------------------
		//   EMCAL TENDER or CORRECTION FRAMEWORK
		// -----------------------------------------
     TString taskNameSpecial                     = ""; // NCellEffPCMEMCTagging
     TString taskNameSpecial2                     = "";
     TString taskNameSpecial3                     = "";
     TString taskNameSpecial4                     = "";
     const UInt_t  kPhysSel                      = AliVEvent::kAny;
     if(useCorrTask){
         AliEmcalCorrectionTask * correctionTask = AliEmcalCorrectionTask::AddTaskEmcalCorrectionTask("");
         correctionTask->SelectCollisionCandidates(kPhysSel);
         // correctionTask->SetForceBeamType(AliEmcalCorrectionTask::kpp);
         if(intMCrunning==0){
			correctionTask->SetUserConfigurationFilename("$ALICE_PHYSICS/PWG/EMCAL/config/PWGJESampleConfig.yaml");
             // correctionTask->SetUserConfigurationFilename("/Users/hanseopark/alice/AliPhysics/PWG/EMCAL/config/AliEmcalCorrectionConfiguration.yaml");
			 // correctionTask->SetUserConfigurationFilename("/afs/cern.ch/user/h/hapark/hapark/Local_Analysis/AliEmcalCorrectionConfiguration_park.yaml"); // For lxplus
           }
         else{
			correctionTask->SetUserConfigurationFilename("$ALICE_PHYSICS/PWG/EMCAL/config/PWGJESampleConfig.yaml");
           }
		 correctionTask->Initialize(true);
//		 taskNameSpecial                 = "S500A100";
//		 AliEmcalCorrectionTask * correctionTaskSpezial = AliEmcalCorrectionTask::AddTaskEmcalCorrectionTask(taskNameSpecial);
//		 correctionTaskSpezial->SelectCollisionCandidates(kPhysSel);
//		 if(intMCrunning==0)
//             correctionTaskSpezial->SetUserConfigurationFilename("/Users/hanseopark/alice/work/Local_Analysis/AliEmcalCorrectionConfiguration_park.yaml");
//		 else
//             correctionTaskSpezial->SetUserConfigurationFilename("/Users/hanseopark/alice/work/Local_Analysis/AliEmcalCorrectionConfiguration_park.yaml");
//		 correctionTaskSpezial->Initialize();

//         #if !defined (__CINT__) || defined (__CLING__)
//            AliAnalysisTask *correctionTask= reinterpret_cast<AliAnalysisTask*>(gInterpreter->ExecuteMacro("$ALICE_PHYSICS/PWG/EMCAL/macros/AddTaskEmcalCorrectionTask.C ( \"\" ) "));
//         #else
//            gROOT->LoadMacro("$ALICE_PHYSICS/PWG/EMCAL/macros/AddTaskEmcalCorrectionTask.C");
//            AliAnalysisTask *correctionTask = AddTaskEmcalCorrectionTask("");
//         #endif
//         correctionTask->SelectCollisionCandidates(kPhysSel);
//		 correctionTask->SetUserConfigurationFilename("/Users/hanseopark/alice/AliPhysics/PWG/EMCAL/config/AliEmcalCorrectionConfiguration.yaml");

     } else{
         AliEmcalCorrectionTask * correctionTask = AliEmcalCorrectionTask::AddTaskEmcalCorrectionTask("");
         #if !defined (__CINT__) || defined (__CLING__)
             AliAnalysisTask *tenderTask=reinterpret_cast<AliAnalysisTask*>(
             gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/PWG/EMCAL/macros/AddTaskEMCALTender.C( kTRUE, kTRUE, kTRUE, kTRUE, kFALSE, kFALSE, kTRUE, kFALSE, kFALSE, kTRUE, AliEMCALRecoUtils::kNoCorrection, kTRUE, 0.5, 0.1, AliEMCALRecParam::kClusterizerv2, kFALSE, kFALSE, -500e9, 500e9, 1e6, \"%s\", kFALSE )",tenderPassData.Data())));
         #else
             gROOT->LoadMacro("$ALICE_PHYSICS/PWG/EMCAL/macros/AddTaskEMCALTender.C");
             AliAnalysisTask *emcalTender            = AddTaskEMCALTender(  kTRUE, kTRUE, kTRUE, kTRUE, kFALSE, kFALSE, kTRUE, kFALSE, kFALSE, kTRUE, AliEMCALRecoUtils::kNoCorrection, kTRUE, 0.5, 0.1, AliEMCALRecParam::kClusterizerv2, kFALSE, kFALSE, -500e9, 500e9, 1e6,tenderPassData,kFALSE   );
         #endif
     }
    
    // -----------------------------------------
    //               PHOS TENDER
    // -----------------------------------------
//    TString customPHOSBadMap = "$ALICE_PHYSICS/PWGGA/GammaConv/macros/data/PHOSBadChannelMapRunDepRun2.root";
//    Int_t intCustomPHOSBadMap = 1;
//    #if !defined (__CINT__) || defined (__CLING__)
//        AliAnalysisTask *phosTendTask=reinterpret_cast<AliAnalysisTask*>(
//        gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_PHOSTender_PCMconfig.C( \"PHOSTenderTask\", \"PHOSTender\", \"%s\", 1, kFALSE, %i, \"%s\", \"Run2\",kTRUE )",runPeriod.Data(),intCustomPHOSBadMap,customPHOSBadMap.Data())));
//    #else
//        gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_PHOSTender_PCMconfig.C");
//        AliAnalysisTask *phosTenderTask  = AddTask_PHOSTender_PCMconfig("PHOSTenderTask", "PHOSTender", runPeriod.Data(), 1, kFALSE,0, "",kFALSE);
//    #endif

    // -----------------------------------------
    //               V0 READER
    // -----------------------------------------
    #if !defined (__CINT__) || defined (__CLING__)
      AddTask_V0Reader(runPeriod.Data(),kFALSE,kFALSE,kTRUE,collsys,aodConversionCutnumber.Data(), "06000008400100001000000000");
    #else
      gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_V0Reader.C");
      AliAnalysisTask *taskV0Reader                = AddTask_V0Reader(runPeriod.Data(),kFALSE,kFALSE,kTRUE,collsys,aodConversionCutnumber.Data(), "06000008400100001000000000");
    #endif


    // -----------------------------------------
    //               TRIGGER MAKER
    // -----------------------------------------
    // #if !defined (__CINT__) || defined (__CLING__)
    //     AliEmcalTriggerMakerTask *triggerTask=reinterpret_cast<AliEmcalTriggerMakerTask*>(
    //     gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/PWG/EMCAL/macros/AddTaskEmcalTriggerMakerNew.C()")));
    //     // triggerTask->SelectCollisionCandidates(AliVEvent::kINT7 | AliVEvent::kEMC7 | AliVEvent::kEMCEJE | AliVEvent::kEMCEGA);
    //     //
    //     //
    //     //         TF1 *meanmodel = new TF1("meanmodel", "pol1", 0., 1000.);
    //     //         meanmodel->SetParameter(0, -0.0206247);
    //     //         meanmodel->SetParameter(1, 0.966160);
    //     //         // Power law smearing
    //     //         TF1 *widthmodel = new TF1("widthmodel", "[0] * TMath::Power(x, [1]) + [2]", 0., 1000.);
    //     //         widthmodel->SetParameter(0, 0.0273139);
    //     //         widthmodel->SetParameter(1, 1.36187);
    //     //         widthmodel->SetParameter(2, 0.0736051);
    //     //         triggerTask->GetTriggerMaker()->SetSmearModel(meanmodel, widthmodel);
    //     if(intMCrunning == 0){
    //       triggerTask->SetMaskedFastorOADBContainer("/Users/joshua/PCG_Software/LocalTrainTest/MaskedFastors/MaskedFastors.root");
    //       triggerTask->SelectCollisionCandidates(AliVEvent::kEMCEGA);
    //
    //       // TF1 *meanmodel = new TF1("meanmodel", "pol1", 0., 1000.);
    //       // meanmodel->SetParameter(0, -0.0206247);
    //       // meanmodel->SetParameter(1, 0.966160);
    //       // // Power law smearing
    //       // TF1 *widthmodel = new TF1("widthmodel", "[0] * TMath::Power(x, [1]) + [2]", 0., 1000.);
    //       // widthmodel->SetParameter(0, 0.0273139);
    //       // widthmodel->SetParameter(1, 1.36187);
    //       // widthmodel->SetParameter(2, 0.0736051);
    //       // triggerTask->GetTriggerMaker()->SetSmearModel(meanmodel, widthmodel);
    //       // //__R_ADDTASK__->GetTriggerMaker()->SetAcceptCellTimeRange(-20e-9, 15e-9);
    //       // triggerTask->GetTriggerMaker()->SetAcceptCellTimeRange(-30e-9, 30e-9);
    //       triggerTask->SetApplyTRUMaskingToFEE(true);
    //     }
    //
    //
    // #else
    //     gROOT->LoadMacro("$ALICE_PHYSICS/PWG/EMCAL/macros/AddTaskTriggerMakerNew.C");
    //     AliEmcalTriggerMakerTask *triggermaker = AddTaskTriggerMakerNew();
    // #endif


    // -----------------------------------------
    //               TRIGGER SELECTION
    // -----------------------------------------
    // #if !defined (__CINT__) || defined (__CLING__)
    //     PWG::EMCAL::AliAnalysisTaskEmcalTriggerSelection* trgseltask=reinterpret_cast<PWG::EMCAL::AliAnalysisTaskEmcalTriggerSelection*>(
    //     gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/PWG/EMCAL/macros/AddEmcalTriggerSelectionTask.C()")));
    //     trgseltask->SetGlobalDecisionContainerName("EmcalTriggerDecision");
    //     if(intMCrunning) trgseltask->AutoConfigure("lhc18f5");
    //     else trgseltask->AutoConfigure("lhc18m");
    //
    //
    // #else
    //     gROOT->LoadMacro("$ALICE_PHYSICS/PWG/EMCAL/macros/AddTaskTriggerMakerNew.C");
    //     AliEmcalTriggerMakerTask *triggermaker = AddTaskTriggerMakerNew();
    // #endif



    // -----------------------------------------
    //               MC track selector
    // -----------------------------------------
//    #if !defined (__CINT__) || defined (__CLING__)
//        AddTaskMCTrackSelector("mcparticlesSelected", kFALSE, kFALSE, -1, kFALSE);
//    #else
//        gROOT->LoadMacro("$ALICE_PHYSICS/PWG/EMCAL/macros/AddTaskMCTrackSelector.C");
//        AliAnalysisTask *taskTrackSelector                = AddTaskMCTrackSelector("mcparticlesSelected", kFALSE, kFALSE, -1, kFALSE);
//    #endif




    // -----------------------------------------
    //               Jet
    // -----------------------------------------
    // PWGJE/EMCALJetTasks/macros

	////////////////////
	// EMCalJetFinder //
	////////////////////

    #if !defined (__CINT__) || defined (__CLING__)
        // AliEmcalJetTask* EMCalJet=reinterpret_cast<AliEmcalJetTask*>(
        // AliEmcalJetTask* EMCJetTask=reinterpret_cast<AliEmcalJetTask*>(
        //                      gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskEmcalJet.C(\"usedefault\", \"\", AliJetContainer::antikt_algorithm, 0.4, AliJetContainer::kChargedJet, 0.15, 0., 0.01, AliJetContainer::pt_scheme, \"Jet\", 10., kFALSE, kFALSE))")));
        AddTaskEmcalJet("usedefault", "", AliJetContainer::antikt_algorithm, 0.4, AliJetContainer::kChargedJet, 0.15, 0., 0.01, AliJetContainer::pt_scheme, "Jet", 10., kFALSE, kFALSE);
        // AddTaskEmcalJet("usedefault", "", AliJetContainer::antikt_algorithm, 0.4, AliJetContainer::kChargedJet, 0.15, 0., 0.01, AliJetContainer::pt_scheme, "Jet", 10., kFALSE, kFALSE);
    #else
        gROOT->LoadMacro("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskEmcalJet.C");
        // AddTaskEmcalJet("usedefault", "", AliJetContainer::antikt_algorithm, 0.4, AliJetContainer::kChargedJet, 0.15, 0., 0.01, AliJetContainer::pt_scheme, "Jet", 10., kFALSE, kFALSE);
        AliAnalysisTask *EMCalJet                = AddTaskEmcalJet("usedefault", "", AliJetContainer::antikt_algorithm, 0.4, AliJetContainer::kChargedJet, 0.15, 0., 0.01, AliJetContainer::pt_scheme, "Jet", 10, kFALSE, kFALSE);
    #endif
        // EMCalJet->SelectCollisionCandidates(AliVEvent::kAny);


    if(intMCrunning > 0){
    #if !defined (__CINT__) || defined (__CLING__)
          AddTaskEmcalJet("mcparticles", "", AliJetContainer::antikt_algorithm, 0.4, AliJetContainer::kChargedJet, 0.15, 0., 0.01, AliJetContainer::pt_scheme, "Jet", 10., kFALSE, kFALSE);
          //AddTaskEmcalJet("mcparticles", "", AliJetContainer::antikt_algorithm, 0.4, AliJetContainer::kChargedJet, 0.15, 0., 0.01, AliJetContainer::pt_scheme, "Jet", 5., kFALSE, kFALSE);
    #else
          gROOT->LoadMacro("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskEmcalJet.C");
          // AliAnalysisTask *EMCalJet                = AddTaskEmcalJet("mcparticles", "", AliJetContainer::antikt_algorithm, 0.2, AliJetContainer::kFullJet, 0., 0., 0.01, AliJetContainer::E_scheme, "Jet", 0.1, kFALSE, kFALSE);
          AliAnalysisTask *EMCalJet                = AddTaskEmcalJet("mcparticles", "", AliJetContainer::antikt_algorithm, 0.4, AliJetContainer::kChargedJetd, 0.15, 0., 0.01, AliJetContainer::pt_scheme, "Jet", 10, kFALSE, kFALSE);
    #endif
    }

	////////////////////
	// EmcalJetReader //
	////////////////////

    #if !defined (__CINT__) || defined (__CLING__)
        AliAnalysisTaskConvJet* ConvJetTask=reinterpret_cast<AliAnalysisTaskConvJet*>(
                          gInterpreter->ExecuteMacro(Form("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_GammaConvJet.C(\"usedefault\", \"usedefault\", \"usedefault\", \"Jet\")")));
        // AddTask_GammaConvJet("usedefault", "usedefault", "usedefault", "Jet");

        AliJetContainer* jetContRec = ConvJetTask->AddJetContainer("Jet_AKTChargedR040_tracks_pT0150_pt_scheme", AliEmcalJet::kTPCfid, 0.4);
        // ConvJetTask->SelectCollisionCandidates(AliVEvent::kINT7);
        ConvJetTask->SelectCollisionCandidates(AliVEvent::kAny);
        // ConvJetTask->SelectCollisionCandidates(AliVEvent::kEMCEGA);

        AliParticleContainer *trackCont  =  ConvJetTask->AddParticleContainer("tracks");
        AliClusterContainer *clusterCont =  ConvJetTask->AddClusterContainer("caloClusters");

        jetContRec->ConnectParticleContainer(trackCont);
        jetContRec->ConnectClusterContainer(clusterCont);

        if(intMCrunning > 0){
            AliJetContainer* jetContMC = ConvJetTask->AddJetContainer("Jet_AKTChargedR040_mcparticles_pT0150_pt_scheme", "calo", 0.4);
            // return;
            jetContMC->ConnectParticleContainer(trackCont);
            jetContMC->ConnectClusterContainer(clusterCont);
        }
    #else
        //gROOT->LoadMacro("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTask_GammaConvJet.C");
        gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_GammaConvJet.C");
        AliAnalysisTask *EMCalJetReader                = AddTask_GammaConvJet("usedefault", "usedefault", "usedefault", "Jet");
    #endif



//
//
//
//
//
//        // // -----------------------------------------
//        // //               Jet
//        // // -----------------------------------------
//        // #if !defined (__CINT__) || defined (__CLING__)
//        //     AddTask_GammaOutlierRemoval();
//        // #else
//        //     gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_GammaOutlierRemoval.C");
//        //     AliAnalysisTask *EMCalJet2                = AddTask_GammaOutlierRemoval();
//        // #endif
//
//
//
//
//
//    // -----------------------------------------
//    //                SKIMMING
//    // -----------------------------------------
//    if(runMode.Contains("S")){
//        #if !defined (__CINT__) || defined (__CLING__)
//            // if(dataType.Contains("AOD"))
//            //     AddTaskAodSkim(-1, 0.9, Form("GammaConv_%s_gamma",aodConversionCutnumber.Data()), AliVEvent::kAny, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE,"skim");
//            // if(dataType.Contains("ESD"))
//            //     AddTaskEsdSkim(kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE, "Tracks", "AliSkimmedESD.root");
//        #else
//        #endif
//    }
//
//    // -----------------------------------------
//    //                QA Wagons
//    // -----------------------------------------
//    if(runMode.Contains("QA")){
//        #if !defined (__CINT__) || defined (__CLING__)
//            // AddTask_PhotonQA("06000008400100001000000000","00010113","090000092663743800000000",intMCrunning,collsys);
//            // if(!dataType.Contains("AOD"))
//            // AddTask_ConvCaloTree("10000008400100001500000000", "00010113", "", "411790009f030000000" , "",  1, 1, intMCrunning, 0, kTRUE, 1, "", kFALSE, kFALSE, runPeriod.Data(), "", 0, 0, 0.5, 0, 1, 1, kTRUE, "3.", kFALSE);
//            // AddTask_ConvCaloTree(intMCrunning, 0, "06000008400100001000000000", "00010103", "4117900060e30220000", "" , "00200009227302008254404000",  1, runPeriod.Data(), "", 0, 2, 1, 0.5, kTRUE, "3."); //24466190sa01cc00000
//            // AddTask_ConvCaloTree(intMCrunning, 0, "06000008400100001000000000", "00010113", "411799909f030000000", "" , "00200009227302008254404000",  1, runPeriod.Data(), "", 0, 0, 1, 0.5, kTRUE, "3.", true, 2, false); //24466190sa01cc00000
//
//            // AddTask_ClusterQA("06000008400100001000000000", "00010113", "4117900050020000000", 1, 1, intMCrunning, 0, kTRUE, 1, "", kFALSE, kFALSE, runPeriod.Data(), "", 0, kTRUE, kTRUE, 0.6/*minclsE*/, 10./*cone*/, 0.3, kTRUE, kTRUE,0, kTRUE, "2.5", kFALSE);
//            // AddTask_ClusterQA("06000008400100001000000000", "00010113", "4117900050020000000", 1, 1, intMCrunning, 0, kTRUE, 1, "", kFALSE, kFALSE, runPeriod.Data(), "", 0, kTRUE, kTRUE, 0.6/*minclsE*/, 10./*cone*/, 0.3, kTRUE, kTRUE,0, kTRUE, "2.5", kFALSE);
//            // AddTask_ClusterQA("06000008400100001000000000", "00010113", "4117900070ei0220000", 1, 1, intMCrunning, 0, kTRUE, 1, "", kFALSE, kFALSE, runPeriod.Data(), "", 0, 0, 0.5, 0, 1, 1, kTRUE, "3.", kFALSE);
//            // AddTask_ClusterQA("06000008400100001000000000", "00010113", "4117900050020000000", 1, 1, intMCrunning, 0, kTRUE, 1, "", kFALSE, kFALSE, runPeriod.Data(), "S500A100", 0, kTRUE, kTRUE, 0.6/*minclsE*/, 10./*cone*/, 0.3, kTRUE, kTRUE,0, kTRUE, "2.5", kFALSE);
//        #else
//        #endif
//    }
//
//
//    // -----------------------------------------
//    //               GammaConv PCM
//    // -----------------------------------------
//    if(runMode.Contains("P")){
//        #if !defined (__CINT__) || defined (__CLING__)
//            if(collsys==0){
//                // AddTask_GammaConvV1_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  2, 2, kFALSE, kFALSE, kFALSE, kFALSE,3, 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE, "",0,kFALSE,kFALSE,kFALSE,1,0,0,"3010");
//                // AddTask_GammaConvV1_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  2, 2, kFALSE, kFALSE, kFALSE, kFALSE,3, 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE, "",0,kFALSE,kFALSE,kFALSE,1,0,0,"3011");
//                // AddTask_GammaConvV1_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  2, 2, kFALSE, kFALSE, kFALSE, kFALSE,3, 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE, "",0,kFALSE,kFALSE,kFALSE,1,0,0,"3012");
//            }
//            // if(collsys==1){
//            //     AddTask_GammaConvV1_PbPb(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),runPeriod.Data(),  2, 2, kFALSE, kFALSE, kFALSE, kFALSE,3, 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, kFALSE, "",kFALSE, "",0,kFALSE,0, kFALSE,kFALSE,kFALSE,"1042");
//            // }
//            // if(collsys==2){
//            //     AddTask_GammaConvV1_pPb(0, intMCrunning, "06000008400100001000000000",runPeriod.Data(),  2, 2, kFALSE, kFALSE, kFALSE, kFALSE,3, 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE,"",0,kFALSE,kFALSE,"300");
//            // }
//        #else
//            // if(collsys==0){
//            //     gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_GammaConvV1_pp.C");
//            //     AliAnalysisTask *taskPCMpp1 = AddTask_GammaConvV1_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  2, 2, kFALSE, kFALSE, kFALSE, kFALSE,3, 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE, "",0,kFALSE,kFALSE,kFALSE,1,0,0,"1020");
//            //     AliAnalysisTask *taskPCMpp2 = AddTask_GammaConvV1_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  2, 2, kFALSE, kFALSE, kFALSE, kFALSE,3, 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE, "",0,kFALSE,kFALSE,kFALSE,1,0,0,"400");
//            // }
//            // if(collsys==1){
//            //     gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_GammaConvV1_PbPb.C");
//            //     AliAnalysisTask *taskPCMPbPb = AddTask_GammaConvV1_PbPb(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),runPeriod.Data(),  2, 2, kFALSE, kFALSE, kFALSE, kFALSE,3, 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, kFALSE, "",kFALSE, "",0,kFALSE,0, kFALSE,kFALSE,kFALSE,"1042");
//            // }
//            // if(collsys==2){
//            //     gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_GammaConvV1_pPb.C");
//            //     AliAnalysisTask *taskPCMpPb = AddTask_GammaConvV1_pPb(0, intMCrunning, "06000008400100001000000000",runPeriod.Data(),  2, 2, kFALSE, kFALSE, kFALSE, kFALSE,3, 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE,"",0,kFALSE,kFALSE,"300");
//            // }
//        #endif
//    }
//
//
    // -----------------------------------------
    //          GammaCalo EMC/DMC/PHOS
    // -----------------------------------------
    if(runMode.Contains("C")){
        #if !defined (__CINT__) || defined (__CLING__)
            if(collsys==0){
	            AddTask_GammaCalo_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(), 1, 1, 3, 0, kFALSE, 0, kFALSE,"3.", 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, runPeriod.Data(),kFALSE, runPeriod.Data(),kTRUE,!taskNameSpecial.CompareTo("") ? "950" : Form("950_CF%s",taskNameSpecial.Data()));
                // AddTask_GammaCalo_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(), 1, 1, 3, 0, kFALSE, 0, kFALSE,"3.", 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, runPeriod.Data(),kFALSE, runPeriod.Data(),kTRUE,!taskNameSpecial.CompareTo("") ? "2062" : Form("2062_CF%s",taskNameSpecial.Data()));
            }
            if(collsys==2){
              //AddTask_GammaConvNeutralMesonPiPlPiMiNeutralMeson_CaloMode_pPb(0,intMCrunning,"06000008400100001000000000",0, 0, 1,kFALSE,kFALSE,"3.",kFALSE,"",-1,runPeriod.Data(),0,1500, !taskNameSpecial.CompareTo("") ? "1010" : Form("1010_CF%s",taskNameSpecial.Data()));

                // AddTask_GammaCalo_PbPb(0, intMCrunning, "10000008400100001500000000", runPeriod.Data(), 1, 1, 1, kFALSE, kFALSE, kFALSE, kFALSE,"3.", 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", kFALSE, 0, "",kFALSE, "",kTRUE,kFALSE,!taskNameSpecial.CompareTo("") ? "264" : Form("264_CF%s",taskNameSpecial.Data()));
            }
            // if(collsys==2){
            //     AddTask_GammaCalo_pPb(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(), runPeriod.Data(),  1, 1, 0, kFALSE, kFALSE, kFALSE, kFALSE,3., 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE, "",kTRUE, !taskNameSpecial.CompareTo("") ? "200" : Form("200_CF%s",taskNameSpecial.Data()));
            // }
        #else
            if(collsys==0){
                gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_GammaCalo_pp.C");
                //AliAnalysisTask *taskPCMppCalo1 = AddTask_GammaCalo_pp(0, intMCrunning, "06000008d00100001100000000", runPeriod.Data(), 0, 0, 0, kFALSE, kFALSE, kTRUE, kFALSE,3., 0,"", 0, runPeriod.Data(),kFALSE, runPeriod.Data(),kTRUE,!taskNameSpecial.CompareTo("") ? "950" : Form("950_CF%s",taskNameSpecial.Data()));
                // AliAnalysisTask *taskPCMppCalo2 = AddTask_GammaCalo_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(), 1, 2, 5, kFALSE, kFALSE, kFALSE, kFALSE,3., 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE, "",kTRUE,!taskNameSpecial.CompareTo("") ? "502" : Form("541_CF%s",taskNameSpecial.Data()));
                // AliAnalysisTask *taskPCMppCalo = AddTask_ConvCaloCalibration_CaloMode_pp(0, 0, intMCrunning, "06000008400100001000000000", runPeriod.Data(), 0, 0, 0, 0, kFALSE, kFALSE, kFALSE,3., 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE, "",kTRUE,!taskNameSpecial.CompareTo("") ? "200" : Form("200_CF%s",taskNameSpecial.Data()));
            }
            // if(collsys==1){
            //     gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_GammaCalo_PbPb.C");
            //     AliAnalysisTask *taskPCMPbPb = AddTask_GammaCalo_PbPb(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),runPeriod.Data(),  1, 2, 5, kFALSE, kFALSE, kFALSE, kFALSE,3., 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", kFALSE, 0, "",kFALSE, "",kTRUE,kFALSE,!taskNameSpecial.CompareTo("") ? "218" : Form("218_CF%s",taskNameSpecial.Data()));
            // }
            // if(collsys==2){
            //     gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_GammaCalo_pPb.C");
            //     AliAnalysisTask *taskPCMpPb = AddTask_GammaCalo_pPb(0, intMCrunning, "06000008400100001000000000", "",  1, 1, 1, kFALSE, kFALSE, kTRUE, kFALSE,1., 1,"", 0, "",kFALSE, "",kTRUE, !taskNameSpecial.CompareTo("") ? "502" : Form("502_CF%s",taskNameSpecial.Data()));
            //
            // }
        #endif
    }


//
//    // -----------------------------------------
//    //      GammaConvCalo PCM-(EMC/DMC/PHOS)
//    // -----------------------------------------
//    if(runMode.Contains("H")){
//        #if !defined (__CINT__) || defined (__CLING__)
//            if(collsys==0){
//              AddTask_GammaConvCalo_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  0, 0, 0, 0, kFALSE, 0, kFALSE,"3.", 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE, 0, kFALSE, runPeriod.Data(),kTRUE,kFALSE,kFALSE,1,0,0,kTRUE, !taskNameSpecial.CompareTo("") ? "2012" : Form("2012_CF%s",taskNameSpecial2.Data()));
//              // AddTask_GammaConvNeutralMesonPiPlPiMiNeutralMeson_MixedMode_pp(0,  intMCrunning, "06000008400100001000000000", 0,  0, 0, 0, kFALSE, "3.", "", 0, 0, "", -1, runPeriod.Data(), 0, 1500, kFALSE, kTRUE, "986");
//            }
////     //         if(collsys==1){
////     //             AddTask_GammaConvCalo_PbPb(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),runPeriod.Data(),  1, 2, 5, 0, kFALSE, kFALSE, kFALSE,3., 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", kFALSE, "",kFALSE, "",kFALSE,kTRUE, kFALSE, kTRUE, 0, kTRUE, !taskNameSpecial.CompareTo("") ? "300" : Form("300_CF%s",taskNameSpecial.Data()));
////     //         }
////     //         if(collsys==2){
////     //             AddTask_GammaConvCalo_pPb(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  1, 2, 5, 0, kFALSE, kFALSE, kFALSE,3., 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE, "",kTRUE,kFALSE, !taskNameSpecial.CompareTo("") ? "508" : Form("508_CF%s",taskNameSpecial.Data()));
////     //             AddTask_GammaConvCalo_pPb(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  1, 2, 5, 0, kFALSE, kFALSE, kFALSE,3., 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE, "",kTRUE,kFALSE, !taskNameSpecial.CompareTo("") ? "700" : Form("700_CF%s",taskNameSpecial.Data()));
////     //         }
//        #else
//            if(collsys==0){
//                AddTask_GammaConvCaloCalibration_MixedMode(/*0, 0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  1, 2, 5, 0, kFALSE, kFALSE, kFALSE,3., 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE, "",kTRUE,kFALSE,kFALSE,1.,0.,0.,kTRUE, !taskNameSpecial.CompareTo("") ? "200" : Form("200_CF%s",taskNameSpecial.Data())*/);
//            }
////     //         if(collsys==0){
////     //             gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_GammaConvCalo_pp.C");
////     //             AliAnalysisTask *taskPCMEMCpp = AddTask_GammaConvCalo_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(), runPeriod.Data(),  1, 2, 5, 0, kFALSE, kFALSE, kFALSE,3., 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", kFALSE, "",kFALSE, "",kTRUE,kFALSE,kFALSE,1.,0.,0.,kTRUE, !taskNameSpecial.CompareTo("") ? "106" : Form("106_CF%s",taskNameSpecial.Data()));
////     //         }
////     //         if(collsys==1){
////     //             gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_GammaConvCalo_PbPb.C");
////     //             AliAnalysisTask *taskPCMEMCPbPb = AddTask_GammaConvCalo_PbPb(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),runPeriod.Data(),  1, 2, 5, 0, kFALSE, kFALSE, kFALSE,3., 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", kFALSE, "",kFALSE, "",kFALSE,kTRUE, kFALSE, kTRUE, 0, kTRUE,!taskNameSpecial.CompareTo("") ? "300" : Form("300_CF%s",taskNameSpecial.Data()));
////     //         }
////     //         if(collsys==2){
////     //             gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_GammaConvCalo_pPb.C");
////     //             AliAnalysisTask *taskPCMEMCpPb1 = AddTask_GammaConvCalo_pPb(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  1, 2, 5, 0, kFALSE, kFALSE, kFALSE,3., 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE, "",kTRUE,kFALSE,!taskNameSpecial.CompareTo("") ? "508" : Form("508_CF%s",taskNameSpecial.Data()));
////     //             AliAnalysisTask *taskPCMEMCpPb2 = AddTask_GammaConvCalo_pPb(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  1, 2, 5, 0, kFALSE, kFALSE, kFALSE,3., 0,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, "",kFALSE, "",kTRUE,kFALSE,!taskNameSpecial.CompareTo("") ? "700" : Form("700_CF%s",taskNameSpecial.Data()));
////     //         }
//        #endif
//    }
////
////
////     // -----------------------------------------
////     //              CaloMerged mEMC
////     // -----------------------------------------
//    if(runMode.Contains("M")){
//        #if !defined (__CINT__) || defined (__CLING__)
//            if(collsys==0){
//                // AddTask_GammaCaloMerged_pp(0, intMCrunning, "10000008400100001500000000", runPeriod.Data(),  2, 1, 3, 0, 0, kFALSE, "3.","", kFALSE, "", 1, kTRUE, kFALSE, 1.0, kFALSE, kFALSE, -1, !taskNameSpecial.CompareTo("") ? "1523" : Form("1523_CF%s",""));
//                // AddTask_GammaCaloMerged_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  1, 1, 1, 0, 0, kFALSE, "3.","", kFALSE, "", 1, kTRUE, kFALSE, 1.0, kFALSE, kFALSE, -1, -1, false, "", "1513");
//                // AddTask_GammaCaloMerged_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  1, 1, 1, 0, 0, kFALSE, "3.","", kFALSE, "", 1, kTRUE, kFALSE, 1.0, kFALSE, kFALSE, -1, -1, false, "", "1700");
//                // AddTask_GammaCaloMerged_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  1, 1, 1, 0, 0, kFALSE, "3.","", kFALSE, "", 1, kTRUE, kFALSE, 1.0, kFALSE, kFALSE, -1, -1, false, "", "1702");
//                // AddTask_GammaCaloMerged_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  1, 1, 1, 0, 0, kFALSE, "3.","", kFALSE, "", 1, kTRUE, kFALSE, 1.0, kFALSE, kFALSE, -1, -1, false, "", "1704");
//                // AddTask_GammaCaloMerged_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  1, 1, 1, 0, 0, kFALSE, "3.","", kFALSE, "", 1, kTRUE, kFALSE, 1.0, kFALSE, kFALSE, -1, -1, false, "", "1706");
//                // AddTask_GammaCaloMerged_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  1, 1, 1, 0, 0, kFALSE, "3.","", kFALSE, "", 1, kTRUE, kFALSE, 1.0, kFALSE, kFALSE, -1, -1, false, "", "1708");
//                // AddTask_GammaCaloMerged_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  1, 1, 1, 0, 0, kFALSE, "3.","", kFALSE, "", 1, kTRUE, kFALSE, 1.0, kFALSE, kFALSE, -1, -1, false, "", "1900");
//                // AddTask_GammaCaloMerged_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  1, 1, 1, 0, 0, kFALSE, "3.","", kFALSE, "", 1, kTRUE, kFALSE, 1.0, kFALSE, kFALSE, -1, -1, false, "", "1901");
//                // AddTask_GammaCaloMerged_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  1, 1, 1, 0, 0, kFALSE, "3.","", kFALSE, "", 1, kTRUE, kFALSE, 1.0, kFALSE, kFALSE, -1, -1, false, "", "1902");
//                // AddTask_GammaCaloMerged_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(),  1, 1, 1, 0, 0, kFALSE, "3.","", kFALSE, "", 1, kTRUE, kFALSE, 1.0, kFALSE, kFALSE, -1, -1, false, "", "1903");
//
//            }
//    //         if(collsys==1){
//    //             AddTask_GammaCaloMerged_PbPb(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(), runPeriod.Data(),  1, 2, 5, 0, kFALSE, kFALSE, 3.,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", kFALSE, 0, 0,1,kTRUE, kFALSE,1.0,kFALSE,kFALSE, !taskNameSpecial.CompareTo("") ? "201" : Form("201_CF%s",taskNameSpecial.Data()));
//    //         }
//    //         if(collsys==2){
//    //             AddTask_GammaCaloMerged_pPb(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(), runPeriod.Data(),  1, 2, 5, 0, kFALSE, kFALSE, 3.,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, 1,kTRUE, kFALSE,1.0,kFALSE,kFALSE, !taskNameSpecial.CompareTo("") ? "200" : Form("200_CF%s",taskNameSpecial.Data()));
//    //         }
//        #else
//    //         if(collsys==0){
//    //             gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_GammaCaloMerged_pp.C");
//    //             AliAnalysisTask *taskPCMpp = AddTask_GammaCaloMerged_pp(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(), runPeriod.Data(),  1, 2, 5, 0, kFALSE, kFALSE, 3.,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", kFALSE, 1,kTRUE, kFALSE,1.0,kFALSE,kFALSE,!taskNameSpecial.CompareTo("") ? "137" : Form("137_CF%s",taskNameSpecial.Data()));
//    //         }
//    //         if(collsys==1){
//    //             gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_GammaCaloMerged_PbPb.C");
//    //             AliAnalysisTask *taskPCMPbPb = AddTask_GammaCaloMerged_PbPb(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(), runPeriod.Data(),  1, 2, 5, 0, kFALSE, kFALSE, 3.,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", kFALSE, 0, 0,1,kTRUE, kFALSE,1.0,kFALSE,kFALSE,!taskNameSpecial.CompareTo("") ? "201" : Form("201_CF%s",taskNameSpecial.Data()));
//    //         }
//    //         if(collsys==2){
//    //             gROOT->LoadMacro("$ALICE_PHYSICS/PWGGA/GammaConv/macros/AddTask_GammaCaloMerged_pPb.C");
//    //             AliAnalysisTask *taskPCMpPb = AddTask_GammaCaloMerged_pPb(0, intMCrunning, "06000008400100001000000000", runPeriod.Data(), runPeriod.Data(),  1, 2, 5, 0, kFALSE, kFALSE, 3.,"FMUW:fileNameMultWeights;FEPC:fileNamedEdxPostCalib", 0, 1,kTRUE, kFALSE,1.0,kFALSE,kFALSE,!taskNameSpecial.CompareTo("") ? "200" : Form("200_CF%s",taskNameSpecial.Data()));
//    //         }
//        #endif
//    }

////////// FOR MY TASK /////////
//#if !defined (__CINT__) || defined (__CLING__)
//    
//    gInterpreter->LoadMacro("AliAnalysisTaskMyTask.cxx++g");
//    AliAnalysisTaskMyTask *task = reinterpret_cast<AliAnalysisTaskMyTask*>(gInterpreter->ExecuteMacro("AddMyTask.C"));
//#else
//    gROOT->LoadMacro("AliAnalysisTaskMyTask.cxx++g");
//    gROOT->LoadMacro("AddMyTask.C");
//    AliAnalysisTaskMyTask *task = AddMyTask();
//#endif
//

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
