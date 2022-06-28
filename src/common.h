
// var
Int_t event = 0;
Float_t jetpt =0.;

TFile *fin, *fout;
TTree *trjets;

Int_t Test(){
    std::cout<< "TEST"<< std::endl;
    return 0;
}

Int_t LoadData(TString jetsfile){

    if (gSystem->AccessPathName(jetsfile.Data())){
        std::cout << "Input file not found!" << std::endl;
        return 0;
    }
    
    fin = TFile::Open(jetsfile, "READ");
    trjets = (TTree*) fin->Get("JetTree_AliAnalysisTaskJetExtractor_Jet_AKTChargedR040_tracks_pT0150_pt_scheme_bJets");
    trjets->Show();
//    trjets->SetBranchStatus("*", 0);
//    trjets->SetBranchStatus("Event", 1);
    trjets->SetBranchStatus("Jet_Pt", 1);
//    trjets->SetBranchAddress("Event", &event);
    trjets->SetBranchAddress("Jet_Pt", &jetpt);
    
    return 1;
}

