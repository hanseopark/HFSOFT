#include "common.h"
#include "AnalysisJets.h"

using namespace std;

void AnalysisJets(TString jetsfile, TString outfile){

	cout << "Using file: " << jetsfile << endl; 
	if (!LoadData(jetsfile)) return;
	InitOutput(outfile);

	//Double_t Nevent = trjets->GetEntriesFast();
	int Nevent = trjets->GetEntries();
	cout << "Number of event: " << Nevent << endl;
	for (Int_t iev=0; iev<Nevent; iev++){
		trjets->GetEntry(iev);
		cout << "Jet Pt: "<< jetpt << endl;
		hJetPt->Fill(jetpt);
	}
	mc(0);
	hJetPt->Draw();
	fout->Write();
}

void InitOutput(TString outfile){
	fout = TFile::Open(outfile.Data(), "RECREATE");
	hJetPt = new TH1D("hJetPt", "hJetPt", 20,0,20);
}
