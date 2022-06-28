#!bin/bash

#INITFILE=${1}
INITFILE=/Users/hanseopark/alice/work/Local_Analysis/bJet/pp_13TeV/MC/LHC17k4/CAOD/AnalysisResults.root
#OUTFILE=${2}
OUTFILE=temp.root

cmd="root -b -l 'src/AnalysisJets.C(\"$INITFILE\", \"$OUTFILE\")'"
echo $cmd
eval $cmd

