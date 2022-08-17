# Config
DATA="/Users/hanseopark/alice/work/Data/RUN3/Data"
MC="/Users/hanseopark/alice/work/Data/RUN3/MC"

echo "Jet analysis"
#o2-analysistutorial-jet-analysis --aod-file ${DATA}/AO2D.root | o2-analysis-je-jet-finder -b | o2-analysis-event-selection -b | o2-analysis-timestamp -b | o2-analysis-trackselection -b | o2-analysis-trackextension -b | o2-analysis-je-jet-finder-hf -b
o2-analysis-hf-d0-candidate-selector --aod-file ${DATA}/AO2D.root | o2-analysis-pid-tof-full -b | o2-analysis-timestamp -b | o2-analysis-pid-tof-base | o2-analysis-pid-tpc-full | o2-analysis-multiplicity-table -b | o2-analysis-hf-track-index-skims-creator -b | o2-analysis-trackselection -b | o2-analysis-trackextension -b | o2-analysis-hf-candidate-creator-2prong -b | o2-analysis-je-jet-finder-hf -b
#o2-analysis-je-jet-finder --aod-file ${DATA}/AO2D.root
