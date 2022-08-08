# Config
DATA="/Users/hanseopark/alice/work/Data/RUN3/Data"
MC="/Users/hanseopark/alice/work/Data/RUN3/MC"

# Event Selection
#o2-analysis-event-selection --aod-file ${DATA}/AO2D.root | o2-analysis-timestamp -b
#o2-analysis-event-selection --aod-file ${MC}/AO2D.root -b --isMC 1 | o2-analysis-timestamp -b --isRun2MC 1

# PID
# It is not making histogram just to make table
echo "PID"
#o2-analysis-spectra-tof --aod-file ${DATA}/AO2D.root -b | o2-analysis-pid-tof -b | o2-analysis-pid-tof-base -b
#o2-analysis-pid-tof --aod-file ${DATA}/AO2D.root -b | o2-analysis-timestamp -b | o2-analysis-pid-tof-base -b
o2-analysis-pid-tof-full --aod-file ${DATA}/AO2D.root -b | o2-analysis-timestamp -b | o2-analysis-pid-tof-base -b
#o2-analysis-pid-tpc-full --aod-file ${DATA}/AO2D.root -b | o2-analysis-timestamp -b | o2-analysis-multiplicity-table -b
#o2-analysis-spectra-tpc --aod-file ${DATA}/AO2D.root -b | o2-analysis-pid-tpc -b

## Track Selection
#echo "Track Selection"
#
## For HF event&track selection
#o2-analysis-hf-track-index-skims-creator --aod-file ${DATA}/AO2D.root | o2-analysis-trackselection -b | o2-analysis-trackextension -b | o2-analysis-timestamp -b | o2-analysis-hf-candidate-creator-2prong -b | o2-analysis-hf-candidate-creator-3prong -b | o2-analysis-hf-candidate-creator-bplus -b

