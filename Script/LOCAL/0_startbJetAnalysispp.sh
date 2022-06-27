#local running with this code requires testfiles downloaded and a testSampleESD.txt or testSampleAOD.txt text file with the input files stored in, for example pPb_5TeV/LHC16q/testSampleESD.txt

# if [ "$1" == "" ]; then
# echo "Please give one or multiple task(s) as argument (example: PC):  QA (photon and cluster QA), P (PCM), C (Calo [EMC, DMC, PHOS]), H (hybrid PCM-Calo), M (merged EMC), S (skimming ESD or AOD)"
# exit
# fi

### config
localpath="$LOCDIR/bJet"

#### Arguments for macro
energy="pp_13TeV"
intMCrunning=1 #0: data, 1: MC, 2: JJ MC
collsys=0 #0: pp, 1: PbPb, 2: pPb
runPeriod="LHC16g"
runPeriodMC="LHC17k4"
dataType="AOD" #ESD or AOD
runMode="C" #switch for which tasks to run: QA (photon and cluster QA), P (PCM), C (Calo [EMC, DMC, PHOS]), H (hybrid PCM-Calo), M (merged EMC), S (skimming ESD or AOD)
recoPassData=1
tenderPassData="pass1"
useCorrTask="kTRUE"
#useCorrTask="kFALSE"
aodConversionCutnumber="00000003_06000008d00100001100000000"; #It is
#aodConversionCutnumber="00000003_06000008400100001000000000";
#aodConversionCutnumber="00000003_00000008400100001500000000";
                        #06000008d00100001100000000
                        #00000008400100001500000000
                        # aod: 00000003_06000008d00100001100000000
numLocalFiles=2
isRun2="kTRUE"
isPileup="kTRUE"
isLx="kFALSE"

#### INFORMATION OF RUN ####
runNumber="254332"
runNumber="274442"

###########################
#### DIRECTORY SETTING ####
###########################

	### Local data or MC directory ###
if (( $intMCrunning == 0 )); then
	dataOrMC="data"
	LocalDIR="/Users/hanseopark/alice/work/Data/LocalFiles/$energy/$runPeriod/$tenderPassData/$dataType/$runNumber"
	echo "CHOICE OF LOCAL DATA"
	ls -ltr $LocalDIR
else
	dataOrMC="MC"
	LocalDIR="/Users/hanseopark/alice/work/Data/LocalFiles/$energy/$dataOrMC/$runPeriodMC/$dataType/$runNumber"
	runPeriod=$runPeriodMC
	echo "CHOICE OF LOCAL MC"
	ls -ltr $LocalDIR
fi

	### Working direcotry ###
workDIR="$energy/$dataOrMC/$runPeriod/$runMode$dataType"

if [ -f workDIR ]; then
	echo "WORKING DIREOCTY: $workDIR is exists"
	ls -ltr $workDIR
else
	mkdir -p $workDIR
fi
if [ -f FileLists ]; then
	echo "FileLists directory exists"
	ls -ltr FileLists
else
	mkdir -p FileLists
fi

###################################
####### CREATE FILE LISTS #########
###################################
cd FileLists
if [ isLX = "kTRUE" ]; then
	fileListName="test$runPeriod${dataType}_lx"
else
	fileListName="test$runPeriod$dataType"
fi
if [ -f ${fileListName}.txt ]; then
	echo "file ${fileListName}.txt has already been made. "
	echo "remove ${fileListName}.txt "
	rm ${fileListName}.txt
	rm ../$energy/$dataOrMC/$runPeriod/${fileListName}.txt
else
	touch -f ${fileListName}.txt
fi
for i in {1..$numLocalFiles}
do
		number=$( printf '%04d' $i)
		echo "$LocalDIR/$number/root_archive.zip" >> ${fileListName}.txt
done
cp ${fileListName}.txt ../$energy/$dataOrMC/$runPeriod/.

######################
####### RUN ##########
######################
cd ../$workDIR
###valgrind --tool=callgrind aliroot -x -l -b -q '../../../runLocalAnalysisROOT6.C('$intMCrunning','$collsys', "'$runPeriod'", "'$dataType'", "'$runMode'", '$recoPassData', "'$tenderPassData'", '$useCorrTask', "'$aodConversionCutnumber'", '$isRun2', '$numLocalFiles')'
aliroot -x -l -b -q '../../../../runbJetpp.C('$intMCrunning','$collsys', "'$runPeriod'", "'$dataType'", "'$runMode'", '$recoPassData', "'$tenderPassData'", '$useCorrTask', "'$aodConversionCutnumber'", '$isRun2', '$numLocalFiles', '$isPileup','$isLx')'
