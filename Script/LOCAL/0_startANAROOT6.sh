#local running with this code requires testfiles downloaded and a testSampleESD.txt or testSampleAOD.txt text file with the input files stored in, for example pPb_5TeV/LHC16q/testSampleESD.txt

# if [ "$1" == "" ]; then
# echo "Please give one or multiple task(s) as argument (example: PC):  QA (photon and cluster QA), P (PCM), C (Calo [EMC, DMC, PHOS]), H (hybrid PCM-Calo), M (merged EMC), S (skimming ESD or AOD)"
# exit
# fi

energy="pp_13TeV"
#energy="pp_13TeV"
intMCrunning=0 #0: data, 1: MC, 2: JJ MC
collsys=0 #0: pp, 1: PbPb, 2: pPb
runPeriod="LHC16g"
runPeriodData="LHC16g"
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
numLocalFiles=1
isRun2="kTRUE"
isLx="kFALSE"

workDIR="$energy/$runPeriod/$runMode$dataType"
mkdir -p $workDIR
cd $energy/$runPeriod
#cd $energy/$runPeriod/$runMode$dataType
#mkdir -p $energy/$runPeriod/$runMode$dataType

###################################
####### CREATE FILE LISTS #########
###################################
runNumber="296623"
period="18q"
LocalDIR="/Users/hanseopark/alice/work/Data/LocalFiles/$energy/$runPeriod/$tenderPassData/$dataType/$runNumber"
if [ isLX = "kTRUE" ]; then
	fileListName="test$runPeriod${dataType}_lx"
else
	fileListName="test$runPeriod$dataType"
fi
if [ -f ${fileListName}.txt ]; then
	echo "file ${fileListName}.txt has already been made. "
	echo "remove ${fileListName}.txt "
	rm ${fileListName}.txt
	rm ../$energy/$runPeriod/${fileListName}.txt
else
	touch -f ${fileListName}.txt
fi
for i in {1..$numLocalFiles}
do
		number=$( printf '%04d' $i)
		echo "$LocalDIR/$number/root_archive.zip" >> ${fileListName}.txt
done
cp ${fileListName}.txt ../$energy/$runPeriod/.


######################
####### RUN ##########
######################
cd $runMode$dataType
# valgrind --tool=massif --error-limit=no --max-stackframe=3060888 --suppressions=$ROOTSYS/etc/valgrind-root.supp --num-callers=40  --log-file=./valgrind_memory.log aliroot -x -l -b -q '../../../runLocalAnalysisROOT6.C('$intMCrunning','$collsys', "'$runPeriod'", "'$runPeriodData'", "'$dataType'", "'$runMode'", '$recoPassData', "'$tenderPassData'", '$useCorrTask', "'$aodConversionCutnumber'", '$isRun2', '$numLocalFiles')'
# valgrind --tool=callgrind aliroot -x -l -b -q '../../../runLocalAnalysisROOT6.C('$intMCrunning','$collsys', "'$runPeriod'", "'$runPeriodData'", "'$dataType'", "'$runMode'", '$recoPassData', "'$tenderPassData'", '$useCorrTask', "'$aodConversionCutnumber'", '$isRun2', '$numLocalFiles')'
# valgrind --tool=callgrind aliroot -b -q 'test.C+()'
if [ $collsys = 0 ]; then
aliroot -x -l -b -q '../../../runInJetpp.C.C('$intMCrunning','$collsys', "'$runPeriod'", "'$runPeriodData'", "'$dataType'", "'$runMode'", '$recoPassData', "'$tenderPassData'", '$useCorrTask', "'$aodConversionCutnumber'", '$isRun2', '$numLocalFiles', '$isLx')'
fi
if [ $collsys = 1 ]; then
aliroot -x -l -b -q '../../../runInJetPbPb.C('$intMCrunning','$collsys', "'$runPeriod'", "'$runPeriodData'", "'$dataType'", "'$runMode'", '$recoPassData', "'$tenderPassData'", '$useCorrTask', "'$aodConversionCutnumber'", '$isRun2', '$numLocalFiles', '$isLx')'
fi
