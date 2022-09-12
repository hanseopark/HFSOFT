#!/bin/bash

# Settings
DATE=`date +%y%m%d`
DAUGHTER=0
collsys="$1"
energy="$2"
ANALYSISNAME="$3"
ANALYSISNAME=${ANALYSISNAME:-"NoNamed"}
WORKDIRNAME="${DATE}_${DAUGHTER}_${ANALYSISNAME}"
LATEST="NULL"
if [ ${energy} = "5TeV" ]
then
	energy="5.023TeVRun2"

fi

# Directory path
INITDIR=`pwd`
LATESTWORKDIR="${INITDIR}/${LATEST}"
WORKDIR="${INITDIR}/${WORKDIRNAME}"
HFDIR="$HOME/alice/work/HFSOFT"
SCRDIR="${HFDIR}/Script/${collsys}/${ANALYSISNAME}"
ANSDIR="${HFDIR}/Script/${collsys}/ANSWER"
RESDIR="${HFDIR}/Result/${collsys}"
CERNBOX="NULL" #cernbox directory path
CERNBOXLATEST="NULL"

# Flags
OWNGIT="NO"
LOCAL="$4"
HFINIT="$5"

# Load cernbox local path
if [ -e ${HOME}/.local/share/data/CERNBox/cernbox.cfg ]
then
	TMP=`grep localPath ${HOME}/.local/share/data/CERNBox/cernbox.cfg`
	CERNBOX="${TMP:22}Main_Analysis"
	echo ${CERNBOX}
fi


# Look for the latest working directory to take over analysis codes
LATEST=`ls | grep ${ANALYSISNAME} | tail -1`
if [ "${LATEST}" != "NULL" ]
then
	echo /////////////////////////////////////////////////////////////////
	echo /////////////////////////////////////////////////////////////////
	echo ///////Latest working directory for ${ANALYSISNAME} = ${LATEST}/
	echo ///////Files will be taken over to today\'s//////////////////////
	echo /////////////////////////////////////////////////////////////////
	echo /////////////////////////////////////////////////////////////////
	LATESTWORKDIR="${INITDIR}/${LATEST}"
fi

while : # Search working directory and make daughter
do
	if [ -e $WORKDIRNAME ]
	then
		echo ${WORKDIRNAME} found.
		#DAUGHTER=`expr $DAUGHTER + 1`
		DAUGHTER=$((DAUGHTER + 1))
		WORKDIRNAME="${DATE}_${DAUGHTER}_${ANALYSISNAME}"
	else
	#	echo Generating new working directory ${WORKDIRNAME}
	#	cd ${INITDIR}
	#	mkdir $WORKDIRNAME
		WORKDIR="${INITDIR}/${WORKDIRNAME}" # Renew working directory full path
		break
	fi
done

# Look for the latest working directory to take over analysis codes in CERNBOX directory
if [ "${CERNBOX}" != "NULL" ]
then
	cd ${CERNBOX}
	CERNBOXLATEST=`ls | grep ${ANALYSISNAME} | tail -1`
	CERNBOXLATEST="${CERNBOX}/${CERNBOXLATEST}"
	echo ${CERNBOXLATEST}
else
	echo Couldn\'t access to ${CERNBOX}. Is the path correct\?
fi

# Generate new working directory in initial directory
cd ${INITDIR}
mkdir ${WORKDIRNAME}

## Clone analysiscodes from git repository
#git clone git@github.com:hanseopark/MyAliceAnalysis.git
#mv MyAliceAnalysis ${WORKDIRNAME}
#cd ${WORKDIRNAME}
#git checkout ${ANALYSISNAME}

# Clone from scripts and andswers in latest directory to HFSOFT

# Take over analysis codes
if [ "${LOCAL}" = "y" ]
then
	cd ${WORKDIR}
	if [ "${CERNBOXLATEST}" = "NULL" ]
	then
		if [ "${LATEST}" != "NULL" ]
		then
			echo Files copied from ${LATESTWORKDIR} clone to HFSOFT
			# /////////////////////////////////////////
			# // Add file type you want to take over //
			# /////////////////////////////////////////
			#cp ${LATESTWORKDIR}/*.cxx ./
			#cp ${LATESTWORKDIR}/*.h ./
			#cp ${LATESTWORKDIR}/*.C ./
			#cp ${LATESTWORKDIR}/* ./
			#cp ${LATESTWORKDIR}/*.sh ./
			#cp ${LATESTWORKDIR}/*.txt ./
			cp -i ${LATESTWORKDIR}/0_Script/*.sh ${SCRDIR}/.
			cp -i ${LATESTWORKDIR}/0_Answers/*.txt ${ANSDIR}/.
		fi
	else
		echo Files are copied from ${CERNBOXLATEST}
		# /////////////////////////////////////////
		# // Add file type you want to take over //
		# /////////////////////////////////////////
		#cp ${CERNBOXLATEST}/*.cxx ./
		#cp ${CERNBOXLATEST}/*.h ./
		#cp ${CERNBOXLATEST}/*.C ./
		cp ${CERNBOXLATEST}/*.sh ./
		cp ${CERNBOXLATEST}/*.txt ./
		#cp ${CERNBOXLATEST}/* ./
	fi
else
	echo File are not taken over.
fi


if [ "${HFINIT}" = "y" ]
then
	cd ${WORKDIR}
	#cp -r /Users/hanseopark/alice/hfsoft/Run3Analysisvalidation ${WORKDIR}/.
	#rm ${WORKDIR}/Run3Analysisvalidation/codeHF/runtest.sh
	#ln -s ${WORKDIR}/Run3Analysisvalidation/exec/runtest.sh ${WORKDIR}/Run3Analysisvalidation/codeHF/.
	bash /Users/hanseopark/alice/hfsoft/Run3Analysisvalidation/prepareResultDirectory.sh hapark

	echo "Copy Run3Analysisvalidation"

	echo "Pull script for ${ANALYSISNAME}"
	mkdir 0_Script # Run script files
	mkdir 0_Answers # answer files for run script
	mkdir work # Work directory
	if [ -d "$SCRDIR" ]; then
		#cp ${SCRDIR}/*.sh .
		ln -s ${SCRDIR}/*.sh 0_Script/.
		ln -s ${ANSDIR}/*.txt 0_Answers/.
		echo "Complete pull linked ALL script and txt for ${ANALYSISNAME}"
	else
		echo ${SCRDIR} is not exists.
	fi
	if [ "${ANALYSISNAME}" = "Comb" ]; then
		echo "Combination of all method"
		ln -s ${HFDIR}/Script/0_CombineMeasurement.sh .
	fi
	if [ -d "$RESDIR" ]; then
		#cp ${RESDIR}/*.root .
		cp ${RESDIR}/Jet_Unfolding_Corrections_${energy}_4.root RooUnfold/.
		cp ${RESDIR}/Jet_Unfolding_Corrections_${energy}_4_Default.root RooUnfold/.
		echo "To copy for Unfolding file with latest analysis"
	else
		echo ${RESDIR} is not exists.
	fi
else
	echo Copy won\'t be executed.
fi

# Check if take over HFSOFT
echo "Do you want to set the directory in working directory as MyHFSOFT(git)? (y/n)"
read OWNGIT

if [ "${OWNGIT}" = "y" ]
then
	cd ${WORKDIR}
	#git init
	#git remote add origin https://github.com/hanseopark/HFSOFT.git
	#git remote add origin git@github.com:hanseopark/HFSOFT.git
	#git add 0_Unfold.sh
	#git commit -m "New File1"
	#git status
	#git remote add origin https://github.com/hanseopark/HFSOFT
	#git push -u origin master
	#git fetch origin

	git clone git@github.com:hanseopark/HFSOFT.git
	echo "git test"
fi

## Check if take over MyAnalysisCode
#echo "Do you want to set the derectory My AnalysisCode for Result directory? (y/n)"
#read OWNRESULT
#
#if [ "${OWNRESULT}" = "y" ]
#then
#	cd ${WORKDIR}
#	mkdir MyResults
#	cp ../Result/Jet_Unfolding_Corrections_13TeV_4.root RooUnfold/./
#	cp ../Result/Jet_Unfolding_Corrections_13TeV_4_Default.root RooUnfold/./
#	cp ../MyCode/MyAnalysis.C ./
#	cp ../MyCode/MyAnalysisCutStudies.C ./
#	cp ../MyCode/MyAnalysisSystematicError.C ./
#	cp ../MyCode/MyUnfold.C ./
#	cp ../MyCode/Jet_Unfolding_Macro.C ./
#	cp ../MyCode/Jet_Unfolding_Macro_INT7.C ./
#	cp ../MyCode/Jet_Unfolding_Macro_EG1.C ./
#	cp ../MyCode/Jet_Unfolding_Macro_EG2.C ./
#	cp ../MyCode/TriggerCombination_PFR.dat ./
#	cp ../Result/MyRejectionFactor.root MyResults/./
#
#	echo "coping my code is done."
#
#fi

