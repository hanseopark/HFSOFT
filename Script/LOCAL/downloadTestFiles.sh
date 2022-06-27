dataOrMC=MC #data or MC
ESDOrAOD=AOD #ESD or AOD
#energy=pPb_5TeV
energy=pp_13TeV
#energy=PbPb_5TeV

# set these if you want to download data
#yearData=2016
#periodData=LHC16q
#pass=pass1_CENT_woSDD
yearData=2017
#periodData=LHC18b
#periodData=LHC17k4
pass=pass3
localpath=/Users/hanseopark/alice/work/Data/LocalFiles

# set these if you want to download MC
yearMC=2017
#periodMC=LHC18f3_cent_woSDD_2
#periodMCshort=LHC18f3_2
periodMC=LHC17k4
periodMCshort=LHC17k4

#AODFILTER=234
AODFILTER=235

# must be set either way
#RUN=265500
#RUN=254332
#RUN=294925
#RUN=285396
#RUN=296623
RUN=274442

downloadFile=root_archive.zip

RUNDL=$RUN
passDL=$pass

# give number of maxim files to be downloaded for the data set
NMaxFiles=10
counter=0

if [ $dataOrMC = "MC" ]; then
    if [ $ESDOrAOD = "AOD" ]; then
        RUNDL="$RUN/AOD$AODFILTER"
        downloadFile=root_archive.zip
    fi
	LocalTemp=$localpath/$energy/$dataOrMC/$periodMCshort/$ESDOrAOD/$RUN
    mkdir -p LocalTemp
    for PARTS in $( alien_ls /alice/sim/$yearMC/$periodMC/$RUNDL/ )
    do
        if [ -f $LocalTemp/$PARTS/$downloadFile ]; then
            echo "file $downloadFile has already been copied for run " $RUN "and part" $PARTS
        else
            mkdir $LocalTemp/$PARTS
            echo /alice/sim/$yearMC/$periodMC/$RUNDL/$PARTS/$downloadFile
            alien_cp alien:/alice/sim/$yearMC/$periodMC/$RUNDL/$PARTS/$downloadFile file:$LocalTemp/$PARTS/$downloadFile
            unzip -t $LocalTemp/$PARTS/$downloadFile
            md5sum $LocalTemp/$PARTS/$downloadFile
        fi
        counter=`expr $counter + 1`
        if [ $counter = $NMaxFiles ]; then
          exit
        fi
    done
fi

if [ $dataOrMC = "data" ]; then
    if [ $ESDOrAOD = "AOD" ]; then
        passDL="$pass/AOD$AODFILTER"
        #downloadFile=root_archive.zip
        #downloadFile=aod_archive.zip
        downloadFile=AliAOD.root
    fi
	LocalTemp="$localpath/$energy/$dataOrMC/$periodData/$pass/$ESDOrAOD/$RUN/"
    mkdir -p $LocalTemp
	if [ $energy = "PbPb_5TeV" ]; then
		partsLINK="/alice/data/$yearData/$periodData/000$RUN/$passDL/$ESDOrAOD"
	else
		partsLINK="/alice/data/$yearData/$periodData/000$RUN/$passDL"
	fi
	for PARTS in $( alien_ls $partsLINK )
#for PARTS in $( alien_ls /alice/data/$yearData/$periodData/000$RUN/$passDL )
	do
	echo $PARTS
        if [ -f $LocalTemp/$PARTS/$downloadFile ]; then
            echo "file $downloadFile has already been copied for run " $RUN "and part" $PARTS
        else
            mkdir $LocalTemp/$PARTS
            echo /alice/data/$yearData/$periodData/000$RUN/$passDL/$PARTS/$downloadFile
            #alien_cp alien:/alice/data/$yearData/$periodData/000$RUN/$passDL/$PARTS/$downloadFile file:$localpath/$energy/$periodData/$pass/$ESDOrAOD/$RUN/$PARTS/$downloadFile
            alien_cp alien:$partsLINK/$PARTS/$downloadFile file:$localpath/$energy/$periodData/$pass/$ESDOrAOD/$RUN/$PARTS/$downloadFile
            unzip -t $LocalTemp/$PARTS/$downloadFile
            md5sum $LocalTemp/$PARTS/$downloadFile
        fi
        counter=`expr $counter + 1`
        if [ $counter = $NMaxFiles ]; then
          exit
        fi
    done
fi
