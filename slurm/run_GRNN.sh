#!/bin/bash

mySRun() {
   strScriptDirName=.tmp_scripts/$(basename $(realpath $1/../../..))/$(basename $(realpath $1/../..))/$(basename $(realpath $1/../))
   mkdir -p $strScriptDirName
   strFilename="$strScriptDirName/grnn_$2.sh"
	rm -f $strFilename

	echo "#!/bin/bash" >> $strFilename
   echo "pushd $1" >> $strFilename
   echo "th ~/mygithub/GRNN_Clean/trainPred.lua ../ $2" >> $strFilename
   echo "popd" >> $strFilename
	chmod u+x $strFilename
	echo sbatch -p low -c 2 -n 1 -N 1 -t 300 --job-name=$strFilename $strFilename
}

lRunGrnn(){
   mkdir -p .tmp_scripts
   dataDir=$(realpath $1)
   for i in  $(ls $dataDir/n10_f[0-9]*.txt)
   do
      strFoldFilename=$(basename $i)
      mySRun $dataDir $strFoldFilename
   done
}

#for baseDir in $HOME/mygithub/grnnPipeline/data/dream5_med2/modules_gnw_a/size-*/
for baseDir in $HOME/mygithub/GRNN_Clean/data/dream5_ecoli/modules/size-10/
do
   for j in $baseDir/top_edges-[0-9]*_gnw_data/d_1/folds
   do
      strFoldDirname=$j
      lRunGrnn $strFoldDirname
   done
done
