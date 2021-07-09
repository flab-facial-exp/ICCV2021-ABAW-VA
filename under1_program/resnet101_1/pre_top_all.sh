#!/bin/bash


# target dataset #
#1:using
#2:not use


echo "shell script: pre_top_all.sh start"
echo "-------------------------------------------"


### check argument ###
if [[ $# != 2 ]]; then
	echo "shell script: bad argument"
	dataset=0
	gpu_num=0
else
	echo "shell script: Your setting argument is below."
	dataset=$1
	gpu_num=$2
	echo "shell script: dataset or top_mode is ${dataset}."
	echo "shell script: GPU no ${gpu_num} will be used."
fi
##############################


### parameter setting ###
#########################

### clear and make directory ###
rm -r result_dataset${dataset}

mkdir result_dataset${dataset}
#########################

### loop number will be decided in accordance with dataset number ###
if [[ $dataset == 1 ]]; then #1:using mode
	#echo "The target dataset will be shuffled for training and testing."
	echo "shell script: The target dataset will be created from list file."
	start_num=1 
	end_num=1
	echo "shell script: Loop start_num is ${start_num} and end_num is ${end_num}"
elif [[ $dataset == 2 ]]; then #2: extended mode, maybe used in the future.
	echo "shell script: The taget dataset has fixed training and testing data."
	start_num=0
	end_num=0
	echo "Loop start_num is ${start_num} and end_num is ${end_num}"
else
	echo "shell script: Please set dataset number from 1 to 2."
	echo "shell script: This shell will be terminated."
	exit
fi
#########################

all_delete_flag=1
#all_delete_flag=0
echo "all_delete_flag is ${all_delete_flag}."

if [[ $all_delete_flag == 1 ]]; then
	echo 'Would you like to delete all files and run initially?'
	echo ""
	read -p "ok? (y/n): " yn
	case "$yn" in 
		[yY]* ) ;; 
	*) echo "abort." ; exit ;; 
	esac

	echo 'delete_all related files to run all shell script.'
	rm -r /home/ubuntu/iccv2021_abaw/abaw_dataset/using_dataset/done_flag_dir
	rm -r /home/ubuntu/iccv2021_abaw/abaw_dataset/using_dataset/annotation_rev
	rm -r /home/ubuntu/iccv2021_abaw/abaw_dataset/using_dataset/train
	rm -r /home/ubuntu/iccv2021_abaw/abaw_dataset/using_dataset/test
	rm -r /home/ubuntu/iccv2021_abaw/abaw_dataset/using_dataset/submit_test

	rm ./fine_tune/transfer_resnet_img_data.model

	rm -r /home/ubuntu/iccv2021_abaw/abaw_dataset/using_dataset/*.pickle

	rm ./classifier/*.pickle

	rm ./submit_make/*.sav
	rm -r ./submit_make/submisssion_files_catboost
	rm -r ./submit_make/submisssion_files_svr
else
	echo "No deletion files. continue programs."
fi

####################
### dataset loop ###
####################
for i in `seq $start_num $end_num`
	do
	echo "----------------------------------"
	echo "shell script: current loop number is :"$i
	random_seed=$i

	if [[ $dataset == 1 ]]; then #set random_seed
		if [[ $i == 1 ]]; then #from 1 to 5
			random_seed=2425
		elif [[ $i == 2 ]]; then #from 1 to 5
			random_seed=2064
		elif [[ $i == 3 ]]; then #from 1 to 5
			random_seed=2074
		elif [[ $i == 4 ]]; then #from 1 to 5
			random_seed=2146
		elif [[ $i == 5 ]]; then #from 1 to 5
			random_seed=2162
		else
			echo "shell script: Please set random seed correctly."
			echo "shell script: This shell will be terminated."
			exit
		fi
	fi


	echo "shell script: clear all temp files start."
	#rm -rf ./combine/train_*
	#rm -rf ./combine/test_*

	#rm ./combine/input_file_list.txt
	#rm ./classifier/test_result.txt

	#rm ./fine_tune/transfer_resnet_img_data.model

	#rm ./classifier/train_*.pickle
	#rm ./classifier/test_*.pickle
	echo "clear all temp files end."

	### split_dataset based on random_seed ###
	if [[ $dataset == 1 ]]; then #
		echo "shell script: split_file_select start."
		cd ./split_file_select
		#if [ -d /home/ubuntu/iccv2021_abaw/abaw_dataset/using_dataset/ ]; then
		if [ -d ./using_dataset/ ]; then
			echo "symbolic link for dataset is exist."
		else
			echo "symbolic link for dataset will be created."
			ln -s /home/ubuntu/iccv2021_abaw/abaw_dataset/using_dataset/
		fi

		if [ -d ./using_dataset/done_flag_dir ]; then
			echo "skip split and file selection due to the existence of done_flag_dir."
		else
			./all.sh
		fi	
		cd ../
		echo "shell script:split_file_select end."
	elif [[ $dataset == 2 ]]; then #2:fix dataset, path is temporary
		echo "copy train and test dataset to fine_tune directory start."
		cp -r  /home/ubuntu/iccv2021_abaw/abaw_dataset/train/img ./fine_tune/train_img
		cp -r  /home/ubuntu/iccv2021_abaw/abaw_dataset/test/img ./fine_tune/test_img
		cp -r  /home/ubuntu/iccv2021_abaw/abaw_dataset/train/nlp ./combine/train_nlp_raw_feature
		cp -r  /home/ubuntu/iccv2021_abaw/abaw_dataset/test/nlp ./combine/test_nlp_raw_feature
		#cp -r  /home/tyamamoto/dataset/sentiment/EmotionROI/train_nlp_raw_feature ./combine
		#cp -r  /home/tyamamoto/dataset/sentiment/EmotionROI/test_nlp_raw_feature ./combine
		echo "copy train and test dataset to fine_tune directory end."
	fi
	#######################


	### fine_tuning ###
	cd ./fine_tune
	echo "shell script: fine_tune start."

	if [ -d ./using_dataset/ ]; then
		echo "symbolic link for dataset is exist."
	else
		echo "symbolic link for dataset will be created."
		ln -s /home/ubuntu/iccv2021_abaw/abaw_dataset/using_dataset/
	fi

	if [ -f ./transfer_resnet_img_data.model ]; then
		echo "skip fine_tune process due to the existence of transfer_resnet_img_data.model."
	else
		./all.sh $dataset $gpu_num
		mv train_class_valence_label.pickle ./using_dataset/
		mv test_class_valence_label.pickle ./using_dataset/
		mv train_class_arousal_label.pickle ./using_dataset/
		mv test_class_arousal_label.pickle ./using_dataset/
	fi	

	cd ../
	echo "shell script: fine_tune end."
	###############################


	###############################
	### loop for op_mode ##########
	### op_mode decides feature combinations
	###############################
	temp_mode=0 #to avoid unnecessary fine-tuning

	#for j in `seq 0 6` #loop number is op_mode(feature combinations)
	for j in `seq 0 0` #loop number is op_mode(feature combinations)
		do

		op_mode=$j
		### combination of features ###
		echo "shell script: combine start."
		cd ./combine
		if [ -f transfer_resnet_img_data.model ]; then #
			echo "symbolic link for transfer model is exist."
		else
			echo "symbolic link for transfer model will be created."
			ln -s ../fine_tune/transfer_resnet_img_data.model
		fi

		if [ -d ./using_dataset/ ]; then
			echo "symbolic link for dataset is exist."
		else
			echo "symbolic link for dataset will be created."
			ln -s /home/ubuntu/iccv2021_abaw/abaw_dataset/using_dataset/
		fi

		./all.sh $op_mode $temp_mode $gpu_num $dataset
		#cd ../
		echo "combine end."
		###############################

		temp_mode=1

		echo "move created pickle files to classifier directory start."
		mv ./using_dataset/train_combined.pickle ../classifier
		mv ./using_dataset/test_combined.pickle ../classifier

		mv ./using_dataset/train_class_valence_label.pickle ../classifier
		mv ./using_dataset/test_class_valence_label.pickle ../classifier

		mv ./using_dataset/train_class_arousal_label.pickle ../classifier
		mv ./using_dataset/test_class_arousal_label.pickle ../classifier

		#mv ./using_dataset/submit_test_combined.pickle ../classifier
		echo "move created pickle files to classifier directory end."
		cd ../


		### classifier ###
		echo "classifier start."
		echo "classifying will be conducted."
		if [ -f ./submit_make/classifier_model_arousal_catboost.sav ]; then
			echo "classifier model exist. skip classification process."
		else
			cp -r ./classifier ./classifier_dataset${dataset}_seed${random_seed}_comb${op_mode}
			#./pre_classifier.sh $dataset $random_seed $op_mode $gpu_num & #conducting classifier in the background
			./pre_classifier.sh $dataset $random_seed $op_mode $gpu_num  #conducting classifier in the background
		fi
		echo "classifier end."
		done
		##########################

		echo "shell_script:submit_make start."
		cd ./submit_make

		if [ -d ./using_dataset/ ]; then
			echo "symbolic link for dataset is exist."
		else
			echo "symbolic link for dataset will be created."
			ln -s /home/ubuntu/iccv2021_abaw/abaw_dataset/using_dataset
		fi

		if [ -d ./submisssion_files_catboost ]; then
			echo "skip submit_make process."
		else
			./all.sh $op_mode $random_seed
		fi
		echo "shell_script:submit_make end."
	###############################
	###############################
	###############################

	echo "---------------------------------------------"
	done

echo "pre_top_all.sh end"
