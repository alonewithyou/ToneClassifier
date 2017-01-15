#!/bin/bash
function set_up_global_variables {
	cntk_activaton_file="/home/lagoon/Downloads/cntk/activate-cntk"
	bashrc_file="/home/lagoon/.bashrc"
	results_dir="/home/lagoon/DL_Project/results/"
	src_dir="/home/lagoon/DL_Project/src/"
}
function setup_cntk {
	source $cntk_activaton_file
}
function clean_cntk {
	source $bashrc_file
}
# evaluate_cntk file
function evaluate_cntk {
	src_file="$src_dir$1".py
	setup_cntk
	python $src_file |& tee "$results_dir/result_$1".txt
	clean_cntk
}
# evaluate_mxnet file
function evaluate_mxnet {
	src_file="$1".py
	cd $src_dir
	python $src_file |& tee "$results_dir/result_$1".txt
	cd ..
}
# evaluate_torch file
function evaluate_torch {
	src_file="$1".lua
	cd $src_dir
	th $src_file |& tee "$results_dir/result_$1".txt
	cd ..
}
function testment_mxnet {
	evaluate_mxnet MXNET
}
function testment_cntk {
	#evaluate_cntk CNTK
	#evaluate_cntk train_978
	evaluate_cntk train_89
	#evaluate_cntk train_877
	#evaluate_cntk train_87
	#evaluate_cntk train_865
	#evaluate_cntk train_dnn
}
function testment_torch {
	evaluate_torch torch
}

set_up_global_variables
testment_cntk
#testment_mxnet
#testment_torch
