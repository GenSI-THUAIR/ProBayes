#!/bin/bash

DATASET="pepbench" # Default dataset

while getopts "d:" opt;
do
  case $opt in
    d) DATASET="$OPTARG";;
    ?)
      echo "Usage: $0 [-d dataset_name]" >&2
      exit 1
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES=0,1,2,3

export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

 if [ "$DATASET" = "pepbdb" ]; then
     export CFG_PATH='./configs/bfn_quat_pepbdb.yaml'
     export TAG='pepbdb_codesign'$ADD_INFO
 elif [ "$DATASET" = "pepbench" ]; then
     export CFG_PATH='./configs/bfn_quat_pepbench.yaml'
     export TAG='pepbench_codesign'$ADD_INFO
 else
     echo "Error: Unknown dataset specified: $DATASET" >&2
     exit 1
 fi

export TORCH_DISTRIBUTED_DEBUG='INFO'

python -m torch.distributed.launch \
    --master_port=25646 --nproc_per_node=$NUM_GPUS train_pep_ddp.py \
        --config $CFG_PATH --tag $TAG \
         --kwargs model.sample_seq=False dataset.n_gen_samples=10 train.val_freq=8000
