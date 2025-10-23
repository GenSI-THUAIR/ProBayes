
export CKPT_DIR='ckpts/pepbdb_codesign'
export CKPT_DIR='ckpts/pepbench_codesign'

export DEVICE='cuda:0'

conda activate probayes

python probayes/eval/get_ckpt_all_metrics.py --ckpt_dir $CKPT_DIR --device $DEVICE\
             --n_test_items 5 --num_samples 40 --sample_mode 'end_back' 

