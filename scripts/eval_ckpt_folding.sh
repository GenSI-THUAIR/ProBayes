export CKPT_DIR='ckpts/pepbdb_folding' 
export CKPT_DIR='ckpts/pepbench_folding'
export DEVICE='cuda:0'

conda activate probayes
python probayes/eval/get_ckpt_all_metrics.py --ckpt_dir $CKPT_DIR --device $DEVICE --n_test_items 93 --sample_mode 'end_back'


