export CKPT_DIR=logs/bfn_antibody[dev-723dcf1][07-14-15-22-00]_no_mixsc

export DEVICE='cuda:7'

python probayes/eval/antibody_eval_multi.py --ckpt_dir $CKPT_DIR --device $DEVICE\
                --run_gen True --num_samples 64 --sample_mode 'end_back'

# python probayes/eval/antibody_eval_single.py --ckpt_dir $CKPT_DIR --device $DEVICE\
#                 --run_gen True --sample_mode 'end_back' #正常work，这个ckpt没问题

