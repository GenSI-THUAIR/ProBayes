export CKPT_DIR=ckpts/antibody_codesign

export DEVICE='cuda:0'

conda activate probayes

python probayes/eval/antibody_eval_multi.py --ckpt_dir $CKPT_DIR --device $DEVICE\
                --run_gen True --num_samples 2 --sample_mode 'end_back'



