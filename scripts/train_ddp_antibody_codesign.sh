export CUDA_VISIBLE_DEVICES=0,1,2,3

export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export CFG_PATH='configs/bfn_antibody.yaml'


export TAG='default'
export TORCH_DISTRIBUTED_DEBUG='INFO'

python -m torch.distributed.launch \
    --master_port=25643 --nproc_per_node=$NUM_GPUS train_antibody_ddp.py \
    --config $CFG_PATH --tag $TAG
