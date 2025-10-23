# generate results on the test set and save to ./results/fixseq
# python generate.py --config configs/pepbdb/test_codesign.yaml --ckpt checkpoints/codesign_pepbdb.ckpt --gpu 3 --save_dir /data/wuhl/bfn4pep/remote/PepGLAD/pepbdb_results

# calculate dG
# python /data/wuhl/bfn4pep/remote/PepGLAD/evaluation/dG/run.py --results /data/wuhl/bfn4pep/remote/PepGLAD/pepbdb_results/results.jsonl
 
# calculate metrics
python cal_metrics.py --results /data/wuhl/bfn4pep/remote/PepGLAD/pepbdb_results/results.jsonl 

# python cal_metrics.py --results /data/wuhl/bfn4pep/remote/PepGLAD/pepbdb_results/results.jsonl --filter_dG /data/wuhl/bfn4pep/remote/PepGLAD/pepbdb_results/dG_report.jsonl
