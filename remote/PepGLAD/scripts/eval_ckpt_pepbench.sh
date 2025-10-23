# generate results on the test set and save to ./results/fixseq
python generate.py --config configs/pepbench/test_codesign.yaml --ckpt checkpoints/codesign.ckpt --gpu 5 --save_dir /data/wuhl/bfn4pep/remote/PepGLAD/results

# calculate dG
# python /data/wuhl/bfn4pep/remote/PepGLAD/evaluation/dG/run.py --results /data/wuhl/bfn4pep/remote/PepGLAD/results/results.jsonl
 
# calculate metrics
# python cal_metrics.py --results /data/wuhl/bfn4pep/remote/PepGLAD/results/results.jsonl 

python /data/wuhl/bfn4pep/probayes/eval/get_metrics_from_json.py --json_path /data/wuhl/bfn4pep/remote/PepGLAD/results/results.jsonl
# python cal_metrics.py --results /data/wuhl/bfn4pep/remote/PepGLAD/pepbdb_results/results.jsonl --filter_dG /data/wuhl/bfn4pep/remote/PepGLAD/pepbdb_results/dG_report.jsonl
