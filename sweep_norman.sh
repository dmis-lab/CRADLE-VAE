# redun run paper/experiments/scversal_norman/1_launch_train_sweeps.py launch_all_train_sweeps --num-agents-per-sweep 6 --start-device 2

redun run paper/experiments/scversal_norman/2_launch_sweep_evals.py launch_all_sweep_evals --jobs-per-agent 6 --device 2 --qc-pass True