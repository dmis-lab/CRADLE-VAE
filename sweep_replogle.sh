# redun run paper/experiments/scversal_replogle/1_launch_train_sweeps.py launch_all_train_sweeps --num-agents-per-sweep 5 --start-device 3

redun run paper/experiments/scversal_replogle/2_launch_sweep_evals.py launch_all_sweep_evals --jobs-per-agent 60 --device 3 --qc-pass True