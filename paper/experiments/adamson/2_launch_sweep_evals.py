from typing import Optional

import redun
import yaml, sys
import cupy as cp

sys.path.append('/'.join(__file__.split('/')[:-4]))
from launch_sweep import evaluate_sweep


@redun.task(cache=False)
def launch_all_sweep_evals(
    jobs_per_agent: Optional[int] = None,
    device: Optional[list] = None,
    qc_pass: Optional[bool] = False,
    batch_size: Optional[int] = 128,
):
    with open("paper/experiments/scversal_adamson/sweep_ids.yaml") as f:
        sweep_ids = yaml.safe_load(f)

    device = [int(i) for i in device]
    # cp.cuda.Device(device[0]).use()
    out = []
    for name, sweep_id in sweep_ids.items():
        ret = evaluate_sweep(
            sweep_id=sweep_id,
            perturbseq=1,
            ate_n_particles=2500,
            jobs_per_agent=jobs_per_agent,
            executor="sweep_agent_batch_large",
            device=device,
            qc_pass=qc_pass,
            batch_size=batch_size,
        )
        out.append(ret)

    return out
