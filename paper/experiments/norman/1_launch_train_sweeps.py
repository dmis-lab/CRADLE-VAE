from typing import Optional
import pdb,sys

import redun

sys.path.append('/'.join(__file__.split('/')[:-4]))
from launch_sweep import launch_sweep


@redun.task(cache=False)
def launch_all_train_sweeps(
    num_agents_per_sweep: Optional[int] = None,
    start_device: Optional[int] = None,
    jobs_per_agent: Optional[int] = None,
):
    # if num_agents_per_sweep is None:
    #     num_agents_per_sweep = len(devices)
    #     devices = [int(i) for i in devices]
    config_paths = redun.file.glob_file("paper/experiments/norman/configs/*.yaml")
    out = []
    for path in config_paths:
        ret = launch_sweep(
            config_path=path,
            num_agents=num_agents_per_sweep,
            start_device=start_device,
            jobs_per_agent=jobs_per_agent,
            executor="sweep_agent_batch_large",
        )
        out.append(ret)
    return out
