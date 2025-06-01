# connect from a new terminal
import ray, time
from qaoa.models import params
from worker import Runner           # same class your first driver used

ray.init(address="auto",namespace="cb8cfc4a-c6ea-4c60-be22-9992966deb22")            # joins the running cluster

cpus_per_worker =  params.CPUS_PER_WORKER
queue          =  ray.get_actor("runqueue")

runners_alive = sum(
    1
    for a in ray._private.state.actors().values()   # internal
    if a["ClassName"] == "Runner" and a["State"] == "ALIVE"
)

total_cpus   = int(ray.cluster_resources()["CPU"])
extra_needed = (total_cpus // cpus_per_worker) - runners_alive

for _ in range(extra_needed):
    Runner.options(num_cpus=cpus_per_worker).remote(queue)

print(f"✅ Launched {extra_needed} new workers.")
print(f"Total workers: {len(ray.actors())} (each using {cpus_per_worker} CPUs)")