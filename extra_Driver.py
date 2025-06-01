# connect from a new terminal
import ray, time
from qaoa.models import params
from worker import Runner           # same class your first driver used

ray.init(address="auto")            # joins the running cluster

cpus_per_worker =  params.CPUS_PER_WORKER
queue          =  ray.get_actor("runqueue")

new_total_cpus = int(ray.cluster_resources()["CPU"])         # live view
already_used   = len(ray.actors()) * cpus_per_worker         # rough…
extra_workers  = (new_total_cpus - already_used) // cpus_per_worker

for _ in range(extra_workers):
    Runner.options(num_cpus=cpus_per_worker).remote(queue)

print(f"✅ Launched {extra_workers} new workers.")
print(f"Total workers: {len(ray.actors())} (each using {cpus_per_worker} CPUs)")