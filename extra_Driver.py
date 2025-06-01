# connect from a new terminal
import ray, time
from qaoa.models import params
from worker import Runner           # same class your first driver used

ray.init(address="auto",namespace="cb8cfc4a-c6ea-4c60-be22-9992966deb22")            # joins the running cluster

cpus_per_worker =  params.CPUS_PER_WORKER
queue          =  ray.get_actor("runqueue")


while ray.available_resources().get("CPU", 0) >= cpus_per_worker:
    Runner.options(num_cpus=cpus_per_worker).remote(queue)
    # brief pause so the scheduler updates the available-CPU number
    time.sleep(0.5)



print(f"Total workers: {ray._private.state.actors()} (each using {cpus_per_worker} CPUs)")