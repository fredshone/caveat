import multiprocessing as mp

import pandas as pd
from pam.activity import Activity, Plan, Trip
from pam.utils import minutes_to_datetime as mtdt


def trace_to_pam(trace: list[tuple], mapping: dict):
    plan = Plan()
    for act, start, end, duration in trace:
        name = mapping[act]
        plan.add(Activity(act=name, start_time=mtdt(start), end_time=mtdt(end)))
        plan.add(Trip(mode="car", start_time=mtdt(end), end_time=mtdt(end)))
    return plan


def generate_population(gen, size: int, cores: int = None):
    if cores is None:
        cores = mp.cpu_count()

    batches = list(split(range(size), cores))

    pools = mp.Pool(cores)
    results = [pools.apply_async(gen_persons, args=(gen, pids)) for pids in batches]
    pools.close()
    pools.join()
    results = [r.get() for r in results]
    pop = pd.concat(results, ignore_index=True)
    return pop


def gen_persons(gen, pids) -> pd.DataFrame:
    return pd.concat([gen_person(gen, pid) for pid in pids], ignore_index=True)


def gen_person(gen, pid) -> pd.DataFrame:
    trace = gen.run()
    return trace_to_df(trace, pid=pid)


def trace_to_df(trace: list[tuple], **kwargs) -> pd.DataFrame:
    df = pd.DataFrame(trace, columns=["act", "start", "end", "duration"])
    for k, v in kwargs.items():
        df[k] = v
    return df


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
