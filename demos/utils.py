import pandas as pd
from pam.activity import Activity, Leg, Plan
from pam.core import Household, Person
from pam.utils import minutes_to_datetime


def to_datetime(minutes: int):
    return minutes_to_datetime(minutes)


def build_plan(schedule: pd.DataFrame):
    plan = Plan()
    try:
        for _, row in schedule.iterrows():
            start, end = to_datetime(row.start), to_datetime(row.end)
            plan.add(Activity(act=row.act, start_time=start, end_time=end))
            plan.add(Leg(mode="", start_time=end, end_time=end, distance=0))
        plan.day.pop(-1)
        return plan
    except Exception as e:
        print(e)
        return None


def plot(schedules: pd.DataFrame):
    hh = Household(0)
    for pid, schedule in schedules.groupby("pid"):
        plan = build_plan(schedule)
        if plan is None:
            continue
        person = Person(pid)
        person.plan = plan
        hh.add(person)
    hh.plot()
