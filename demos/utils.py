import pickle

import pandas as pd
import torch
from pam.activity import Activity, Leg, Plan
from pam.core import Household, Person
from pam.utils import minutes_to_datetime
from pandas import DataFrame
from pytorch_lightning import Trainer

from caveat import models
from caveat.data import build_conditional_dataloader


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
    for pid, schedule in schedules.groupby(schedules.pid):
        plan = build_plan(schedule)
        if plan is None:
            continue
        person = Person(pid)
        person.plan = plan
        hh.add(person)
    hh.plot()


class Generator:
    def __init__(
        self, ckpt_path, schedule_encoder_path, attributes_encoder_path
    ) -> None:
        # load model from checkpoint
        self.model = (
            models.sequence.cond_gen_lstm.CVAE_LSTM.load_from_checkpoint(
                ckpt_path
            )
        )

        # load encoders
        with open(schedule_encoder_path, "rb") as f:
            self.schedule_encoder = pickle.load(f)

        with open(attributes_encoder_path, "rb") as f:
            self.attributes_encoder = pickle.load(f)

        self.ckpt_path = ckpt_path
        self.trainer = Trainer()

    def __call__(self, synthetics):
        return self.gen(synthetics)

    def gen(self, synthetics):
        return trim(stretch(pad(self._gen(synthetics))))

    def _gen(self, synthetics):
        synthetic_conditionals = self.attributes_encoder.encode(synthetics)

        dataloader = build_conditional_dataloader(
            synthetic_conditionals, 6, max(len(synthetic_conditionals), 256)
        )

        predictions = self.trainer.predict(
            model=self.model, ckpt_path=self.ckpt_path, dataloaders=dataloader
        )

        schedules = self.schedule_encoder.decode(torch.concat(predictions))
        return schedules


def stretch(schedules):
    return schedules.groupby(schedules.pid).apply(stretcher)


def stretcher(schedule):
    duration = schedule.duration.sum()
    if duration != 1440:
        a = 1440 / duration
        schedule.duration = (schedule.duration * a).astype(int)
        accumulated = list(schedule.duration.cumsum())
        schedule.start = [0] + accumulated[:-1]
        schedule.end = accumulated
    return schedule


def trim(schedules):
    schedules[schedules.end > 1440] = 1440
    schedules[schedules.start > 1440] = 1440
    schedules.duration = schedules.end - schedules.start
    schedules = schedules[schedules.duration > 0]
    return schedules


def pad(schedules):
    return (
        schedules.groupby(schedules["pid"]).apply(padder).reset_index(drop=True)
    )


def padder(schedule):
    if schedule.end.iloc[-1] < 1440 and schedule.act.iloc[-1] != "home":
        pid = schedule.pid.iloc[0]
        schedule = pd.concat(
            [
                schedule,
                DataFrame(
                    {
                        "pid": pid,
                        "start": schedule.end.iloc[-1],
                        "end": 1440,
                        "duration": 1440 - schedule.end.iloc[-1],
                        "act": "home",
                    },
                    index=[0],
                ),
            ]
        )
    elif schedule.end.iloc[-1] < 1440:
        schedule.end.iloc[-1] = 1440
        schedule.duration.iloc[-1] = 1440 - schedule.start.iloc[-1]
    return schedule


class ImposterGame:
    def __init__(self, generator, observed, n=4):
        self.offset = pd.Series(range(n + 1)).sample(1).values[0]
        pids = pd.Series(observed.pid.unique()).sample(n)
        observed = observed[observed.pid.isin(pids)]
        pids = pd.Series(observed.pid.unique()).sample(self.offset)
        first = observed[observed.pid.isin(pids)]
        last = observed[~observed.pid.isin(pids)]

        synthetic_attributes = pd.DataFrame(
            [{"gender": "M", "age": 30, "income": 10, "area": "urban"}] * 10
        )
        synthetics = generator.gen(synthetic_attributes)
        pids = pd.Series(synthetics.pid.unique()).sample(1)
        synthetics = synthetics[synthetics.pid.isin(pids)]
        synthetics["pid"] = -1

        population = pd.concat([first, synthetics, last])
        pids = population.pid.unique()
        mapper = {pid: i for i, pid in enumerate(pids)}
        population.pid = population.pid.map(mapper)

        plot(population)

    def guess(self, location):
        if location == self.offset + 1:
            print("Correct!")
        else:
            print(f"Wrong! Correct answer is: {self.offset + 1}")

    def answer(self):
        return self.offset + 1
