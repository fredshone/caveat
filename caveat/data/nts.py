from pathlib import Path
from typing import Union

import pandas as pd
from pam import read
from pam.activity import Leg, Plan
from pam.core import Population
from pam.utils import datetime_to_matsim_time


def run():
    # inputs
    trips_csv = "~/Data/UKDA-5340-tab/tab/trip_eul_2002-2021.tab"
    write_dir = Path("processed")
    write_dir.mkdir(exist_ok=True)
    years = [2021]

    pop = read_into_pam(trips_csv, years)
    print(f"Loaded to pam:\n{pop.stats}")
    jobs = (
        (pam_to_population, False, "nts_2021.csv"),
        (pam_to_population, True, "nts_2021_home_based.csv"),
        (pam_to_population_no_trips, False, "nts_2021_acts.csv"),
        (pam_to_population_no_trips, True, "nts_2021_acts_home_based.csv"),
    )
    for f, filter_home_based, name in jobs:
        print(f"Filtering {name}")
        df = f(pop, filter_home_based)
        print(f"Writing {df.pid.nunique()} to {name}")
        write_path = write_dir / name
        write_path.parent.mkdir(exist_ok=True)
        df.to_csv(write_path, index=False)


def read_into_pam(path: Union[str, Path], years: list):
    trips = pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "TripID",
            "JourSeq",
            "DayID",
            "IndividualID",
            "HouseholdID",
            "MainMode_B04ID",
            "TripPurpFrom_B01ID",
            "TripPurpTo_B01ID",
            "TripStart",
            "TripEnd",
            "TripOrigGOR_B02ID",
            "TripDestGOR_B02ID",
            "W5",
            "SurveyYear",
        ],
    )

    trips = trips.rename(
        columns={  # rename data
            "TripID": "tid",
            "JourSeq": "seq",
            "DayID": "day",
            "IndividualID": "iid",
            "HouseholdID": "hid",
            "TripOrigGOR_B02ID": "ozone",
            "TripDestGOR_B02ID": "dzone",
            "TripPurpFrom_B01ID": "oact",
            "TripPurpTo_B01ID": "dact",
            "MainMode_B04ID": "mode",
            "TripStart": "tst",
            "TripEnd": "tet",
            "W5": "freq",
            "SurveyYear": "year",
        }
    )

    trips = trips[trips.year.isin(years)]

    trips.tst = pd.to_numeric(trips.tst, errors="coerce")
    trips.tet = pd.to_numeric(trips.tet, errors="coerce")
    trips.ozone = pd.to_numeric(trips.ozone, errors="coerce")
    trips.dzone = pd.to_numeric(trips.dzone, errors="coerce")
    trips.freq = pd.to_numeric(trips.freq, errors="coerce")

    trips["did"] = trips.groupby("iid")["day"].transform(
        lambda x: pd.factorize(x)[0] + 1
    )
    trips["pid"] = [f"{i}-{d}" for i, d in zip(trips.iid, trips.did)]

    trips = trips.loc[
        trips.groupby("pid")
        .filter(lambda x: pd.isnull(x).sum().sum() < 1)
        .index
    ]
    # travel_diaries.freq = travel_diaries.freq / travel_diaries.groupby("iid").day.transform("nunique")
    trips.loc[trips.tet == 0, "tet"] = 1440

    trips = trips.drop(["tid", "iid", "day", "year", "did"], axis=1)

    mode_mapping = {
        1: "walk",
        2: "bike",
        3: "car",  #'Car/van driver'
        4: "car",  #'Car/van driver'
        5: "car",  #'Motorcycle',
        6: "car",  #'Other private transport',
        7: "pt",  # Bus in London',
        8: "pt",  #'Other local bus',
        9: "pt",  #'Non-local bus',
        10: "pt",  #'London Underground',
        11: "pt",  #'Surface Rail',
        12: "car",  #'Taxi/minicab',
        13: "pt",  #'Other public transport',
        -10: "DEAD",
        -8: "NA",
    }

    purp_mapping = {
        1: "work",
        2: "work",  #'In course of work',
        3: "education",
        4: "shop",  #'Food shopping',
        5: "shop",  #'Non food shopping',
        6: "medical",  #'Personal business medical',
        7: "other",  #'Personal business eat/drink',
        8: "other",  #'Personal business other',
        9: "other",  #'Eat/drink with friends',
        10: "visit",  #'Visit friends',
        11: "other",  #'Other social',
        12: "other",  #'Entertain/ public activity',
        13: "other",  #'Sport: participate',
        14: "home",  #'Holiday: base',
        15: "other",  #'Day trip/just walk',
        16: "other",  #'Other non-escort',
        17: "escort",  #'Escort home',
        18: "escort",  #'Escort work',
        19: "escort",  #'Escort in course of work',
        20: "escort",  #'Escort education',
        21: "escort",  #'Escort shopping/personal business',
        22: "escort",  #'Other escort',
        23: "home",  #'Home',
        -10: "DEAD",
        -8: "NA",
    }

    trips["mode"] = trips["mode"].map(mode_mapping)
    trips["oact"] = trips["oact"].map(purp_mapping)
    trips["dact"] = trips["dact"].map(purp_mapping)
    trips.tst = trips.tst.astype(int)
    trips.tet = trips.tet.astype(int)

    pam_population = read.load_travel_diary(
        trips=trips, trip_freq_as_person_freq=True
    )
    pam_population.fix_plans()
    return pam_population


def dt_to_min(dt) -> int:
    h, m, s = datetime_to_matsim_time(dt).split(":")
    return (int(h) * 60) + int(m)


def is_home_based(plan: Plan) -> bool:
    return plan[0].act == "home" and plan[-1].act == "home"


def pam_to_population(
    population: Population, filter_home_based: bool = True
) -> pd.DataFrame:
    """write trace of population. Including trips."""
    record = []
    for uid, (_, _, person) in enumerate(population.people()):
        if not filter_home_based or is_home_based(person.plan):
            for component in person.plan:
                if isinstance(component, Leg):
                    act = "trip"
                else:
                    act = component.act
                record.append(
                    [
                        uid,
                        act,
                        dt_to_min(component.start_time),
                        dt_to_min(component.end_time),
                    ]
                )
    df = pd.DataFrame(record, columns=["pid", "act", "start", "end"])
    df["duration"] = df.end - df.start
    return df


def pam_to_population_no_trips(
    population: Population, filter_home_based: bool = True
) -> pd.DataFrame:
    """write trace of population. Ignoring trips."""
    record = []
    for uid, (_, _, person) in enumerate(population.people()):
        if not filter_home_based or is_home_based(person.plan):
            for i in range(0, len(person.plan) - 1, 2):
                record.append(
                    [
                        uid,
                        person.plan[i].act,
                        dt_to_min(person.plan[i].start_time),
                        dt_to_min(person.plan[i + 1].end_time),
                    ]
                )
            record.append(
                [
                    uid,
                    person.plan[-1].act,
                    dt_to_min(person.plan[-1].start_time),
                    dt_to_min(person.plan[-1].end_time),
                ]
            )

    df = pd.DataFrame(record, columns=["pid", "act", "start", "end"])
    df["duration"] = df.end - df.start
    return df


if __name__ == "__main__":
    run()
