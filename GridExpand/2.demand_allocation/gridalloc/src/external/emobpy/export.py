"""
This module contains a class that can be used to export results to CSV file in a format that is 
useful for modelling BEV in the power system model DIETER and DIETERpy https://diw-evu.gitlab.io/dieter_public/dieterpy.

The documentation contains examples of Export class https://diw-evu.gitlab.io/emobpy/emobpy

"""

import pandas as pd
import numpy as np
import os
from src.external.emobpy.tools import check_for_new_function_name, display_all
from src.external.emobpy.init import copy_to_user_data_dir
from src.external.emobpy.logger import get_logger

logger = get_logger(__name__)


class Export:
    """
    exp = Export()
    exp.loaddata(db), where db is an instance of DataBase Class.
    exp.to_csv()
    exp.save_files('csv')
    """

    def __init__(self, groupbyhr=True, rows=None, kwto="MW"):
        copy_to_user_data_dir()
        super(Export, self).__init__()
        conversion = {"kW": 1, "MW": 1000, "GW": 1000_000}
        self.kwto = kwto
        self.groupbyhr = groupbyhr
        self.row = rows
        self.conversion = conversion[self.kwto]
        try:
            display_all()
        except:
            pass

    def __getattr__(self, item):
        check_for_new_function_name(item)
        # if the return value is not callable, we get TypeError:

    def loaddata(self, db):
        """
        Load data from database into self.data.

        Args:
            db (): Database to be loaded. 
        """
        self.data = db

    def to_csv(self):
        """
        Change object data to csv structure. 
        """
        self.code = []
        self.name = []
        self.consum = []
        self.availcap = []
        self.fleet = pd.DataFrame()

        self.listavailkeys = []
        self.k = 0
        for key in self.data.db.keys():
            if self.data.db[key]["kind"] == "availability":
                self.k += 1
                self.listavailkeys.append(key)

        for i, key in enumerate(self.listavailkeys):
            df = self.data.db[key]["profile"][
                ["hh", "consumption", "charging_cap"]
            ].copy()
            if self.groupbyhr:
                df.loc[:, "hh"] = df["hh"].astype(int)
                dfcon = df.groupby("hh")["consumption"].sum().reset_index()
                dfcha = df.groupby("hh")["charging_cap"].mean().reset_index()
                dfcon.loc[:, "charging_cap"] = dfcha["charging_cap"]
                df = dfcon.copy()
            if self.row is None:
                self.rows = len(df)
            else:
                self.rows = self.row
                df = df[0 : self.rows].copy()
            self.code.append(key)
            self.name.append("ev" + str(i + 1))
            self.consum.append(df["consumption"].values / self.conversion)
            self.availcap.append(df["charging_cap"].values / self.conversion)

            self.fleet.loc[i, "EV"] = "ev" + str(i + 1)
            self.fleet.loc[i, "efficiency_charge"] = self.data.db[key]["charging_eff"]
            self.fleet.loc[i, "efficiency_discharge"] = self.data.db[key][
                "discharging_eff"
            ]
            self.fleet.loc[i, "ev_start"] = self.data.db[key]["soc_init"]
            self.fleet.loc[i, "ev_capacity"] = (
                self.data.db[key]["battery_capacity"] / self.conversion
            )
            self.fleet.loc[i, "share_ev"] = 1 / self.k
            # self.fleet.loc[i, 'cons_rate'] = self.data.db[key][
            #     'consumption'] / self.conversion
            self.fleet.loc[i, "Passed"] = self.data.db[key]["success"]
            self.fleet.loc[i, "info"] = [self.data.db[key]["description"]]
            self.fleet.loc[i, "code"] = key
            self.fleet.loc[i, "ev_end"] = self.data.db[key]["soc_end"]

        self.arr_cons = np.array(self.consum)
        self.arr_avai = np.array(self.availcap)

        self.consdf = pd.DataFrame(self.arr_cons.T, columns=self.name)
        self.consdf.columns = pd.MultiIndex.from_product(
            [["Demand_" + self.kwto + "h"], ["ev_ed"], self.consdf.columns]
        )

        self.avaidf = pd.DataFrame(self.arr_avai.T, columns=self.name)
        self.avaidf.columns = pd.MultiIndex.from_product(
            [["Grid_connect_" + self.kwto], ["n_ev_p"], self.avaidf.columns]
        )
        self.final = self.consdf.join(self.avaidf)

        self.subscen = {}
        self.setopt = set()

        for cd in self.code:
            temp = {}
            for key in self.data.db.keys():
                if self.data.db[key]["kind"] != "driving":
                    if self.data.db[key]["input"] == cd:
                        option = self.data.db[key]["option"]
                        temp[option] = key
                        self.setopt.add(option)
            if temp:
                self.subscen[cd] = temp

        if self.subscen:
            self.optdict = dict(zip(self.setopt, range(len(self.setopt))))
            self.arr_options = np.empty((len(self.setopt), len(self.code), self.rows))
            for id1, cd in enumerate(self.code):
                if self.subscen[cd]:
                    for key, value in self.subscen[cd].items():
                        id0 = self.optdict[key]
                        df = self.data.db[value]["timeseries"][
                            ["hh", "charge_battery"] # charge_battery instead of charge_grid. DIETERpy requirement to
                            # avoid double counting of efficiency
                        ].copy()
                        if self.groupbyhr:
                            df.loc[:, "hh"] = df["hh"].astype(int)
                            df.loc[:, 'charge_battery'] = df["charge_battery"].astype(float)
                            df = df.groupby(["hh"]).mean().reset_index()
                        df = df[0 : self.rows].copy()
                        self.arr_options[id0, id1, :] = (df["charge_battery"].values / self.conversion)
            # self.arr_options = self.arr_options.round(7)
            self.frame = pd.DataFrame(
                index=range(self.rows), columns=pd.MultiIndex.from_product([[], [], []])
            )
            for opt, idopt in self.optdict.items():
                dfop = pd.DataFrame(self.arr_options[idopt].T, columns=self.name)
                dfop.columns = pd.MultiIndex.from_product(
                    [[opt + "_" + self.kwto + "h"], ["ev_ged_exog"], dfop.columns]
                )
                self.frame = self.frame.join(dfop)
            self.final = self.final.join(self.frame)

        self.final["Hour", "-", "-"] = ["h" + str(j + 1) for j in range(self.rows)]
        self.final.set_index(("Hour", "-", "-"), inplace=True)
        self.final.index.name = "Hour"
        self.final = self.final.round(7)

    def save_files(self, repository=""):
        """
        Saves object as an csv. 

        Args:
            repository (str, optional): Path to repository. Defaults to "".
        """
        if repository:
            self.repository = repository
        else:
            self.repository = self.data.folder
        os.makedirs(self.repository, exist_ok=True)
        ts_pth = os.path.join(self.repository, "bev_time_series.csv")
        di_pth = os.path.join(self.repository, "bev_data_input.csv")
        self.fleet.to_csv(di_pth, index=False)
        self.final.to_csv(ts_pth)
        logger.info(f"Summary file: {di_pth}")
        logger.info(f"Time series file: {ts_pth}")
