"""
This module contains data organisation classes to read, load and edit the resulting time series. 
See also the examples in the documentation https://diw-evu.gitlab.io/emobpy/emobpy

For more details see the article and cite:

.. code-block:: python

    @article{Gaete-Morales_2021,
    author={Gaete-Morales, Carlos and Kramer, Hendrik and Schill, Wolf-Peter and Zerrahn, Alexander},
    title={An open tool for creating battery-electric vehicle time series from empirical data, emobpy},
    journal={Scientific Data}, year={2021}, month={Jun}, day={11}, volume={8}, number={1}, pages={152},
    issn={2052-4463}, doi={10.1038/s41597-021-00932-9}, url={https://doi.org/10.1038/s41597-021-00932-9}}

"""

import time
import pickle
import gzip
import os
import uuid
from src.external.emobpy.tools import (parallelize, check_for_new_function_name, display_all)
from src.external.emobpy.constants import VARIABLES_DB
from src.external.emobpy.logger import get_logger

logger = get_logger(__name__)

class DataBase(object):
    """
    DataBase object useful to manage many.

    important attribute:
    self.db : It is a dictionary that contains all profiles. The dictionary keys are the name of the profile
        Every profile in this dict has nested dictionary. The keys depend on the type of profile.
        Common keys:
            self.db["name of the profile"]["kind"] that can be ["driving", "consumption", "availability", "charging"]
            self.db["name of the profile"]["input"] that is a string only for ["consumption", "availability", "charging"] profiles

    self.__init__(folder)
        folder: path as string of folder where profiles are hosted.
    """
    
    def __init__(self, folder):
        super(DataBase, self).__init__()

        self.name = ''
        self.folder = folder
        self.oldpath = []
        self.db = {}
        try:
            display_all()
        except:
            pass
        

    def __getattr__(self, item):
        check_for_new_function_name(item)

    def loadfiles(self, loaddir=''):
        """
        Load profiles and host in a directory other than the "folder".
        So that directory must be indicated (loaddir). In this way profiles from many directories can be loaded.

        Args:
            loaddir (str, optional): Directory to load from. Defaults to ''.
        """
        if loaddir:
            self.repo = loaddir
        else:
            self.repo = self.folder
        os.makedirs(self.repo, exist_ok=True)
        self.currentpath = [
            f for f in os.listdir(self.repo)
            if os.path.isfile(os.path.join(self.repo, f))
        ]
        self.path_list = list(set(self.currentpath) - set(self.oldpath))
        if self.path_list:
            self.oldpath = self.currentpath

        for f in self.path_list:
            self.fpath = os.path.join(self.repo, f)
            if f.split('.')[-1] == 'pickle':
                self.pickle_off = gzip.open(self.fpath, "rb")
                self.obj = pickle.load(self.pickle_off)
                self.pickle_off.close()
                self.db[self.obj['name']] = self.obj
                del self.pickle_off

    def loadfiles_batch(self, loaddir='', batch=10, nr_workers=4, kind='', add_variables=[]):
        """
        Load datafiles into DataBase object for further usage.

        Args:
            loaddir (str, optional): Directory to load from. Defaults to ''.
            batch (int, optional): Number of batches to load. Defaults to 10.
            nr_workers (int, optional): Number of workers to load. Defaults to 4.
            kind (str, optional): Data kind to load. E.g 'consumption'. Defaults to ''.
            add_variables (list, optional): New variables to load. Defaults to [].
        """
        variables = list(set(VARIABLES_DB[kind] + add_variables))
        if loaddir:
            self.repo = loaddir
        else:
            self.repo = self.folder
        os.makedirs(self.repo, exist_ok=True)
        self.currentpath = [
            f for f in os.listdir(self.repo)
            if os.path.isfile(os.path.join(self.repo, f))
        ]
        # create batch
        i = 0
        nr_files = len(self.currentpath)
        paths_batch = []
        flag = False
        for _ in range(nr_files):
            batch_ = []
            for _ in range(batch):
                if i < nr_files:
                    batch_.append(self.currentpath[i])
                else:
                    flag = True
                    break
                i += 1
            if len(batch_) > 0:
                paths_batch.append(batch_)
            if flag:
                break
        for lt in paths_batch:
            dc = {
                k: {
                    'f': os.path.join(self.repo, v)
                }
                for k, v in enumerate(lt) if v.split('.')[-1] == 'pickle'
            }
            odc = parallelize(self.loadpkl, dc, nr_workers,
                              **dict(variables=variables, kind=kind))
            for j in odc:
                if odc[j][1]:
                    self.db[odc[j][0]['name']] = odc[j][0]

    @staticmethod
    def loadpkl(f, variables, kind):
        """
        Load from pickle file.

        Args:
            f (str): Path to pickle file.
            variables (str): Variables which should be loaded.
            kind ([type]): Data kind to load.

        Returns:
            DataBase: Loaded object.
        """
        pickle_off = gzip.open(f, "rb")
        obj = pickle.load(pickle_off)
        pickle_off.close()
        if obj['kind'] == kind:
            new_obj = {}
            for nm in variables:
                if nm in obj:
                    new_obj[nm] = obj[nm]
            return new_obj, True
        else:
            return {}, False

    def update(self):
        """
        Run self.laodfiles() to load files from database "folder".
        """
        self.loadfiles()

    def getdb(self):
        """
        Run self.loadfiles() and return imported database.

        Returns:
            DataBase: Loaded database object.
        """
        self.update()
        return self.db

    def remove(self, name):
        """
        Remove part of database.

        Args:
            name (str): Key which is to be deleted.
        """
        self.acum = []
        self.db.pop(name, None)
        if os.path.isfile(os.path.join(self.folder, name + '.pickle')):
            os.remove(os.path.join(self.folder, name + '.pickle'))
            self.acum.append(name + '.pickle')
        self.update()
        logger.info('Files deleted:', len(self.acum))
        logger.info(self.acum)
        del self.acum


class DataManager:
    """
    Data Manager to load and save files.
    """
    def __init__(self):
        super(DataManager, self).__init__()

    def __getattr__(self, item):
        check_for_new_function_name(item)
          # if the return value is not callable, we get TypeError:

    def savedb(self, obj, dbdir='db_files'):
        """
        Save database to pickle file.

        Args:
            obj (object): Database to be saved. 
            dbdir (str, optional): Path to database directory. Defaults to 'db_files'.
        """
        obj.update()
        if not obj.name:
            nnn = 'db_' + time.strftime("%Y%m%d_%H%M%S") + '_' + uuid.uuid4(
            ).hex[:5]  # + time.strftime("%Y%m%d_%H%M%S")
            obj.name = nnn[:]
        os.makedirs(dbdir, exist_ok=True)
        with gzip.open(os.path.join(dbdir, obj.name + '.pickle'),
                       'wb') as file:
            pickle.dump(obj, file)
        logger.info(file)
        logger.info('=== Database saved ===')

    def loaddb(self, dbfilepath, profilesdir):
        """
        Load database from pickle file. 

        Args:
            dbfilepath (str): Path to pickle file.
            profilesdir (str): Path to profiles directory.

        Returns:
            object: Loaded database from pickle file.
        """
        with gzip.open(dbfilepath, 'rb') as file:
            obj = pickle.load(file)
        obj.folder = profilesdir
        obj.update()
        return obj
