import filecmp
import json
import glob
import os
import shutil
from src.external.emobpy.constants import (
    CWD,
    DEFAULT_DATA_DIR,
    USER_PATH,
    MODULE_DATA_PATH,
    WEATHER_FILES,
)
from .logger import get_logger

logger = get_logger(__name__)


def _find_files_oswalk(root, templates_search_dir, search=None):
    """
    Finds all files via os.walk.

    Args:
        root (str): Root path.
        templates_search_dir (str): Dictionary of the templates.
        search ([type], optional): Search index. Defaults to None.

    Yields:
        file, root, section, basefile, exists
    """
    for file in glob.glob(os.path.join(root, templates_search_dir, f"**/*.*"), recursive=True):
        exists = False
        section = ""
        basefile = os.path.basename(file)
        inbetween = file[len(root) + 1: -len(basefile) - 1].split(os.sep)
        newsection = inbetween[:]
        if search is not None:
            if search in inbetween:
                exists = True
                newsection.remove(search)
            if templates_search_dir in inbetween:
                newsection.remove(templates_search_dir)
        if newsection:
            section = os.path.join(*newsection)
        yield file, root, section, basefile, exists


def _overwrite_sys_files_in_user_data_dir(location=None):
    """
    Overwrites existing sys files in user data directory, with new ones. This is necessary if those files are changing
    in future emobpy update.
    Args:
        location (str, optional): Location to which files should be copied. Defaults to None.
    """
    user_dir = location or USER_PATH or DEFAULT_DATA_DIR
    sys_files = os.path.join(user_dir,"sys_files")

    # Create folders if they do not exist
    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(os.path.join(user_dir,"user_files"), exist_ok=True)
    os.makedirs(os.path.join(user_dir,"sys_files"), exist_ok=True)

    # Copy files to sys_files
    pkg_data_files = glob.glob(os.path.join(MODULE_DATA_PATH, "*.*"))
    for file in pkg_data_files:
        basefile = os.path.basename(file)
        # todo do not change names
        if basefile in WEATHER_FILES.keys():
            rest = file.rsplit(MODULE_DATA_PATH)[-1][1:]
            destpath = os.path.join(sys_files, rest.rsplit(basefile)[0][:-1], WEATHER_FILES[basefile])
        else:
            rest = file.rsplit(MODULE_DATA_PATH)[-1][1:]
            destpath = os.path.join(sys_files, rest)

        # Check if file has been updated and only copy if so
        try:
            file_identical = filecmp.cmp(file, destpath)
        except FileNotFoundError:
            file_identical = False
        if not file_identical:
            shutil.copyfile(file, destpath)
            logger.info(f"New or updated sysfile found and copied to: {destpath}")


def _merge_user_files_with_sys_files(filenames=[], location=None):
    """
    Merges user generated files with sys files to make both available for emobpy.

    Args:
        filenames (list, optional): Specify which files should be merged. Defaults to [].
        location (string, optional): Path to sys data. Defaults to None.

    Raises:
        Exception: Raises exepction if data type is not known and could not be merged.
    """
    user_dir = location or USER_PATH or DEFAULT_DATA_DIR
    sys_data_files = glob.glob(os.path.join(user_dir,'sys_files', "*.*"))
    sys_basefiles = [os.path.basename(file) for file in sys_data_files]
    # Delete all files but keep the folders if they exist
    for file in os.listdir(user_dir):
        pathf = os.path.join(user_dir, file)
        if os.path.isfile(pathf):
            if file in sys_basefiles:
                try:
                    os.remove(pathf)
                    logger.debug(f"Deleted file: {pathf}")
                except FileNotFoundError:
                    logger.debug(f"File {pathf} not found, deleting failed.")
                    pass
                except PermissionError:
                    logger.debug(f"Permission denied to delete file: {pathf}")
                    pass


    # Loop all files in user_files folder
    for user_file in os.listdir(os.path.join(user_dir,"user_files")):

        # Merge files depending on type
        if not filenames or user_file in filenames:

            # Handle json files
            if user_file.split(".")[-1].lower() == "json":
                def get_keys_for_lowest_dict_layer(data_dict):
                    key_lists = []

                    def inner_help_func(dict_obj, key_list=[], last_call=False):
                        for key, value in dict_obj.items():
                            if isinstance(value, dict):
                                new_key_list = key_list + [key]
                                inner_help_func(value, new_key_list)
                            else:
                                new_key_list = key_list + [key]
                                key_lists.append(tuple(new_key_list))

                    inner_help_func(data_dict)
                    return key_lists

                def nested_dict_get(data_dict, key_list):
                    for key in key_list:
                        data_dict = data_dict[key]
                    return data_dict

                def nested_dict_set(data_dict, key_list, value):
                    for key in key_list[:-1]:
                        data_dict = data_dict.setdefault(key, {})
                    data_dict[key_list[-1]] = value

                # Load files
                with open(os.path.join(user_dir,'user_files',user_file)) as f:
                    user_data = json.load(f)
                with open(os.path.join(user_dir,'sys_files',user_file)) as f:
                    sys_data = json.load(f)

                # Merge dicts
                key_lists = get_keys_for_lowest_dict_layer(user_data)
                for key_list in key_lists:
                    value = nested_dict_get(user_data, key_list)
                    nested_dict_set(sys_data, key_list, value)

                # Save to main folder
                with open(os.path.join(user_dir, user_file), 'w') as f:
                    json.dump(sys_data, f, indent=4)
                logger.info(user_file)

            elif user_file in WEATHER_FILES.values():

                shutil.copyfile(os.path.join(user_dir,'user_files',user_file), os.path.join(user_dir, user_file))

            elif user_file in ["md5sums.txt", "urls.txt"]:
                continue

            # If file could not be handled raise error
            else:
                raise Exception(f"Can not handle user_file {user_file}, because data type is unknown. ")

    # Copy all files which are not changed
    for sys_file in os.listdir(os.path.join(user_dir, "sys_files")):
        if sys_file not in os.listdir(user_dir):
            shutil.copy(os.path.join(user_dir,"sys_files", sys_file), user_dir)


def copy_to_user_data_dir():
    """
    This function combines the generation of user data directory and marges the user-defined data
    This usually runs when we create a project folder from command line.
    If we install emobpy and do not run the funtion create_project then it is likely that 'emobpy
    user data folder' has not been created and an error will ocour when using Consumption class.
    Check out the folder in 
    ### linux:   ~/.local/share/emobpy
    ### Windows: C:/Users/<USER>/AppData/Roaming/emobpy
    if embopy user data folder does not exist then run this function:
    ``from emobpy.init import copy_to_user_data_dir; copy_to_user_data_dir()``
    """
    _overwrite_sys_files_in_user_data_dir()
    _merge_user_files_with_sys_files()


def create_project(project_name, template):
    """
    Creates project based on selected template and copies these files.

    Args:
        project_name (str): Chosen project name.
        template (str): Chosen template.

    Raises:
        Exception: Template arguments not valid.
        Exception: Chosen folder does not exist.
    """
    
    if os.path.isdir(os.path.join(MODULE_DATA_PATH,template)):
        pass
    else:
        raise Exception(f"--template argument '{template}' not in {MODULE_DATA_PATH}")

    template_dir_path = os.path.join(MODULE_DATA_PATH, template)
    if not os.path.exists(template_dir_path):
        msg = "from emobpy.init import copy_to_user_data_dir; copy_to_user_data_dir()"
        raise Exception(
            f"Directory '{template_dir_path}' does not exist. Make sure you call copy_to_user_data_dir function first '{msg}'"
        )

    logger.info(f"Copy files from {template_dir_path}")
    for file, _, section, basefile, _ in _find_files_oswalk(template_dir_path, ""):
        destination_file_abspath = os.path.join(CWD, project_name, section, basefile)
        os.makedirs(os.path.split(destination_file_abspath)[0], exist_ok=True)
        shutil.copyfile(file, destination_file_abspath)
        logger.info(f"   {destination_file_abspath}")
    logger.info("Done!")


