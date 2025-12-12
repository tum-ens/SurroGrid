import argparse
import time
import os

try:
    from src.external.emobpy.init import (create_project, copy_to_user_data_dir)
    from src.external.emobpy.constants import CWD, MODULE_DATA_PATH
    from src.external.emobpy.tools import display_all
    from src.external.emobpy.logger import get_logger
    logger = get_logger(__name__)
except ImportError as exc:
    raise ImportError(
        "Couldn't import emobpy. Are you sure it's installed and "
        "available on your PYTHONPATH environment variable? Did you "
        "forget to activate a virtual environment?"
    ) from exc


def parser():
    argcollect = argparse.ArgumentParser(description="emobpy command line")
    # add positional argument create_project and run
    argcollect.add_argument(
        "command",
        help='Start with this script "emobpy create -n [here a name]" or "emobpy list".',
        type=str,
    )
    argcollect.add_argument(
        "-n",
        "--name",
        help='Required argument for "create". A project name must be provided.',
        type=str,
    )
    argcollect.add_argument(
        "-t",
        "--template",
        help="""Optional argument for "create". Examples can be selected through templates,
        name of templates are eg1, eg2 ...""",
        type=str,
    )
    argcollect.add_argument(
        "-s",
        "--seed",
        help="""Optional argument for "create". An own seed can be set for the project. 
        Otherwise it will be set automatically. """,
        type=str,
    )

    args = argcollect.parse_args()
    return args


def main():
    arg_option = ["create", "list"] # list of possible commands
    args = parser()
    if not args.command in arg_option:
        raise Exception(f"First positional argument must be {arg_option}")
    if args.command == "create":
        if args.name:
            # Use specific template if given
            if args.template:
                tmpl = args.template
            else:
                tmpl = "base"

            # Use specific seed if given
            seed = int(time.time())
            if args.seed:
                seed = int(args.seed)

            copy_to_user_data_dir()
            create_project(args.name, tmpl)

            # Save seed in seed.txt
            with open(os.path.join(CWD,args.name,'config_files','seed.txt'), 'w') as f:
                logger.info(f"Randomization seed set to {seed}")
                f.write(str(seed))
            
            try:
                display_all()
            except:
                pass

        else:
            raise Exception(
                "create is an argument that must have a project name as -n argument"
            )
    elif args.command == "list":
        logger.info("List of available templates:\n")
        for i in os.listdir(MODULE_DATA_PATH):
            if os.path.isdir(os.path.join(MODULE_DATA_PATH, i)):
                logger.info(i)
        logger.info("\nType 'emobpy create -n [give a name] -t [template]' to create a project folder")


if __name__ == "__main__":
    main()
