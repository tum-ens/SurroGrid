import pandas as pd
from pyomo.environ import SolverFactory
import urbs
from .model import create_model
from .report import *
from .plot import *
from .input import *
from .validation import *
from .saveload import *
from .features import *
from .scenarios import *
import os
import multiprocessing as mp
import time
import warnings

def run_worker(data_cluster, global_settings, log_dir, scenario_name, return_dict=None, i=""):
    """ run 
    Args:
        - cluster: assigned list of building nodes to solve (parallelized by parent process)
        - data: data to extract clusters from
        - global_settings: dict with global settings
        - result_dir: directory in which results should be saved 
        - scenario_name: scenario name string
        - return_dict: shared dictionary among parent and child processes in which data is written
        - i: which index to save results in in return dict

    Returns:
        dataframe dict with results
    """   
    ###################### Setup pyomo model instance + solver ###########################
    start_time=time.time()   # Timer for measuring model setup duration

    ### returns pyomo model instance (objective, constraints, parameters, variables, expressions) 
    # IMPLEMENT: no fixed costs in output
    prob_cluster = create_model(data_cluster,               # selected model data subset
                                global_settings)            # settings to apply


    ### Setup solver
    optim = SolverFactory(global_settings["solver_name"])   # Create solver
    # logfile = repr(os.path.join(result_dir, f'{scenario_name}_{i}.log'))
    logfile = repr(f'{log_dir}/{global_settings["input_file"][:-3]}_{scenario_name}_{i}.log')
    if os.path.exists(logfile): os.remove(logfile)
    optim = setup_solver_mip(optim,                # solver instance
                             logfile = logfile)    # logfile save path + name
    
    end_time=time.time()    # Timer for measuring model setup duration
    print(f"Model setup {i} took {(end_time-start_time)/60:.2f} minutes to run!")
    

    ##################### Solve the pyomo model instance ##########################
    start_time=time.time()     # Timer for measuring model solve duration

    result = optim.solve(prob_cluster, tee=False, report_timing=False)
    if str(result.solver.termination_condition) == 'infeasibleOrUnbounded': raise ValueError(f"Problem infeasible or unbounded!")
    
    end_time=time.time()       # Timer for measuring model solve duration
    print(f"Model solve {i} took {(end_time-start_time)/60:.2f} minutes to run")

    
    ################### Extract and return results ####################
    model_results = create_result_cache(prob_cluster)

    if global_settings["parallel"]:
        return_dict[i] = model_results   # Insert this workers pyomo model instance
        return None
    else:
        return model_results
   

# def prepare_result_directory(input_file, script_name):
#     # timestamp for result directory
#     now = datetime.now().strftime('%Y%m%dT%H%MS%S%f')

#     # create result directory if not existent
#     result_dir = os.path.join('result', '{}-{}-{}'.format(now, input_file, script_name))
#     if not os.path.exists(result_dir): os.makedirs(result_dir)

#     return result_dir

def prepare_result_directory(input_file, script_name):
    return "result"


def setup_solver_mip(optim, logfile='solver.log'):
    ### Gurobi settings
    if optim.name == 'gurobi': # http://www.gurobi.com/documentation/5.6/reference-manual/parameters
        optim.set_options(f"logfile={logfile}")
        optim.set_options("Method=4")                 # -1 automatic, 0 Primal Simplex, 1 Dual Simplex, 2 Barrier Method, 3 Concurrent Optimization, 4 Deterministic Concurrent
        optim.set_options("MIPFocus=2")               # 0 default balance, 1 Feasibility focus, 2 Optimality focus, 3 Bound tightening focus
        optim.set_options("MIPGap=0.05")
        optim.set_options("Presolve=2")
        optim.set_options("Threads=4")

    ### Other solvers:
    if optim.name == 'cplexdirect' or optim.name == 'cplex_direct':
        optim.options['threads'] = 32
        optim.options['mip_tolerances_mipgap'] = 0.05

    return optim


def run_lvds_opt(input_path,        # path to input file  
                 result_path,        # path to output directory
                 result_dir,
                 global_settings):  # global input settings
    """ Run an urbs model for given input path, result directory and global settings
    
    Args:
        - input_path: path string to input file which is used to setup model
        - result_dir: path string to output directory where model results are saved
        - global_settings: dictionary including user adjustable global run settings
    """
    ### Ignore selected warnings:
    start_time = time.time()
    warnings.filterwarnings("ignore", category=FutureWarning, module="tsam.timeseriesaggregation")

    ################# Extract and modify input data ###################
    ### Add additional input settings: ###
    forced_settings = {
        # only change if you know what you are doing
        "timesteps": range(0,8761), # possible timestep values + 1 (for storage initialization at time=0)
        "dt": 1,                    # length of time steps in hours
        "solver_name": "gurobi",    # "gurobi"  # current code optimized for gurobi, might need to adjust hyperparameters down the pipeline
        "parallel": True,           # True      # makes no sense to not be parallel anymore, as then just increased computation time for building models 

        # Currently not implemented
        "vartariff": 0,         # 0             # % of building nodes opting into a variable tariff with low, normal, high pricing (0-100)
        "power_price_kw": 0     # 0             # % additional pricing proportional to imported/feed-in power (€/kW)
    }
    global_settings.update(forced_settings)


    ### Read out, validate and modify input_file data: ###
    print("Reading and validating input data...")
    if os.path.splitext(global_settings["input_file"])[1] == ".h5": data = read_input_h5(input_path)
    else: data = read_input(input_path)    # standard Excel readout: read out support timeframes, commodities, commodity-process, processes, demand, weight typeperiod, intermediate supply, transmission, storage, DSM, buy-sell-price, timevareff, availability, uhp
    # validate_input(data)          # check vertex rules, no duplicates, infeasible capacities, check if sites/processes/commodities/storage/dsm if present in each others sheets 


    ### Insert settings into data and read out modes/name: ###
    print("\nReading running modes...")
    scenario_name = urbs.read_scenario_name(global_settings) # from global settings create a filename
    data = urbs.insert_scenario(data, global_settings)       # insert global settings as df into input data

    mode = identify_mode(data)   # check whether intertemporal, transmission, storage, dsm, bsp, tve, availability, acpf/dcpf, type period weight, tsam, tsam season, onoff, minfraction, power_price, uncoordinated, transdist, 14a, uhp
    print(f"Identified running modes: {mode}")               # for us should be present: sto, bsp, tve, ava, tsam, exp(pro, sto-c, sto-p), uncoordinated

    end_time = time.time()
    print(f"Preprocesssing took {(end_time-start_time)/60:.2f} minutes to run!\n")


    ######################### Apply input settings ###################################
    ##### Conduct time series aggregation (TSAM) #####
    if mode["tsam"]: 
        print("Running time series aggregation (TSAM)...")
        start_time=time.time()  # Timer for measuring tsam duration
        # run timeseries aggregation method before creating model (to reduce computational load by reducing considered weeks)
        data, global_settings["timesteps"], tsam_data = run_tsam(data,
                                                                global_settings["noTypicalPeriods"],
                                                                global_settings["hoursPerPeriod"])  
        end_time = time.time()  # Timer for measuring tsam duration
        if mode["tsam"]: print(f"TSAM took {(end_time-start_time)/60:.2f} minutes to run!\n") 
    else: # tsam disabled, just filter the time series according to the defined time steps
        data, tsam_data = select_predefined_timesteps(data, global_settings["timesteps"])

    ###### Safe tsam data ######
    with pd.HDFStore(result_path, mode='a', complib='blosc', complevel=9) as store:
        for name in tsam_data.keys():
            store['urbs_out/tsam/' + name] = tsam_data[name]


    ##### Electrification: #####
    if global_settings["PV_electr"] < 100: data = remove_pv_in_random(data, global_settings["PV_electr"])
    if global_settings["HP_electr"] < 100: data = remove_heat_in_random(data, global_settings["HP_electr"])
    if global_settings["EV_electr"] < 100: data = remove_mobility_in_random(data, global_settings["EV_electr"])

    ##### Variable Tariff: #####
    if global_settings["vartariff"]!=0: raise NotImplementedError("Variable Tariff: Any values different from 0 are currently not safely implemented!") 

    ##### Power Price: #####
    if global_settings["power_price_kw"]!=0: raise NotImplementedError("Power price: Any values different from 0 are currently not safely implemented!") 


    ############### Carry out building optimization ###############
    # Launch parallel processes equal to previously given thread count (= cpu_count)
    print("Setting up and running parallel pyomo models...")
    time_A=time.time()

    model_results = {}
    if global_settings["parallel"]:
        ### Assign building nodes to paralelly solved threads: ###
        print(f"Parallelize building nodes...")
        clusters = get_parallel_building_clusters(data, global_settings["n_cpu"])

        procs = []                     # list constaining the parallel processes
        manager = mp.Manager()         # parallel process manager
        return_dict = manager.dict()   # shared dictionary among parent and daughter processes in which date can be written

        for i, cluster in enumerate(clusters):
            print(i)
            data_cluster = get_cluster_data(data, cluster)  # only data for selected buildings for this thread

            proc = mp.Process(target=run_worker,
                                args=(data_cluster,
                                    global_settings,        # settings of the run
                                    "logs/gurobi",                 # output directory in which to save logfiles
                                    scenario_name,          # name of the scenario for saving files
                                    return_dict,            # shared dict in which to save data 
                                    i))                     # location in dict in which to save    
            procs.append(proc)
            proc.start()

        for i, proc in enumerate(procs):
            ### Retrieve data from child processes
            proc.join()
            model_results[i] = return_dict[i]  # solved model instances
    else:
        model_results = run_worker(data,                   # whole data
                                global_settings,        # settings of the run
                                "logs",             # output directory in which to save logfiles
                                scenario_name)          # name of the scenario for saving files

    time_B=time.time()
    print(f"Solving process took {(time_B-time_A)/60:.2f} minutes to run!\n")


    ################ Save optimization results ###############
    start_time=time.time()      # Timer for measuring saving durations

    save(data, model_results, result_path, global_settings["parallel"])
    file_name, file_extension = os.path.splitext(result_path)
    new_result_path = f"{file_name}_{scenario_name}{file_extension}"
    if os.path.exists(new_result_path): os.remove(new_result_path)
    os.rename(result_path, new_result_path)

    end_time=time.time()        # Timer to measure duration of model results saving
    print(f"Model results saving took {(end_time-start_time)/60:.2f} minutes\n")


    ############## Exit ###############
    return None