import pandas as pd
from .pyomoio import get_entity, list_entities
import warnings

def create_result_cache(prob):
    entity_types = ['set', 'par', 'var', 'exp']
    if hasattr(prob, 'dual'):
        entity_types.append('con') # won't have constraint for us

    # list_entities: list of member names for each entitiy_type (set, par, ...) where columns are (name, doc, multiindex_domain e.g. (tm, stf, sit, com)) 
    entities = []
    for entity_type in entity_types:
        entities.extend(list_entities(prob, entity_type).index.tolist())

    # for each entity save model results in result_cache[name]
    result_cache = {}
    for entity in entities:
        result_cache[entity] = get_entity(prob, entity)
    return result_cache


def save_reduced_data(data, save_file_name):
    """Save reduced/filtered urbs input data without optimization results."""
    with pd.HDFStore(save_file_name, mode='a', complib='blosc', complevel=9) as store:
        for name in data.keys():
            if name == "global_prop":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
                    store['urbs_out/reduced_data/' + name] = data[name]
            else:
                warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
                store['urbs_out/reduced_data/' + name] = data[name]


def save(data, model_results, save_file_name, manyprob=False):
    """Save urbs model input and result cache to a HDF5 store file.

    Args:
        - prob:     a urbs model instance containing a solution
        - filename: HDF5 store file to be written
        - manyprob: if prob is defined as a dictionary of Pyomo.ConcreteModel instances instead of a single one

    Returns: None
    """

    ### Normal saving operation if model is not parallelized
    if not manyprob: 
        results_all = model_results
    else: 
        ### Concatenate all results of parallelly run models into one dataframe
        results_all = {}
        for model_res in model_results.values():
            for name, result in model_res.items():
                if name not in results_all or results_all[name].empty:
                    results_all[name] = result
                elif name == 'costs':
                    results_all[name] += result
                else:
                    results_all[name] = pd.concat([results_all[name], result])

    ### save data and results
    with pd.HDFStore(save_file_name, mode='a', complib='blosc', complevel=9) as store:
        # Save data
        for name in data.keys(): 
            if name=="global_prop":                 # For this it is valid to ignore as dataset is really small, otherwise check!
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
                    store['urbs_out/reduced_data/'+name] = data[name]
            else:
                warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
                store['urbs_out/reduced_data/'+name] = data[name]
        # Save results
        for name in results_all.keys():
            try: results_all[name] = results_all[name][~results_all[name].index.duplicated(keep='first')]
            except: pass
            if name in ['dt', 'obj', 'weight']:     # For these it is valid to ignore as datasets are really small, otherwise check!
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
                    store['urbs_out/MILP/'+name] = results_all[name]
            else: 
                store['urbs_out/MILP/'+name] = results_all[name]
