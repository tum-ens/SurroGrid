import pandas as pd
from .pyomoio import get_entity, list_entities
import warnings

def rename_duplicate_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

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
        results_all = dict.fromkeys(list(model_results[0].keys()), pd.DataFrame()) # initialize results with empty dataframes
        for model_res in model_results.values():            
            for name in model_res.keys():
                if (name == 'costs') and not results_all[name].empty:          # only if costs and if cost result frame already has some costs inserted
                    results_all[name] += model_res[name]                       # for costs, add them, not concat
                else: # all other results -> concatenate
                    if results_all[name].empty: 
                        results_all[name] = model_res[name]             # if empty assign
                    else: results_all[name] = pd.concat([results_all[name], model_res[name]])   # else concat

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

class ResultContainer(object):
    """ Result/input data container for reporting functions. """
    def __init__(self, data, result):
        self._data = data
        self._result = result


def load(filename):
    """Load a urbs model result container from a HDF5 store file.

    Args:
        filename: an existing HDF5 store file

    Returns:
        prob: the modified instance containing the result cache
    """
    with pd.HDFStore(filename, mode='r') as store:
        data_cache = {}
        for group in store.get_node('data'):
            data_cache[group._v_name] = store[group._v_pathname]

        result_cache = {}
        for group in store.get_node('result'):
            result_cache[group._v_name] = store[group._v_pathname]

    return ResultContainer(data_cache, result_cache)
