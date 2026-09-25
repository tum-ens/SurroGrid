"""Extract the solved model entities and write them to the result HDF5 file."""

import warnings

import pandas as pd

from .pyomoio import get_entity, list_entities

HDF_OPTIONS = {"complib": "blosc", "complevel": 9}


def create_result_cache(prob):
    """Return ``{name: Series}`` for every set, parameter, variable and expression."""
    entity_types = ['set', 'par', 'var', 'exp']
    if hasattr(prob, 'dual'):
        entity_types.append('con')

    entities = []
    for entity_type in entity_types:
        entities.extend(list_entities(prob, entity_type).index.tolist())
    return {entity: get_entity(prob, entity) for entity in entities}


def _same_structure(values):
    first = values[0]
    return all(
        isinstance(value, pd.Series)
        and value.index.nlevels == first.index.nlevels
        and list(value.index.names) == list(first.index.names)
        and value.dtype == first.dtype
        and value.name == first.name
        for value in values
    )


def _merge_entity(name, values):
    """Combine one entity over the clusters exactly as the former pairwise loop did.

    Leading empty results are skipped, ``costs`` are summed in cluster order, all
    other entities are concatenated in cluster order (one ``pd.concat`` when the
    parts share their structure).
    """
    first = next((i for i, value in enumerate(values) if not value.empty), None)
    if first is None:
        return values[-1]
    merged = values[first]
    rest = values[first + 1:]
    if name == 'costs':
        for value in rest:
            merged += value
        return merged
    if not rest:
        return merged
    parts = [merged, *rest]
    if all(not value.empty for value in rest) and _same_structure(parts):
        return pd.concat(parts)
    for value in rest:
        merged = pd.concat([merged, value])
    return merged


def _drop_duplicate_rows(value):
    # Site-indexed entities are disjoint across clusters; only the entities
    # without a site level (dt, weight, ...) repeat once per cluster.
    if 'sit' in list(getattr(value.index, 'names', []) or []):
        return value
    try:
        return value[~value.index.duplicated(keep='first')]
    except Exception:
        return value


def merge_cluster_results(cluster_results):
    """Merge the result caches of all clusters (given in cluster order)."""
    parts = {}
    for cache in cluster_results:
        for name, value in cache.items():
            parts.setdefault(name, []).append(value)
    return {
        name: _drop_duplicate_rows(_merge_entity(name, values))
        for name, values in parts.items()
    }


def save_reduced_data(data, save_file_name):
    """Write the (possibly reduced) urbs input tables to ``urbs_out/reduced_data``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        with pd.HDFStore(save_file_name, mode='a', **HDF_OPTIONS) as store:
            for name, table in data.items():
                store['urbs_out/reduced_data/' + name] = table


def save(data, results, save_file_name, solver_audit=None):
    """Write inputs (``reduced_data``), merged results (``MILP``) and the solver audit.

    Args:
        data: urbs input dict of the whole grid.
        results: merged entity results (see ``merge_cluster_results``).
        save_file_name: result HDF5 file (appended to).
        solver_audit: optional DataFrame with one row per cluster solve.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        with pd.HDFStore(save_file_name, mode='a', **HDF_OPTIONS) as store:
            for name, table in data.items():
                store['urbs_out/reduced_data/' + name] = table
            for name, value in results.items():
                store['urbs_out/MILP/' + name] = value
            if solver_audit is not None:
                store['urbs_out/solver_audit'] = solver_audit
