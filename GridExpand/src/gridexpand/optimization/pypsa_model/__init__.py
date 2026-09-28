"""PyPSA/linopy implementation of the Step 3 building model (the ``pypsa`` optimizer).

Reads the same Step 2 input as the urbs optimizer (``gridexpand.optimization.urbs``),
solves the same model (see ``building_model.py``; no power grid: every building is an
independent energy system) and writes a result file
with the same layout and the ``results.RESULT_KEYS`` entities. Select it with
``gridexpand optimize --optimizer pypsa`` or ``GRIDEXPAND_OPTIMIZER=pypsa``.
"""

from .results import RESULT_KEYS
from .runfunctions import run_pypsa_opt
from .solve import solve_group

__all__ = ["RESULT_KEYS", "run_pypsa_opt", "solve_group"]
