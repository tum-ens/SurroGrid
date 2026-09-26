"""GridExpand HTTP API for the GridPlanner UI: pipeline jobs, result queries, scenario files.

``gridexpand api`` starts it (see :mod:`gridexpand.api.cli`). The API is a thin shell around
the command line: every pipeline job runs ``gridexpand run`` as a subprocess, and all read
endpoints are plain SQL on the ``surrogrid`` and ``pylovo`` tables. It needs the optional
``api`` extra (FastAPI, uvicorn). The browser UI lives in the GridPlanner repository.

Contract: ``API_VERSION`` (reported by ``/api/health``) changes only with a breaking change
of the ``/api`` routes; ``docs/openapi.json`` is the committed schema (see
:mod:`gridexpand.api.contract`).
"""

API_VERSION = 1  # bump on incompatible changes of the /api routes (the UI refuses unknown versions)
CSRF_HEADER = "X-GridExpand-UI"
