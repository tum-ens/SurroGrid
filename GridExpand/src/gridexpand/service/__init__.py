"""GridExpand web service: job API, result queries and the pylovo-ui plugin panels.

``gridexpand serve`` starts it (see :mod:`gridexpand.service.cli`). The service is a thin
shell around the command line: every pipeline job runs ``gridexpand synthetic`` as a
subprocess, and all read endpoints are plain SQL on the ``surrogrid`` and ``pylovo``
tables. It needs the optional ``service`` extra (FastAPI, uvicorn).
"""

API_VERSION = 1  # bump on incompatible changes of the HTTP API or the plugin manifest
PLUGIN_SCHEMA = 1  # schema of /ui/manifest.json understood by the pylovo-ui plugin host
CSRF_HEADER = "X-GridExpand-UI"
