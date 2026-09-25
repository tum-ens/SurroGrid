"""Point gridexpand at an unreachable database before any test imports it.

``gridexpand.paths`` reads ``GRIDEXPAND_ENV_FILE`` and ``GRIDEXPAND_WORK_DIR``
when it is first imported, so they are set here, at conftest import time.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

UNREACHABLE_ENV = """\
DB_NAME="gridexpand_test_unreachable"
DB_USER="nobody"
DB_PASSWORD="nobody"
DB_HOST="127.0.0.1"
DB_PORT="9"
PYLOVO_VERSION_ID="1"
"""

_ROOT = Path(tempfile.mkdtemp(prefix="gridexpand-tests-"))
atexit.register(shutil.rmtree, _ROOT, ignore_errors=True)
(_ROOT / "unreachable.env").write_text(UNREACHABLE_ENV, encoding="utf-8")
os.environ["GRIDEXPAND_ENV_FILE"] = str(_ROOT / "unreachable.env")
os.environ["GRIDEXPAND_WORK_DIR"] = str(_ROOT / "work")
