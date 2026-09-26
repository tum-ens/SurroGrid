"""SurroGrid PostgreSQL access: ``SurroGridDatabase`` facade and schema helpers.

Modules: ``engine`` (connection settings, cached engines), ``schema``
(migrations, views), ``grids`` (pylovo grids and grid cases), ``runs``
(scenario and run rows), ``writers`` (COPY writers), ``maintenance``
(``gridexpand db`` commands).
"""

from gridexpand.db.database import SurroGridDatabase, normalize_ags
from gridexpand.db.engine import DatabaseNotConfigured, get_engine
from gridexpand.db.schema import SchemaMigrationRequired, ensure_schema, refresh_qgis_views

__all__ = [
    "DatabaseNotConfigured",
    "SchemaMigrationRequired",
    "SurroGridDatabase",
    "ensure_schema",
    "get_engine",
    "normalize_ags",
    "refresh_qgis_views",
]
