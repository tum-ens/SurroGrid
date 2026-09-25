# GridExpand (CLI + web service) in one image. Build from GridExpand/:
#   docker build -f docker/Dockerfile -t gridexpand .
# Nothing private is baked in: the database credentials (.env), the runtime folder (work/) and the
# large untracked inputs (elec_lps.h5, mobility pools) are mounted at run time (see compose.yaml).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never

WORKDIR /app
# Dependencies first (cached layer), then the project itself. All scientific packages ship
# manylinux wheels with their native libraries (GDAL, PROJ, GEOS, HDF5, libpq, HiGHS, Gurobi).
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project --extra service
COPY . .
RUN uv sync --locked --no-dev --extra service
# TEASER opens its shipped JSON input files with mode "r+" although it only reads them; the
# container runs as the caller's uid, so these files (and only these) must be writable.
RUN find /app/.venv -path "*/teaser/data/input/inputdata/*.json" -exec chmod a+w {} +

# Runtime locations (gridexpand.paths): work/ and .env are mounted; caches go to /tmp so the
# container can run as any user id.
ENV PATH="/app/.venv/bin:${PATH}" \
    GRIDEXPAND_WORK_DIR=/work \
    GRIDEXPAND_ENV_FILE=/run/gridexpand/.env \
    HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba \
    PYTHONUNBUFFERED=1
EXPOSE 18766
# Binds 127.0.0.1: run the container with host networking (see compose.yaml) so the service
# stays local and a database on the host is reachable as for the CLI.
CMD ["gridexpand", "serve", "--host", "127.0.0.1", "--port", "18766"]
