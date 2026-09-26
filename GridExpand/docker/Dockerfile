# GridExpand (CLI + web service) in one image. Build from GridExpand/:
#   docker build -f docker/Dockerfile -t gridexpand .
# Nothing private is baked in: the database credentials (DB_* environment variables, or a .env
# mounted at $GRIDEXPAND_ENV_FILE), the runtime folder (work/) and the large untracked inputs
# (elec_lps.h5, mobility pools) come at run time (compose.yaml; GridPlanner). CI publishes the image
# as ghcr.io/tum-ens/gridexpand (.github/workflows/gridexpand-image.yml).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never

WORKDIR /app
# Dependencies first (cached layer), then the project itself. All scientific packages ship
# manylinux wheels with their native libraries (GDAL, PROJ, GEOS, HDF5, libpq, HiGHS, Gurobi).
# uv's download cache stays out of the image (a build cache mount; ~1 GB).
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project --extra service
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --extra service
# TEASER opens its shipped JSON input files with mode "r+" although it only reads them; the
# container runs as the caller's uid, so these files (and only these) must be writable.
RUN find /app/.venv -path "*/teaser/data/input/inputdata/*.json" -exec chmod a+w {} +

# Runtime locations (gridexpand.paths): work/ is mounted, .env optional (without it the DB_*
# variables of the environment are used); caches go to /tmp so the container can run as any uid.
ENV PATH="/app/.venv/bin:${PATH}" \
    GRIDEXPAND_WORK_DIR=/work \
    GRIDEXPAND_ENV_FILE=/run/gridexpand/.env \
    HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba \
    PYTHONUNBUFFERED=1
EXPOSE 18766
# Binds 127.0.0.1: run the container with host networking (see compose.yaml) so the service
# stays local and a database on the host is reachable as for the CLI.
CMD ["gridexpand", "serve", "--host", "127.0.0.1", "--port", "18766"]
