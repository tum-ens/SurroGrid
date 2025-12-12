"""Configuration for Step 1 (grid sampling).

Most settings are simple constants. Database credentials are expected as
environment variables and are commonly provided via a local `.env` file in
`gridreadout/`.
"""

import os
from dotenv import load_dotenv

# Load .env file for environment-specific settings (DB credentials, etc.)
load_dotenv(override=True)

class Config:
    #--------------------------------------------------------------#
    #---------------- Database/API Connections --------------------#
    #--------------------------------------------------------------#
    # Pylovo Database settings
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = int(os.getenv('DB_PORT', 5432))
    DB_NAME = os.getenv('DB_NAME')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')

    # PVGIS API
    PVGIS_URL = "https://re.jrc.ec.europa.eu/api/tmy"               # URL from which to fetch typical meterological year weather data
    REF_YEAR = 2009                                                 # Reference year on which whole simulation is based (in terms of holidays!, 2009 is choosen as the real electricity data matches its holidays)
    TIME_ZONE = 1   # Currently only implemented for UTC+1!!!       # Shift between the location's time and GMT in hours. CET would be 1.

    # OpenMeteo API
    OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive" # URL from which to fetch soil temperature data
    OPENMETEO_TIME_ZONE = "UTC+01:00"  # Currently only  UTC+1!!!   # Timezone at location (for alignment of weather with human actions)

    #--------------------------------------------------------------#
    #----------------- Paths/Dataset Readout ----------------------#
    #--------------------------------------------------------------#
    # Important paths
    STORAGE_DIR = "results"                # Directory in which to store results
    PYLOVO_COORD_FORMAT = "EPSG:3035"
    TARGET_COORD_FORMAT = "EPSG:4326"

config = Config()