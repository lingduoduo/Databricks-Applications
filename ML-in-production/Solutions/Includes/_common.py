# Databricks notebook source
# MAGIC %pip install \
# MAGIC git+https://github.com/databricks-academy/dbacademy-gems@e8183eed9481624f25b34436810cf6666b4438c0 \
# MAGIC git+https://github.com/databricks-academy/dbacademy-rest@bc48bdb21810c4fd69d27154bfff2076cc4d02cc \
# MAGIC git+https://github.com/databricks-academy/dbacademy-helper@e0e819d661e6bf972c6fb1872c1ae1a7d2a74b23 \
# MAGIC --quiet --disable-pip-version-check

# COMMAND ----------

# MAGIC %run ./_dataset_index

# COMMAND ----------

import re
from dbacademy_gems import dbgems
from dbacademy_helper import DBAcademyHelper, Paths

helper_arguments = {
    "course_code" : "mlp",             # The abreviated version of the course
    "course_name" : "ml-in-production",      # The full name of the course, hyphenated
    "data_source_name" : "ml-in-production", # Should be the same as the course
    "data_source_version" : "v01",     # New courses would start with 01
    "enable_streaming_support": False, # This couse uses stream and thus needs checkpoint directories
    "install_min_time" : "1 min",      # The minimum amount of time to install the datasets (e.g. from Oregon)
    "install_max_time" : "5 min",      # The maximum amount of time to install the datasets (e.g. from India)
    "remote_files": remote_files,      # The enumerated list of files in the datasets
}

