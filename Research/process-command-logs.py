# Databricks notebook source
# MAGIC %pip install pyre-check Jinja2 Pygments

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC # clone pyre-check repo to get pre-defined stubs and configs
# MAGIC git clone https://github.com/facebook/pyre-check /tmp/scanner/pyre-check

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import Window

# COMMAND ----------

# adjust accordingly, could be job parameters

lookback_days = 1
audit_log_table = "log_data.workspace_audit_logs"

# COMMAND ----------

# get all successful notebook commands for the last day
commands = (spark.read.table(audit_log_table)
            .filter(f"serviceName = 'notebook' and actionName in ('runCommand', 'attachNotebook') and date >= current_date() - interval {lookback_days} days")
            .filter("(requestParams.path is not null or requestParams.commandText not like '\%%')"))

# COMMAND ----------

# sessionize based on attach events
sessionized = (commands
               .withColumn("notebook_path", F.when(F.col("actionName") == "attachNotebook", F.col("requestParams.path")).otherwise(None))
               .withColumn("session_started", F.col("actionName") == "attachNotebook")
               .withColumn("session_id", F.sum(F.when(F.col("session_started"), 1).otherwise(0)).over(Window.partitionBy("requestParams.notebookId").orderBy("timestamp")))
               .withColumn("notebook_path", F.first("notebook_path").over(Window.partitionBy("session_id", "requestParams.notebookId").orderBy("timestamp"))))

# COMMAND ----------

combined = (sessionized
            .select(
                "timestamp",
                "notebook_path",
                "userIdentity.email",
                "session_id",
                F.col("requestParams.notebookId").alias("notebook_id"),
                F.col("requestParams.commandId").alias("command_id"),
                F.col("requestParams.commandText").alias("command_text")
            )
            .filter("command_id is not null"))

# COMMAND ----------

pdf_combined = combined.toPandas()

if len(pdf_combined) == 0:
    print("WARNING: No commands found in audit logs!")

# COMMAND ----------

import os

conf_path = os.path.abspath("../conf")
stubs_path = os.path.abspath("../conf/stubs")

scanner_path = "/tmp/scanner"

code_base_path = "/".join([scanner_path, "code"])

pyre_config_path = "/".join([code_base_path, ".pyre_configuration"])

output_path = "/".join([code_base_path, "pysa-output"])

# COMMAND ----------

import ast
import shutil

# currently command logs don't include what language, so
# we attempt to parse the line to check if it's valid
# Python code
def verify_is_python(code):
    try:
        ast.parse(code)
    except:
        return False
    
    return True

# clean up any existing notebook code
shutil.rmtree(code_base_path, ignore_errors=True)

os.makedirs(code_base_path, exist_ok=True)

# iterate over the results, write the code out to files based on notebook id and session id
for row in pdf_combined.itertuples():
    command_id = row.command_id
    command_text = row.command_text
    notebook_path = row.notebook_path
    notebook_id = row.notebook_id
    session_id = str(row.session_id)
    
    if verify_is_python(command_text):
        code_path = "/".join([code_base_path, notebook_id, session_id])

        os.makedirs(code_path, exist_ok=True)

        with open(f"{code_path}/code.py", "a") as f:
            # we add a command_id to be able to link directly to the cell in the report
            f.write(f"## command_id: {command_id}\n")
            f.write(f"{command_text}\n\n")

# COMMAND ----------

import sys
import json

# generate a Pyre config file to use for scanning
# we set the paths for dependencies and stubs
# note that if you have common 3rd party or custom libraries you should install them so they get indexed by Pyre
pyre_config = {
    "source_directories": [code_base_path],
    "taint_models_path": [conf_path, f"{scanner_path}/pyre-check/stubs/taint"],
    "site_package_search_strategy": "pep561",
    "search_path": [
        f"{scanner_path}/pyre-check/stubs/",
        "/databricks/spark/python/",
        f"/databricks/python/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/",
        stubs_path
    ],
    "workers": 4
}

with open(pyre_config_path, 'w') as f:
    f.write(json.dumps(pyre_config))

# COMMAND ----------

import subprocess

# clean up any existing output
shutil.rmtree(output_path, ignore_errors=True)

# clean up any previous pyre temp data
shutil.rmtree(f"{code_base_path}/.pyre", ignore_errors=True)

# run pysa scan
run = subprocess.run(["pyre", "--noninteractive", "analyze", "--save-results-to", output_path], cwd=code_base_path, capture_output=True)

if run.returncode != 0:
    print(run.stdout.decode('utf-8'))
    print()
    print(run.stderr.decode('utf-8'))

# COMMAND ----------

# to download the Pysa output to use with SAPP, copy it to DBFS or another accessible storage location
# be aware that DBFS may be accessible to other users on the workspace
# shutil.copytree(output_path, "/dbfs/tmp/pysa-output")

# you may also want to copy the code so that SAPP can show the affected lines of code
# shutil.copytree(code_base_path, "/dbfs/tmp/pysa-code-scanned")

# COMMAND ----------

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import jinja2
import re

# load report template for displaying in the notebook
env = jinja2.Environment(loader=jinja2.FileSystemLoader("../conf"))
template = env.get_template("report.template")

pattern = re.compile("## command_id: (\d+)")

# load errors from pysa scan
errors = json.load(open(f"{output_path}/errors.json"))

lexer = PythonLexer(stripall=False)
highlight_css = HtmlFormatter().get_style_defs('.highlight')

categories = set([x["name"] for x in errors])
items = []

for error in errors:
    item = dict()
    
    item["name"] = error["name"]    
    
    notebook_id = os.path.dirname(os.path.dirname(error["path"]))
    session_id = int(os.path.basename(os.path.dirname(error["path"])))
    
    result_row = pdf_combined[(pdf_combined["notebook_id"] == notebook_id) & (pdf_combined["session_id"] == session_id)]
    
    item["notebook_name"] = "Unknown notebook name"
    item["email"] = ""
    item["timestamp"] = ""
    item["notebook_id"] = notebook_id
    
    if len(result_row) > 0:
        item["notebook_name"] = result_row.notebook_path.iloc[0] or item["notebook_name"]
        item["email"] = result_row.email.iloc[0] or ""
        item["timestamp"] = str(result_row.timestamp.iloc[0]) or ""
    
    with open(f'{code_base_path}/{error["path"]}') as f:
        lines = f.readlines()
        
        command_id = "0"
        
        # try to find injected command_id preceding the error line
        # we use this for building direct links
        for i in range(error["line"], 0, -1):
            match = pattern.match(lines[i-1])
            if match:
                command_id = match.group(1)
                break
        
        item["command_id"] = command_id
        
        # show sample code 4 lines before and after
        start = max(0, error["line"]-4)
        stop = min(len(lines)-1, error["stop_line"]+4)
        
        highlight_line_number = error["line"] - start
        
        code_lines = []
        
        for x in range(start, stop):            
            # don't add the command_id comment
            match = pattern.match(lines[x])
            if match is None:
                code_lines.append(lines[x])
            else:
                highlight_line_number = highlight_line_number - 1
        
        formatter = HtmlFormatter(hl_lines=[highlight_line_number])
        
        item["code_sample"] = highlight("".join(code_lines), lexer, formatter)
        
    items.append(item)

# render the report
report = template.render(highlight_css=highlight_css, categories=categories, items=items)

displayHTML(report)

# COMMAND ----------


