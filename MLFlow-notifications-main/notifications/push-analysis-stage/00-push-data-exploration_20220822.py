# Databricks notebook source
# MAGIC %md 
# MAGIC ### Initial Data Table Setup

# COMMAND ----------

# MAGIC %md 
# MAGIC ```
# MAGIC   select 
# MAGIC       common__timestamp,
# MAGIC       network_user_id,
# MAGIC       event_type,
# MAGIC       event_status,
# MAGIC       notification_type,
# MAGIC       notification_name,
# MAGIC       device_type,
# MAGIC       from_user_id,
# MAGIC       from_user_network
# MAGIC   from tmg.s_tmg_notification n
# MAGIC   where n._partition_date >= date_add('day', -1, current_date)
# MAGIC     and n._partition_date < current_date
# MAGIC     and notification_name in ('broadcastStartWithDescriptionPush', 'broadcastStartPush')
# MAGIC     and network = 'meetme'
# MAGIC   group by 1,2,3,4,5,6,7,8,9
# MAGIC ```

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/shared_uploads/lhuang@themeetgroup.com/0bdc7eed_e83e_445d_88e0_4da185392a3f.csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Basic Data Standardization

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

import pyspark.pandas as pd

# COMMAND ----------

df = (spark.read 
                 .format("csv")
                 .option("header", True)
                 .option("inferSchema", True)
                 .load("dbfs:/FileStore/shared_uploads/lhuang@themeetgroup.com/0bdc7eed_e83e_445d_88e0_4da185392a3f.csv")
           )
 
display(df)

# COMMAND ----------

dbutils.data.summarize(df)

# COMMAND ----------

# Set config for database name, file paths, and table names
database_name = 'ml_push'

# Create database to house 
spark.sql('CREATE DATABASE IF NOT EXISTS {} '.format(database_name))
spark.sql('USE {}'.format(database_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploratory Data Analysis

# COMMAND ----------

# Paths for various Delta tables
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
raw_tbl_path = '/home/{}/ml_push/raw/'.format(user)
spark.sql('USE {}'.format(database_name))

# Drop any old delta lake files if needed (e.g. re-running this notebook with the same path variables)
dbutils.fs.rm(raw_tbl_path, recurse = True)

# COMMAND ----------

df.write.format('delta').mode('overwrite').save(raw_tbl_path)

# COMMAND ----------

raw_tbl_name = 'daily_push_meetme'

# Create bronze table to query with SQL
spark.sql('DROP TABLE IF EXISTS {}'.format(raw_tbl_name))
spark.sql('''
             CREATE TABLE {0}
             USING DELTA 
             LOCATION '{1}'
          '''.format(raw_tbl_name, raw_tbl_path)
         )

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM daily_push_meetme
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Data Visualization

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT event_status, count(*), count(distinct network_user_id) AS user_cnt
# MAGIC FROM daily_push_meetme
# MAGIC GROUP BY 1
# MAGIC ORDER BY 2 DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT event_type, count(*), count(distinct network_user_id) AS user_cnt
# MAGIC FROM daily_push_meetme
# MAGIC WHERE event_status = 'success'
# MAGIC GROUP BY 1
# MAGIC ORDER BY 2 DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT event_type, count(*), count(distinct network_user_id) AS user_cnt
# MAGIC FROM daily_push_meetme
# MAGIC WHERE event_status = 'success' and event_type = 'send'
# MAGIC GROUP BY 1
# MAGIC ORDER BY 2 DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC send_data.*,
# MAGIC case when open_data.open_ts > 0 then 1 else 0 end open_flag
# MAGIC FROM
# MAGIC (
# MAGIC   SELECT network_user_id, common__timestamp as send_ts
# MAGIC   FROM daily_push_meetme
# MAGIC   WHERE event_status = 'success' and event_type = 'send'
# MAGIC ) send_data
# MAGIC LEFT JOIN
# MAGIC (
# MAGIC   SELECT network_user_id, common__timestamp as open_ts
# MAGIC   FROM daily_push_meetme
# MAGIC   WHERE event_status = 'success' and event_type = 'open'
# MAGIC ) open_data
# MAGIC on send_data.network_user_id = open_data.network_user_id and send_data.send_ts < open_data.open_ts

# COMMAND ----------


