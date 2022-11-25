-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Step 1: Read In Dataset

-- COMMAND ----------

-- MAGIC %md
-- MAGIC In step 1, A CSV dataset on cryptocurrency prices is read from a mounted S3 bucket. The dataset is a subset of the [Kaggle G-Research Crypto Forecasting dataset](https://www.kaggle.com/competitions/g-research-crypto-forecasting/data). To learn how to mount an AWS S3 bucket to Databricks, please refer to my previous tutorial [Databricks Mount To AWS S3 And Import Data](https://grabngoinfo.com/databricks-mount-to-aws-s3-and-import-data/).

-- COMMAND ----------

-- mode "FAILFAST" will abort file parsing with a RuntimeException if any malformed lines are encountered
CREATE OR REPLACE TEMPORARY VIEW crypto_100k_records
USING CSV
OPTIONS (path "/mnt/demo4tutorial/data/crypto_100k_records.csv", header "true", mode "FAILFAST");

-- Take a look at the data
SELECT * FROM crypto_100k_records

-- COMMAND ----------

-- MAGIC %md
-- MAGIC After reading the data, we will do some data processing. The timestamp is in UNIX epoch format, which is the number of seconds since January 1st of 1970 Coordinated Universal Time (UTC). Using `from_unixtime`, we changed it to a DateTime format. The columns that are not used in the visualization are dropped. We also created a new column for asset names.

-- COMMAND ----------

-- Data processing
CREATE OR REPLACE TEMPORARY VIEW df AS
SELECT 
  asset_id,
  volume,
  --Change epoch to datetime format
  from_unixtime(timestamp) AS datetime, 
  -- Create asset name
  CASE WHEN asset_id = 1 THEN 'Bitcoin' WHEN asset_id = 6 THEN 'Ethereum' ELSE 'Other' END AS asset_name
FROM crypto_100k_records;

-- Take a look at the data
SELECT * FROM df

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Step 2: Create Databricks Widgets Using Python

-- COMMAND ----------

-- MAGIC %md
-- MAGIC In step 2, we use SQL to create different types of Databricks widgets.
-- MAGIC * The dropdown widget is for the `Asset_Name` column. It has the name of `dropdown_filter` and the default value of `Bitcoin`. There are four choices in the dropdown. `Bitcoin`, `Ethereum`, `Other`, and `All`. `All` means selecting all the asset names.
-- MAGIC * The multiselect widget is based on the `Asset_Name` column too. It has the name of `multiselect_filter` and the default value of `Bitcoin`. The three choices, `Bitcoin`, `Ethereum`, and `Other` are the three unique values for the `Asset_Name` column. We can select multi-values using the multiselect widget.
-- MAGIC * The combobox widget is based on the `Asset_ID` column. It has the name of `combobox_filter` and the default value of `0`. The unique values of the `Asset_ID` column is pulled using SELECT DISTINCT.
-- MAGIC * The text widget is based on the `Asset_ID` column too. It has the name of `text_filter` and the default value of `0`. Users can enter the asset ID into the box.

-- COMMAND ----------

-- Create a dropdown widget
CREATE WIDGET DROPDOWN dropdown_filter DEFAULT "Bitcoin"  CHOICES (VALUES 'Bitcoin', 'Ethereum', 'Other', 'All');

-- Create a multiselect widget
CREATE WIDGET MULTISELECT multiselect_filter DEFAULT "Bitcoin"  CHOICES (VALUES 'Bitcoin', 'Ethereum', 'Other');

-- Create a combobox widget
CREATE WIDGET COMBOBOX combobox_filter DEFAULT "0"  CHOICES SELECT DISTINCT asset_id FROM df;

-- Create a text widget
CREATE WIDGET TEXT text_filter DEFAULT "0";

-- COMMAND ----------

-- MAGIC %md
-- MAGIC After running the code, the widgets show on the top of the notebook.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Step 3: Pass Widget Values in SQL Code

-- COMMAND ----------

-- MAGIC %md
-- MAGIC After creating the widgets, in step 3, we will talk about how to pass the widget values using SQL.
-- MAGIC A widget value can be retrieved by passing the widget name into the `getArgument()` function. 

-- COMMAND ----------

-- Pass widget value to SQL code
SELECT * 
FROM df
WHERE asset_id = getArgument('text_filter')

-- COMMAND ----------

-- MAGIC %md
-- MAGIC When the dropdown widget has `All` as an option, we need to treat `All` and other options differently because `All` is not a value in the dataframe column. One way to do this is to use the `CASE WHEN` conditions to include all records if the dropdown widget value is `All`, and filter by the widget value otherwise.

-- COMMAND ----------

-- Dropdown widget with All as one option
SELECT 
  datetime,
  volume
FROM df
WHERE CASE getArgument('dropdown_filter') WHEN 'Bitcoin' THEN asset_name = 'Bitcoin'
                                          WHEN 'Ethereum' THEN asset_name = 'Ethereum'
                                          WHEN 'Other' THEN asset_name = 'Other'    
                                          ELSE asset_name IN (SELECT DISTINCT asset_name FROM df) END

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Step 4: Use Widgets As Filters For Dashboard

-- COMMAND ----------

-- MAGIC %md
-- MAGIC In step 4, we will talk about how to use widgets as filters for Databricks dashboard.
-- MAGIC 
-- MAGIC Firstly, let's create a chart using Databricks' built-in tool.
-- MAGIC 
-- MAGIC Click the downward triangle next to the bar chart icon, then select the chart type.
-- MAGIC 
-- MAGIC Next, click the Plot Options icon to check if the settings for the chart are correct and make changes if necessary.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC To create a dashboard, click the bar chart icon on the upper right corner of the cell, then click **Add to New Dashboard**. This opens the dashboard. We can see the filters on the top of the dashboard. To learn more about how to create a Databricks dashboard, please refer to my tutorial [Databricks Dashboard For Big Data](https://grabngoinfo.com/databricks-dashboard-for-big-data/)

-- COMMAND ----------

-- Pass widget value to SQL code for dashboard
SELECT volume 
FROM df
WHERE asset_id = getArgument('text_filter')

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Step 5: Configure Databricks Widgets

-- COMMAND ----------

-- MAGIC %md
-- MAGIC In step 5, we will configure the Databricks widgets by clicking the gear icon on the top right of the notebook. The Widgets Panel Settings window will pop up. Under On Widget Change, there are three options.
-- MAGIC 
-- MAGIC * **Run Accessed Commands** means that when the widget values change, only the cells that directly retrieve the changed widget are rerun. This is the default setting, but it does not work for SQL cells.
-- MAGIC * **Do Nothing** means that the notebook will not rerun based on the new widget values.
-- MAGIC * **Run Notebook** means rerun the whole notebook. I recommend choosing this option to prevent missing some of the important steps in the code. 
-- MAGIC 
-- MAGIC The widgets panel is pin to the top of the notebook by default, but we can uncheck the Pinned to top option to show it above the first cell.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Step 6: Pass Values to Widgets in Another Notebook

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Sometimes we may need to run the notebook with specific parameters from another notebook. In that case, we can use `%run` to run the notebook and pass the parameters at the same time. The sample code below is from the Databricks documentation for widgets.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Pass parameters to widgets in another notebook
-- MAGIC %run /path/to/notebook $X="10" $Y="1"

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Step 7: Delete Databricks Widgets

-- COMMAND ----------

-- MAGIC %md
-- MAGIC In step 7, we will talk about how to delete Databricks widgets. To delete one widget, use `REMOVE WIDGET widget_name`.

-- COMMAND ----------

-- Remove one widget
REMOVE WIDGET text_filter

-- COMMAND ----------

-- MAGIC %md
-- MAGIC To remove all widgets, we need to use python code `dbutils.widgets.removeAll()`.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Remove all widgets
-- MAGIC dbutils.widgets.removeAll()
