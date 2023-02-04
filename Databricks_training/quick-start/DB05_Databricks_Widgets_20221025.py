# Databricks notebook source
# MAGIC %md
# MAGIC # Step 0: Databricks Widget Types

# COMMAND ----------

# MAGIC %md
# MAGIC There are four types of Databricks widgets:
# MAGIC * `text` takes text as inputs.
# MAGIC * `dropdown` creates a dropdown list with values.
# MAGIC * `combobox` is a combination of text and dropdown. Users can either select values from the dropdown list or input their own values.
# MAGIC * `multiselect` creates a list of values. Users can select one or more values from the list.
# MAGIC 
# MAGIC To get the help information about widgets, use `dbutils.widgets.help()`. The output has the methods available for widgets and their syntax.

# COMMAND ----------

# Get documentation about widgets
dbutils.widgets.help()

# COMMAND ----------

# MAGIC %md
# MAGIC To get the help information about a specific widget method, use `dbutils.widgets.help(methodName)`. For example, I can use `dbutils.widgets.help('combobox')` to get information about the `combobox` method.

# COMMAND ----------

# Get help informatuon for one method
dbutils.widgets.help('combobox')

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 1: Import Libraries

# COMMAND ----------

# MAGIC %md
# MAGIC In the first step, we will import the pyspark SQL functions for data processing. `to_timestamp` is for processing time data, `when` is for creating new columns based on conditions, and `col` is for working with columns.

# COMMAND ----------

# Functions for data processing
from pyspark.sql.functions import to_timestamp, when, col

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 2: Read In Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC In step 2, A CSV dataset on cryptocurrency prices is read from a mounted S3 bucket. The dataset is a subset of the [Kaggle G-Research Crypto Forecasting dataset](https://www.kaggle.com/competitions/g-research-crypto-forecasting/data). To learn how to mount an AWS S3 bucket to Databricks, please refer to my previous tutorial [Databricks Mount To AWS S3 And Import Data](https://grabngoinfo.com/databricks-mount-to-aws-s3-and-import-data/).

# COMMAND ----------

# Read in CSV data
df = (spark.read.format('csv')
  .option("inferSchema", True)
  .option("header", True)
  .option("sep", ',')
  .load("/mnt/demo4tutorial/data/crypto_100k_records.csv"))
 
# Take a look at the data
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC After reading the data, we will do some data processing. The timestamp is in UNIX epoch format, which is the number of seconds since January 1st of 1970 Coordinated Universal Time (UTC). Using `to_timestamp`, we changed it to a DateTime format. The columns that are not used in the visualization are dropped. We also created a new column for asset names.

# COMMAND ----------

# Change epoch to datetime format and drop unwanted columns
df = df.withColumn('DateTime', to_timestamp(df['timestamp'])).drop('timestamp', 'Count', 'Open', 'High', 'Low', 'Close', 'VWAP', 'Target')
 
# Create asset name
df = df.withColumn('Asset_Name', when(df['Asset_ID']==1, 'Bitcoin')
                                  .when(df['Asset_ID']==6, 'Ethereum')
                                  .otherwise('Other'))

# Take a look at the data    
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3: Create Databricks Widgets Using Python

# COMMAND ----------

# MAGIC %md
# MAGIC In step 3, we use python to create different types of Databricks widgets.
# MAGIC * The dropdown widget is for the `Asset_Name` column. It has the name of `dropdown_filter` and the default value of `Bitcoin`. There are four choices in the dropdown. `Bitcoin`, `Ethereum`, `Other`, and `All`. `All` means selecting all the asset names.
# MAGIC * The multiselect widget is based on the `Asset_Name` column too. It has the name of `multiselect_filter` and the default value of `Bitcoin`. The three choices, `Bitcoin`, `Ethereum`, and `Other` are the three unique values for the `Asset_Name` column. We can select multi-values using the multiselect widget.
# MAGIC * The combobox widget is based on the `Asset_ID` column. It has the name of `combobox_filter` and the default value of `0`. The unique values of the `Asset_ID` column is pulled using the `distinct()` function.
# MAGIC * The text widget is based on the `Asset_ID` column too. It has the name of `text_filter` and the default value of `0`. Users can enter the asset ID into the box.

# COMMAND ----------

# Create a dropdown widget
dbutils.widgets.dropdown(name='dropdown_filter', defaultValue='Bitcoin', choices=['Bitcoin', 'Ethereum', 'Other', 'All'], label='Select asset from the dropdown')

# Create a multiselect widget
dbutils.widgets.multiselect(name='multiselect_filter', defaultValue='Bitcoin', choices=['Bitcoin', 'Ethereum', 'Other'], label='Select multiple assets')

# COMMAND ----------

# Get unique asset id
unique_asset_id = [str(df.select('Asset_ID').distinct().collect()[i][0]) for i in range(len(df.select('Asset_ID').distinct().collect()))]

# Create a combobox widget
dbutils.widgets.combobox(name='combobox_filter', defaultValue='0', choices=unique_asset_id, label='Select or enter asset ID')

# Create a text widget
dbutils.widgets.text(name='text_filter', defaultValue='0', label='Enter asset ID')

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 4: Get Values From Databricks Widgets

# COMMAND ----------

# MAGIC %md
# MAGIC After creating the widgets, in step 4, we will check the value of each widget.
# MAGIC A widget value can be retrieved by passing the widget name into `dbutils.widgets.get()`. We saved the retrieved widget values into variables and printed it out.

# COMMAND ----------

# Save the dropdown widget value into a variable
dropdown_filter_value = dbutils.widgets.get("dropdown_filter")

# Save the multi-select widget value into a variable
multiselect_filter_value = dbutils.widgets.get("multiselect_filter")

# Save the combobox widget value into a variable
combobox_filter_value = dbutils.widgets.get("combobox_filter")

# Save the text widget value into a variable
text_filter_value = dbutils.widgets.get("text_filter")

# Print the widget values
print(f'The dropdown filter value is {dropdown_filter_value}.\nThe multiselect filter value is {multiselect_filter_value}.\nThe combobox filter value is {combobox_filter_value}.\nThe text filter value is {text_filter_value}.')

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 5: Pass Widget Values in Python Code

# COMMAND ----------

# MAGIC %md
# MAGIC In step 5, we will talk about how to pass the widget values using python.
# MAGIC 
# MAGIC We can use `dbutils.widgets.get()` function to pull the value directly from the widget like the code below.

# COMMAND ----------

# Pass widget values using get()
display(df.filter(col('Asset_ID')==dbutils.widgets.get('text_filter')))

# COMMAND ----------

# MAGIC %md
# MAGIC Or alternatively, we can pass the value using the variable created in step 4. In this example, we take the variable called `multiselect_filter_value`, split the string using comma as the delimeter, and filter the dataframe by checking if the asset name is in the multiselect filter list.

# COMMAND ----------

# Pass widget values using the varaibles created in step 4.
display(df.filter(col('Asset_Name').isin(multiselect_filter_value.split(','))))

# COMMAND ----------

# MAGIC %md
# MAGIC When the dropdown widget has `All` as an option, we need to treat `All` and other options differently because `All` is not a value in the dataframe column. One way to do this is to use `if` `else` conditions to display all records if the dropdown widget value is `All`, and filter by the widget value otherwise.

# COMMAND ----------

# Pass dropdown value with All as one option
if dropdown_filter_value == 'All':
    display(df)
else:
    display(df.filter(col('Asset_Name').isin(dropdown_filter_value)))

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 6: Use Widgets As Filters For Dashboard

# COMMAND ----------

# MAGIC %md
# MAGIC In step 6, we will talk about how to use widgets as filters for Databricks dashboard.
# MAGIC 
# MAGIC Firstly, let's create a time series chart for volume using Databricks's built-in tool.
# MAGIC 
# MAGIC Click the downward triangle next to the bar chart icon, then select the line chart option.
# MAGIC 
# MAGIC Next, click the Plot Options icon to check if the settings for the chart are correct and make changes if necessary.

# COMMAND ----------

# MAGIC %md
# MAGIC To create a dashboard, click the bar chart icon on the upper right corner of the cell, then click **Add to New Dashboard**. This opens the dashboard. We can see the filters on the top of the dashboard. To learn more about how to create a Databricks dashboard, please refer to my tutorial [Databricks Dashboard For Big Data](https://grabngoinfo.com/databricks-dashboard-for-big-data/)

# COMMAND ----------

# Example for dashboard filter
if dropdown_filter_value == 'All':
    display(df)
else:
    display(df.filter(col('Asset_Name').isin(dropdown_filter_value)))

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 7: Configure Databricks Widgets

# COMMAND ----------

# MAGIC %md
# MAGIC In step 7, we will configure the Databricks widgets by clicking the gear icon on the top right of the notebook. The Widgets Panel Settings window will pop up. Under On Widget Change, there are three options.
# MAGIC 
# MAGIC * **Run Accessed Commands** means that when the widget values change, only the cells that directly retrieve the changed widget are rerun. This is the default setting, but it does not work for SQL cells.
# MAGIC * **Do Nothing** means that the notebook will not rerun based on the new widget values.
# MAGIC * **Run Notebook** means rerun the whole notebook. I recommend choosing this option to prevent missing some of the important steps in the code. 
# MAGIC 
# MAGIC The widgets panel is pin to the top of the notebook by default, but we can uncheck the Pinned to top option to show it above the first cell.

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 8: Pass Values to Widgets in Another Notebook

# COMMAND ----------

# MAGIC %md
# MAGIC Sometimes we may need to run the notebook with specific parameters from another notebook. In that case, we can use `%run` to run the notebook and pass the parameters at the same time. The sample code below is from the Databricks documentation for widgets.

# COMMAND ----------

# Pass parameters to widgets in another notebook
%run /path/to/notebook $X="10" $Y="1"

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 9: Delete Databricks Widgets

# COMMAND ----------

# MAGIC %md
# MAGIC In step 9, we will talk about how to delete Databricks widgets. To delete one widget, use `dbutils.widgets.remove("widget_name")`.

# COMMAND ----------

# Remove one widget
dbutils.widgets.remove("text_filter")

# COMMAND ----------

# MAGIC %md
# MAGIC To remove all widgets, use `dbutils.widgets.removeAll()`.

# COMMAND ----------

# Remove all widgets
dbutils.widgets.removeAll()

# COMMAND ----------

# MAGIC %md
# MAGIC After removing a widget, we cannot create new widgets in the same cell. New widgets need to be created in a separate cell.
