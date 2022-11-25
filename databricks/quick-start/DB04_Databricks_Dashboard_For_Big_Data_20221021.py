# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1: Import Libraries

# COMMAND ----------

# MAGIC %md
# MAGIC In the first step, we will import the pyspark SQL functions for data processing.

# COMMAND ----------

# Functions for data processing
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 2: Read In Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC In step 2, A CSV dataset on cryptocurrency prices is read from a mounted S3 bucket. To learn how to mount an AWS S3 bucket to Databricks, please refer to my previous tutorial [Databricks Mount To AWS S3 And Import Data](https://grabngoinfo.com/databricks-mount-to-aws-s3-and-import-data/).

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
# MAGIC After reading the data, we will do some data processing. The timestamp is in UNIX epoch format, which is the number of seconds since January 1st of 1970 Coordinated Universal Time (UTC). Using `F.to_timestamp`, we changed it to a DateTime format. The columns that are not used in the visualization are dropped. We also created a new column for asset names.

# COMMAND ----------

# Change epoch to datetime format and drop unwanted columns
df = df.withColumn('DateTimeType', F.to_timestamp(df['timestamp'])).drop('timestamp', 'Count','High', 'Low', 'Close', 'VWAP', 'Target')
 
# Create asset name
df = df.withColumn('Asset_Name', F.when(df['Asset_ID']==1, 'Bitcoin')
                                  .when(df['Asset_ID']==6, 'Ethereum')
                                  .otherwise('Other'))

# Take a look at the data    
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3: Create Dashboard Filters Using Widget

# COMMAND ----------

# MAGIC %md
# MAGIC In step 3, we will create dashboard filters using widgets. The widget enables the dashboard to rerun with different parameters. There are four types of widgets:
# MAGIC * `text` takes text as inputs.
# MAGIC * `dropdown` creates a dropdown list with values.
# MAGIC * `combobox` is a combination of text and dropdown. Users can either select values from the dropdown list or input their own values.
# MAGIC * `multiselect` creates a list of values. Users can select one or more values from the list.
# MAGIC 
# MAGIC In this example, we will use `multiselect` as an example.
# MAGIC 
# MAGIC Before creating the new widget, we first removed all existing widgets using the `removeAll()` function. When creating the widget using `multiselect`, we need to give the widget a name, a default value, a list of values to choose from, and the text displayed next to the filter.

# COMMAND ----------

# Remove existing widgets if there are any
dbutils.widgets.removeAll()

# Create widget
dbutils.widgets.multiselect('name', 'Bitcoin', ['Bitcoin', 'Ethereum', 'Other'], 'Select Crypto Currency Name')

# # Save the filter selection in a variable
# selected_name = dbutils.widgets.get("name").split(",")

# # Print the current selected values
# print(selected_name)

# COMMAND ----------

# MAGIC %md
# MAGIC After creating the widget, we can set up the configuration by clicking the gear icon on the upper right corner of the notebook. Clicking this icon opens the widget configuration. There are three settings to choose from:
# MAGIC 
# MAGIC * Run Notebook reruns the entire notebook when there is a change in the widget value selection.
# MAGIC * Run Accessed Commnds reruns only the cells that retrieve the values for the particular widget.
# MAGIC * Do Nothing does not rerun any cell when there is a change in widget value selection.
# MAGIC 
# MAGIC To make sure the dashboard is updated with the newly selected values, either **Run Notebook** or **Run Accessed Commands** need to be selected. We will choose **Run Accessed Commands** in this example because we only want to rerun the cells that are in the dashboard.

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 4: Create Charts

# COMMAND ----------

# MAGIC %md
# MAGIC In step 4, we will create some commonly used charts using the Databricks built-in chart functions in the notebook.

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4.1: Line Chart

# COMMAND ----------

# MAGIC %md
# MAGIC Step 4.1 creates a line chart. We first selected the columns needed for the chart, then filter the data to include only the values selected in the widget. After that, group the data by date and asset name, and calculate the average. Finally, sort the data by date.
# MAGIC 
# MAGIC The default output is a data table, but we can click the downward arrow next to the bar chart icon and select the line chart option.
# MAGIC 
# MAGIC To make changes to the chart, click Plot Options below the chart.
# MAGIC 
# MAGIC We can change Keys, Series groupings, Values, and Aggregation calculation. We can also change the Y-axis Range, Show Points, and make the color consistent for groups.
# MAGIC 
# MAGIC The legend for the groups are clickable. We can show a subset of groups by deselecting some groups.

# COMMAND ----------

# Line chart
display(df.select('Asset_Name', 'DateTimeType', 'Open')
          .filter(F.col('Asset_Name').alias('Asset Name').isin(dbutils.widgets.get("name").split(",")))
          .groupby(F.to_date('DateTimeType').alias('date'), 'Asset_Name').agg(F.avg('Open').alias('averge open price'))
          .sort(F.to_date('DateTimeType')))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4.2: Bar Chart

# COMMAND ----------

# MAGIC %md
# MAGIC Step 4.2 creates a bar chart. To create a grouped bar chart, choose Grouped in Plot Options.
# MAGIC 
# MAGIC Because the Global color consistency is selected, the color for different groups are consistent with the previous line chart.

# COMMAND ----------

# Bar chart
display(df.select('Asset_Name', 'DateTimeType', 'Volume')
          .filter(F.col('Asset_Name').alias('Asset Name').isin(dbutils.widgets.get("name").split(",")))
          .groupby(F.to_date('DateTimeType').alias('date'), 'Asset_Name').agg(F.sum('Volume').alias('transaction volume'))
          .sort(F.to_date('DateTimeType')))

# COMMAND ----------

# MAGIC %md
# MAGIC We can also create a stacked bar chart by selecting **Stacked** in **Plot Options**.

# COMMAND ----------

# Bar chart
display(df.select('Asset_Name', 'DateTimeType', 'Volume')
          .filter(F.col('Asset_Name').alias('Asset Name').isin(dbutils.widgets.get("name").split(",")))
          .groupby(F.to_date('DateTimeType').alias('date'), 'Asset_Name').agg(F.sum('Volume').alias('transaction volume'))
          .sort(F.to_date('DateTimeType')))

# COMMAND ----------

# MAGIC %md
# MAGIC To get the percent stacked bar chart, choose **100% stacked** in **Plot Options**.

# COMMAND ----------

# Bar chart
display(df.select('Asset_Name', 'DateTimeType', 'Volume')
          .filter(F.col('Asset_Name').alias('Asset Name').isin(dbutils.widgets.get("name").split(",")))
          .groupby(F.to_date('DateTimeType').alias('date'), 'Asset_Name').agg(F.sum('Volume').alias('transaction volume'))
          .sort(F.to_date('DateTimeType')))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4.3 Histogram

# COMMAND ----------

# MAGIC %md
# MAGIC Step 4.3 creates a histogram for cryptocurrency volume distributions.
# MAGIC 
# MAGIC When there is more than one group, use the group category name as the key. This produces one histogram for each group. We can change the number of bins for the histogram.

# COMMAND ----------

# Histogram
display(df.select('Asset_Name', 'DateTimeType', 'Volume')
          .filter(F.col('Asset_Name').alias('Asset Name').isin(dbutils.widgets.get("name").split(",")))
       )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4.4: Pie Chart

# COMMAND ----------

# MAGIC %md
# MAGIC Step 4.4 creates a pie chart for cryptocurrency volume.
# MAGIC 
# MAGIC We set the Assset_Name as the key, and take the sum of the volume by asset name. The pie chart has the option of displaying a donut chart by selecting the Donut option.

# COMMAND ----------

# Pie chart
display(df.select('Asset_Name', 'Volume')
          .filter(F.col('Asset_Name').alias('Asset Name').isin(dbutils.widgets.get("name").split(",")))
          .groupby('Asset_Name').agg(F.sum('Volume'))
           )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4.5: Area Chart

# COMMAND ----------

# Area Chart
display(df.select('Asset_Name', 'DateTimeType', 'Volume')
          .filter(F.col('Asset_Name').alias('Asset Name').isin(dbutils.widgets.get("name").split(",")))
          .groupby(F.to_date('DateTimeType').alias('date'), 'Asset_Name').agg(F.sum('Volume').alias('transaction volume'))
          .sort(F.to_date('DateTimeType')))

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 5: Create Databricks Dashboard

# COMMAND ----------

# MAGIC %md
# MAGIC In step 5, we will talk about how to create a new Databricks dashboard.
# MAGIC 
# MAGIC To create a new dashboard, click the picture icon in the menu, and click the last item, **+ New Dashboard**.
# MAGIC 
# MAGIC All the existing markdown cells and outputs in the notebook will be automatically included in the dashboard. We can delete unwanted content by clicking the cross icon on the upper right corner of the content.
# MAGIC 
# MAGIC The new notebook outputs generated after creating the dashboard will not be included automatically. To include new content, click the bar chart icon on the upper right corner of the cell and check the dashboard name.

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 6: Format a Databricks Dashboard

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks Dashboard tables and charts can be moved around by clicking and dragging.
# MAGIC 
# MAGIC We can also change the size of dashboard objects by clicking the arrow in the lower right corner. Using the markdown cell, we can include headers, texts, images, math equations, and more in the dashboard. 
# MAGIC 
# MAGIC My previous tutorial [Databricks Notebook Markdown Cheat Sheet](https://grabngoinfo.com/databricks-notebook-markdown-cheat-sheet/) covers how to use Databricks markdown. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 7: Switch Between Notebook View and Dashboard View

# COMMAND ----------

# MAGIC %md
# MAGIC In step 7, we will talk about how to switch between a notebook and its dashboard.
# MAGIC 
# MAGIC To switch from a notebook view to a dashboard view, click the picture icon in the menu and select the dashboard name under the Dashboards section. One notebook can have multiple dashboards associated with it.
# MAGIC 
# MAGIC To switch from a dashboard view to a notebook view, we can click the picture icon in the menu and select the dashboard name under **Dashboards**.
# MAGIC 
# MAGIC Alternatively, we can click the notebook name link on the right pane to switch to the notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 8: Share Databricks Dashboard

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks dashboard can be shared with others by clicking the lock icon in the notebook menu. In the Databricks community edition, the permission control is disabled, but users can publish the notebook by clicking the share button in the upper right corner. Anyone with the link can access the notebook and dashboard.
# MAGIC 
# MAGIC To present the dashboard results to others, click the Present Dashboard button on the right panel of the dashboard.
# MAGIC 
# MAGIC To delete a dashboard, click the red Delete this dashboard button.

# COMMAND ----------

# MAGIC %md
# MAGIC * [Databricks Documentation on Dashboard](https://docs.databricks.com/notebooks/dashboards.html)
# MAGIC * [Databricks Documentation on widgets](https://docs.databricks.com/notebooks/widgets.html)
