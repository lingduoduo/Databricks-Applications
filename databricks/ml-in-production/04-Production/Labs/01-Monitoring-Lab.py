# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %run "../../Includes/Drift-Monitoring-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="90df02e3-24c6-4bc6-99db-31417257255f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Drift Monitoring Lab
# MAGIC 
# MAGIC In this lab, you will look at simulated data for an ice cream shop. This data contains a first and second period of data, like we saw in the previous lesson. Your job is to use the techniques you just learned to identify any potential drift occuring across the two time periods. 
# MAGIC 
# MAGIC The data contains the following columns:
# MAGIC 
# MAGIC **Numeric:**
# MAGIC * **`temperature`**: The temperature on the given day
# MAGIC * **`number_of_cones_sold`**: The number of ice cream cones sold on the given day
# MAGIC * **`number_bowls_sold`**: The number of bowls sold, as opposed to cones
# MAGIC * **`total_store_sales`**: The total amount of money in sales done by the other, non ice cream products at the shop.
# MAGIC * **`total_sales_predicted`**: Our imaginary model's prediction for the total_store_sales that day. 
# MAGIC 
# MAGIC **Categorical:**
# MAGIC * **`most_popular_ice_cream_flavor`**: The most popular ice cream flavor on a given day
# MAGIC * **`most_popular_sorbet_flavor`**: The most popular sorbet flavor on a given day
# MAGIC 
# MAGIC 
# MAGIC In this situation, we have an imaginary model attempting to predict the total sales at the store of other, non ice cream items at the store, such as t-shirts or other merchandise. 
# MAGIC Given the first and second time period of simulated data, identify any potential drift and analyze how you might handle it.

# COMMAND ----------

# MAGIC %md <i18n value="41653424-6d5b-4214-b16f-bf5c6fe8e284"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Let's take a look at the first time period ice cream dataframe!

# COMMAND ----------

df1.head()

# COMMAND ----------

# MAGIC %md <i18n value="e608da11-41af-42db-a6ef-95e26e17e530"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC You will try to identify the forms of simulated drift in this dataset. The dataset was changed in the following ways:
# MAGIC 
# MAGIC 1. An upstream data management error converted Fahrenheit to Celsius
# MAGIC 2. The number of cones sold stayed constant
# MAGIC 3. The most popular flavor of ice cream distribution changed, but no nulls were introduced
# MAGIC 4. Bowls became more popular, and the number of bowls sold increased
# MAGIC 5. The most popular sorbet flavors had nulls introduced, and although they are still evenly distributed, the counts thus changed
# MAGIC 6. The **`total_store_sales`** of other, non ice cream merchandise, increased
# MAGIC 7. The prediction of **`total_store_sales`** decreased
# MAGIC 
# MAGIC Keep these changes in mind and see how we would detect them using the tools we have learned.

# COMMAND ----------

# MAGIC %md <i18n value="132cbccb-ebd5-40fa-b0a0-f0c829fe2779"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Let's take a look at the second time period ice cream dataframe!

# COMMAND ----------

df2.head()

# COMMAND ----------

# MAGIC %md <i18n value="e9e9c8c7-8904-45d1-be3a-e8ff188656b8"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC We have defined a **`Monitor`** class for you. Please invoke it below to answer the following questions.

# COMMAND ----------

import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np 
from scipy.spatial import distance

class Monitor():
  
    def __init__(self, pdf1, pdf2, cat_cols, num_cols, alpha=.05, js_stat_threshold=0.2):
        """
        Pass in two pandas dataframes with the same columns for two time windows
        List the categorical and numeric columns, and optionally provide an alpha level
        """
        assert (pdf1.columns == pdf2.columns).all(), "Columns do not match"
        self.pdf1 = pdf1
        self.pdf2 = pdf2
        self.categorical_columns = cat_cols
        self.continuous_columns = num_cols
        self.alpha = alpha
        self.js_stat_threshold = js_stat_threshold
    
    def run(self):
        """
        Call to run drift monitoring
        """
        self.handle_numeric_js()
        self.handle_categorical()
        
        pdf1_nulls = self.pdf1.isnull().sum().sum()
        pdf2_nulls = self.pdf2.isnull().sum().sum()
        print(f"{pdf1_nulls} total null values found in pdf1 and {pdf2_nulls} in pdf2")
        
  
    def handle_numeric_ks(self):
        """
        Handle the numeric features with the Two-Sample Kolmogorov-Smirnov (KS) Test with Bonferroni Correction 
        """
        corrected_alpha = self.alpha / len(self.continuous_columns)

        for num in self.continuous_columns:
            ks_stat, ks_pval = stats.ks_2samp(self.pdf1[num], self.pdf2[num], mode="asymp")
            if ks_pval <= corrected_alpha:
                self.on_drift(num)
                
    def handle_numeric_js(self):
        """
        Handles the numeric features with the Jensen Shannon (JS) test using the threshold attribute
        """
        for num in self.continuous_columns:
            # Run test comparing old and new for that attribute
            range_min = min(self.pdf1[num].min(), self.pdf2[num].min())
            range_max = max(self.pdf1[num].max(), self.pdf2[num].max())
            base = np.histogram(self.pdf1[num], bins=20, range=(range_min, range_max))
            comp = np.histogram(self.pdf2[num], bins=20, range=(range_min, range_max))
            js_stat = distance.jensenshannon(base[0], comp[0], base=2)
            if js_stat >= self.js_stat_threshold:
                self.on_drift(num)
      
    def handle_categorical(self):
        """
        Handle the Categorical features with Two-Way Chi-Squared Test with Bonferroni Correction
        Note: null counts can skew the results of the Chi-Squared Test so they're currently dropped
            by `.value_counts()`
        """
        corrected_alpha = self.alpha / len(self.categorical_columns)

        for feature in self.categorical_columns:
            pdf_count1 = pd.DataFrame(self.pdf1[feature].value_counts()).sort_index().rename(columns={feature:"pdf1"})
            pdf_count2 = pd.DataFrame(self.pdf2[feature].value_counts()).sort_index().rename(columns={feature:"pdf2"})
            pdf_counts = pdf_count1.join(pdf_count2, how="outer")#.fillna(0)
            obs = np.array([pdf_counts["pdf1"], pdf_counts["pdf2"]])
            _, p, _, _ = stats.chi2_contingency(obs)
            if p < corrected_alpha:
                self.on_drift(feature)

    def generate_null_counts(self, palette="#2ecc71"):
        """
        Generate the visualization of percent null counts of all features
        Optionally provide a color palette for the visual
        """
        cm = sns.light_palette(palette, as_cmap=True)
        return pd.concat([100 * self.pdf1.isnull().sum() / len(self.pdf1), 
                          100 * self.pdf2.isnull().sum() / len(self.pdf2)], axis=1, 
                          keys=["pdf1", "pdf2"]).style.background_gradient(cmap=cm, text_color_threshold=0.5, axis=1)
    
    def generate_percent_change(self, palette="#2ecc71"):
        """
        Generate visualization of percent change in summary statistics of numeric features
        Optionally provide a color palette for the visual
        """
        cm = sns.light_palette(palette, as_cmap=True)
        summary1_pdf = self.pdf1.describe()[self.continuous_columns]
        summary2_pdf = self.pdf2.describe()[self.continuous_columns]
        percent_change = 100 * abs((summary1_pdf - summary2_pdf) / (summary1_pdf + 1e-100))
        return percent_change.style.background_gradient(cmap=cm, text_color_threshold=0.5, axis=1)
  
    def on_drift(self, feature):
        """
        Complete this method with your response to drift.  Options include:
          - raise an alert
          - automatically retrain model
        """
        print(f"Drift found in {feature}!")


# COMMAND ----------

# MAGIC %md <i18n value="b181cd42-1406-4bf8-8dc6-77a9e9f60cdd"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Create a **`Monitor`** object based on our first and second period of ice cream data to identify drift.

# COMMAND ----------

drift_monitor = Monitor(
  df1,
  df2, 
  cat_cols = ["most_popular_ice_cream_flavor", "most_popular_sorbet_flavor"], 
  num_cols = ["temperature", "number_of_cones_sold", "number_bowls_sold", "total_store_sales", "total_sales_predicted"],
  alpha=.05, 
  js_stat_threshold=0.2
)

# COMMAND ----------

# MAGIC %md <i18n value="24755f69-2a0e-45ba-a1f3-b45871e25dbb"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Summary Statistics
# MAGIC 
# MAGIC Look over and compare some of the data and their summary stats. Use the **`drift_monitor`** class to generate the null counts. Does anything jump out at you?

# COMMAND ----------

# TODO

# COMMAND ----------

# MAGIC %md <i18n value="52effbfd-a185-4d1e-a711-fe5997db94ed"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Use the **`drift_monitor`** class to generate percent changes. Does anything jump out at you?

# COMMAND ----------

# TODO

# COMMAND ----------

# MAGIC %md <i18n value="ee2a0b06-5a3f-4e1c-b255-3a2f59db70d5"/>
# MAGIC Investigate why `temperature` has such a big percent change! If you compare `df1.describe()` and `df2.describe()`, what differences do you see? `df1` uses Fahrenheit whereas `df2` uses Celsius! In this case, it was relatively easy to find out the root cause of the drift; however, in real use cases, it might be much harder!

# COMMAND ----------

# MAGIC %md <i18n value="29aaae05-cbd2-4515-b483-3b7224bf6187"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Statistical Tests
# MAGIC 
# MAGIC Now let's try the Jensen Shannon and Two-Way Chi-Squared Test with Bonferroni Correction. 
# MAGIC 
# MAGIC Both of these are implemented for you when you call **`drift_monitor.run()`**. It will print a feature name if a statisitically significant p-value was found by the respective test on that feature or if the JS stat is above our predetermined threshold.
# MAGIC 
# MAGIC Examine the results and compare them to the changes we made.

# COMMAND ----------

# TODO

# COMMAND ----------

# MAGIC %md <i18n value="063891c2-6b81-47ec-8a01-76511bb52349"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Closer Look
# MAGIC 
# MAGIC ***Using these summary statistics and statistical tests were we able to catch all of our drift?***
# MAGIC 
# MAGIC Imagine you were running this ice cream shop:
# MAGIC * ***How would you handle each situation?***
# MAGIC * ***How would it affect our model or the business?***

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
