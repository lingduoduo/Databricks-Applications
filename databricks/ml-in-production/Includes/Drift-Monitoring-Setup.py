# Databricks notebook source
import numpy as np 
import pandas as pd
from scipy import stats

# COMMAND ----------

# Simulate original ice cream dataset 
df1 = pd.DataFrame()
df1['temperature'] = np.random.uniform(60, 80, 1000)
df1['number_of_cones_sold'] = np.random.uniform(0, 20, 1000) 
flavors = ["Vanilla"] * 300 + ['Chocolate'] * 200 + ['Cookie Dough'] * 300 + ['Coffee'] * 200
np.random.shuffle(flavors)
df1['most_popular_ice_cream_flavor'] = flavors
df1['number_bowls_sold'] = np.random.uniform(0, 20, 1000) 
sorbet = ["Raspberry "] * 250 + ['Lemon'] * 250 + ['Lime'] * 250 + ['Orange'] * 250
np.random.shuffle(sorbet)
df1['most_popular_sorbet_flavor'] = sorbet
df1['total_store_sales'] = np.random.normal(100, 10, 1000)
df1['total_sales_predicted'] = np.random.normal(100, 10, 1000)

# Simulate new ice cream dataset
df2 = pd.DataFrame()
df2['temperature'] = (df1['temperature'] - 32) * (5/9) # F -> C
df2['number_of_cones_sold'] = np.random.uniform(0, 20, 1000) #stay same
flavors = ["Vanilla"] * 100 + ['Chocolate'] * 300 + ['Cookie Dough'] * 400 + ['Coffee'] * 200
np.random.shuffle(flavors)
df2['most_popular_ice_cream_flavor'] = flavors
df2['number_bowls_sold'] = np.random.uniform(10, 30, 1000)
sorbet = ["Raspberry "] * 200 + ['Lemon'] * 200 + ['Lime'] * 200 + ['Orange'] * 200 + [None] * 200
np.random.shuffle(sorbet)
df2['most_popular_sorbet_flavor'] = sorbet
df2['total_store_sales'] = np.random.normal(150, 10, 1000) # increased
df2['total_sales_predicted'] = np.random.normal(80, 10, 1000) # decreased

