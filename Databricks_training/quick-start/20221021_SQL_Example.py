# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Group 5 Final Project
# MAGIC 
# MAGIC ### SQL Queries for Reference

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Top Level Aggregated Queries

# COMMAND ----------

# DBTITLE 1,Gold Table - Net Sentiment by Crypto Ticker
# MAGIC %sql
# MAGIC SELECT
# MAGIC   ticker,
# MAGIC   DATE(created_at) as Day,
# MAGIC   date_trunc('hour', created_at) AS Hour,
# MAGIC   count(CASE WHEN sentiment = 'positive' THEN 1 END) AS positive,
# MAGIC   count(CASE WHEN sentiment = 'neutral' THEN 1 END) AS neutral,
# MAGIC   count(CASE WHEN sentiment = 'negative' THEN 1 END) AS negative,
# MAGIC   count(CASE WHEN sentiment = 'positive' THEN 1 END) - count(CASE WHEN sentiment = 'negative' THEN 1 END) AS net_sentiment,
# MAGIC   avg(delta),
# MAGIC   ticker AS `ticker::filter`, COUNT(0) AS Ticker
# MAGIC FROM
# MAGIC     group5_finalproject.gold
# MAGIC WHERE
# MAGIC   created_at > '{{ Date Range.start }}' AND created_at < '{{ Date Range.end }}'
# MAGIC AND CASE
# MAGIC     WHEN is_member('group-a') THEN ticker IN ('SHIB-USD', 'DOT1-USD', 'LTC-USD', 'BTC-USD')
# MAGIC     WHEN is_member('group-b') THEN ticker IN ('AVAX-USD', 'ALGO-USD', 'ETH-USD', 'LUNA1-USD')
# MAGIC     WHEN is_member('group-c') THEN ticker IN ('ADA-USD', 'XLM-USD', 'SOL1-USD', 'BCH-USD','SHIB-USD', 'DOT1-USD', 'LTC-USD', 'BTC-USD')
# MAGIC     WHEN is_member('group-d') THEN ticker IN ('DOGE-USD', 'UNI3-USD', 'XRP-USD')
# MAGIC     ELSE TRUE 
# MAGIC   END
# MAGIC GROUP BY 
# MAGIC     ticker, DATE(created_at), Hour

# COMMAND ----------

# DBTITLE 1,Gold Table - Sentiment and Price by Ticker and Day
# MAGIC %sql
# MAGIC SELECT
# MAGIC   ticker,
# MAGIC   DATE(created_at) as Day,
# MAGIC   count(CASE WHEN sentiment = 'positive' THEN 1 END) AS positive,
# MAGIC   count(CASE WHEN sentiment = 'neutral' THEN 1 END) AS neutral,
# MAGIC   count(CASE WHEN sentiment = 'negative' THEN 1 END) AS negative,
# MAGIC   count(CASE WHEN sentiment = 'positive' THEN 1 END) - count(CASE WHEN sentiment = 'negative' THEN 1 END) AS net_sentiment,
# MAGIC   avg(delta)
# MAGIC FROM
# MAGIC     group5_finalproject.gold
# MAGIC WHERE
# MAGIC   created_at > '{{ Date Range.start }}' AND created_at < '{{ Date Range.end }}'
# MAGIC AND CASE
# MAGIC     WHEN is_member('group-a') THEN ticker IN ('SHIB-USD', 'DOT1-USD', 'LTC-USD', 'BTC-USD')
# MAGIC     WHEN is_member('group-b') THEN ticker IN ('AVAX-USD', 'ALGO-USD', 'ETH-USD', 'LUNA1-USD')
# MAGIC     WHEN is_member('group-c') THEN ticker IN ('ADA-USD', 'XLM-USD', 'SOL1-USD', 'BCH-USD','SHIB-USD', 'DOT1-USD', 'LTC-USD', 'BTC-USD')
# MAGIC     WHEN is_member('group-d') THEN ticker IN ('DOGE-USD', 'UNI3-USD', 'XRP-USD')
# MAGIC     ELSE TRUE 
# MAGIC   END
# MAGIC GROUP BY 
# MAGIC     ticker, DATE(created_at)

# COMMAND ----------

# DBTITLE 1,Full Gold Table
# MAGIC %sql
# MAGIC SELECT
# MAGIC   *
# MAGIC FROM
# MAGIC   group5_finalproject.gold
# MAGIC WHERE
# MAGIC     date > '{{ Date Range.start }}' AND date < '{{ Date Range.end }}'
# MAGIC AND  CASE
# MAGIC     WHEN is_member('group-a') THEN ticker IN ('SHIB-USD', 'DOT1-USD', 'LTC-USD', 'BTC-USD')
# MAGIC     WHEN is_member('group-b') THEN ticker IN ('AVAX-USD', 'ALGO-USD', 'ETH-USD', 'LUNA1-USD')
# MAGIC     WHEN is_member('group-c') THEN ticker IN ('ADA-USD', 'XLM-USD', 'SOL1-USD','SHIB-USD', 'DOT1-USD')
# MAGIC     WHEN is_member('group-d') THEN ticker IN ('DOGE-USD', 'UNI3-USD', 'XRP-USD')
# MAGIC     ELSE TRUE 
# MAGIC   END

# COMMAND ----------

# MAGIC %sql

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Twitter Specific Queries

# COMMAND ----------

# DBTITLE 1,All Tweet Statistics
# MAGIC %sql
# MAGIC SELECT
# MAGIC   QUERY,
# MAGIC   count(result_type) AS num_recent,
# MAGIC   avg(favorite_count) AS avg_favorite,
# MAGIC   avg(followers_count) AS avg_followers,
# MAGIC   avg(retweet_count) AS avg_retweets
# MAGIC FROM
# MAGIC   group5_finalproject.crypto_df_tf
# MAGIC WHERE
# MAGIC     created_at > '{{ Date Range.start }}' AND created_at < '{{ Date Range.end }}'
# MAGIC AND  CASE
# MAGIC     WHEN is_member('group-a') THEN yfinance_ticker IN ('SHIB-USD', 'DOT1-USD', 'LTC-USD', 'BTC-USD')
# MAGIC     WHEN is_member('group-b') THEN yfinance_ticker IN ('AVAX-USD', 'ALGO-USD', 'ETH-USD', 'LUNA1-USD')
# MAGIC     WHEN is_member('group-c') THEN yfinance_ticker IN ('ADA-USD', 'XLM-USD', 'SOL1-USD', 'BCH-USD','SHIB-USD', 'DOT1-USD')
# MAGIC     WHEN is_member('group-d') THEN yfinance_ticker IN ('DOGE-USD', 'UNI3-USD', 'XRP-USD')
# MAGIC     ELSE TRUE 
# MAGIC   END
# MAGIC GROUP BY
# MAGIC     QUERY
# MAGIC ORDER BY
# MAGIC     avg(retweet_count) DESC

# COMMAND ----------

# DBTITLE 1,Tweet Counts by Year, Month, Day, Hour by Crypto
# MAGIC %sql
# MAGIC SELECT
# MAGIC   count(*),
# MAGIC   year(created_at),
# MAGIC   month(created_at),
# MAGIC   day(created_at),
# MAGIC   hour(created_at),
# MAGIC   query,
# MAGIC   created_at
# MAGIC FROM
# MAGIC   group5_finalproject.crypto_df_tf
# MAGIC GROUP BY
# MAGIC   hour(created_at),
# MAGIC   created_at,
# MAGIC   QUERY

# COMMAND ----------

# DBTITLE 1,Tweet Counts by Crypto - filtered for security group
# MAGIC %sql
# MAGIC SELECT
# MAGIC   QUERY,
# MAGIC   count(QUERY) AS tweets
# MAGIC FROM
# MAGIC   group5_finalproject.crypto_df_tf
# MAGIC WHERE
# MAGIC     created_at > '{{ Date Range.start }}' AND created_at < '{{ Date Range.end }}'
# MAGIC AND  CASE
# MAGIC     WHEN is_member('group-a') THEN yfinance_ticker IN ('SHIB-USD', 'DOT1-USD', 'LTC-USD', 'BTC-USD')
# MAGIC     WHEN is_member('group-b') THEN yfinance_ticker IN ('AVAX-USD', 'ALGO-USD', 'ETH-USD', 'LUNA1-USD')
# MAGIC     WHEN is_member('group-c') THEN yfinance_ticker IN ('ADA-USD', 'XLM-USD', 'SOL1-USD', 'BCH-USD','SHIB-USD')
# MAGIC     WHEN is_member('group-d') THEN yfinance_ticker IN ('DOGE-USD', 'UNI3-USD', 'XRP-USD')
# MAGIC     ELSE TRUE 
# MAGIC   END
# MAGIC GROUP BY
# MAGIC   QUERY
# MAGIC ORDER BY
# MAGIC   tweets DESC

# COMMAND ----------

# DBTITLE 1,Most Active Twitter User
# MAGIC %sql
# MAGIC SELECT
# MAGIC   user_name as top_tweeter,
# MAGIC   count(id) as tweets
# MAGIC FROM
# MAGIC   group5_finalproject.crypto_df_tf
# MAGIC WHERE
# MAGIC     created_at > '{{ Date Range.start }}' AND created_at < '{{ Date Range.end }}'
# MAGIC GROUP BY
# MAGIC     user_name
# MAGIC ORDER BY 
# MAGIC     tweets DESC
# MAGIC LIMIT 5

# COMMAND ----------

# DBTITLE 1,Top Twitter Influencers based on Followers
# MAGIC %sql
# MAGIC SELECT
# MAGIC   user_name as tweeter,
# MAGIC   followers_count as followers,
# MAGIC   avg(retweet_count) as retweets,
# MAGIC   count(id) as tweets
# MAGIC FROM
# MAGIC   group5_finalproject.crypto_df_tf
# MAGIC WHERE
# MAGIC     created_at > '{{ Date Range.start }}' AND created_at < '{{ Date Range.end }}'
# MAGIC GROUP BY
# MAGIC     user_name, followers
# MAGIC ORDER BY 
# MAGIC     followers DESC
# MAGIC LIMIT 5

# COMMAND ----------

# DBTITLE 1,Most Popular Crypto on Twitter by Tweet Count
# MAGIC %sql
# MAGIC SELECT
# MAGIC   QUERY,
# MAGIC   count(QUERY) as num_tweets
# MAGIC FROM
# MAGIC   group5_finalproject.crypto_df_tf
# MAGIC WHERE
# MAGIC     created_at > '{{ Date Range.start }}' AND created_at < '{{ Date Range.end }}'
# MAGIC GROUP BY
# MAGIC     QUERY
# MAGIC ORDER BY
# MAGIC     num_tweets DESC
# MAGIC LIMIT 1

# COMMAND ----------

# DBTITLE 1,Total and Recent Tweet Counter
# MAGIC %sql
# MAGIC SELECT
# MAGIC   count(id) as num_tweets,
# MAGIC   ---user_name as top_tweeter,
# MAGIC   count(result_type) as recent_tweets
# MAGIC FROM
# MAGIC   group5_finalproject.crypto_df_tf
# MAGIC WHERE
# MAGIC     created_at > '{{ Date Range.start }}' AND created_at < '{{ Date Range.end }}'

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Finance Specific Queries

# COMMAND ----------

# DBTITLE 1,All Raw Stock Ticker Data
# MAGIC %sql
# MAGIC SELECT
# MAGIC   *
# MAGIC FROM
# MAGIC   group5_finalproject.pricing_info_raw
# MAGIC WHERE
# MAGIC     date > '{{ Date Range.start }}' AND date < '{{ Date Range.end }}'
# MAGIC AND  CASE
# MAGIC     WHEN is_member('group-a') THEN ticker IN ('SHIB-USD', 'DOT1-USD', 'LTC-USD', 'BTC-USD')
# MAGIC     WHEN is_member('group-b') THEN ticker IN ('AVAX-USD', 'ALGO-USD', 'ETH-USD', 'LUNA1-USD')
# MAGIC     WHEN is_member('group-c') THEN ticker IN ('ADA-USD', 'XLM-USD', 'SOL1-USD', 'BCH-USD','SHIB-USD')
# MAGIC     WHEN is_member('group-d') THEN ticker IN ('DOGE-USD', 'UNI3-USD', 'XRP-USD')
# MAGIC     ELSE TRUE 
# MAGIC   END

# COMMAND ----------

# DBTITLE 1,Top Performing Stock
# MAGIC %sql
# MAGIC SELECT
# MAGIC   ticker as stock,
# MAGIC   SUM(delta),
# MAGIC   min(delta),
# MAGIC   max(delta)
# MAGIC FROM
# MAGIC   group5_finalproject.pricing_info_raw
# MAGIC WHERE
# MAGIC     date > '{{ Date Range.start }}' AND date < '{{ Date Range.end }}'
# MAGIC GROUP BY
# MAGIC     stock
# MAGIC ORDER BY 
# MAGIC     MAX(delta) DESC
# MAGIC LIMIT 15

# COMMAND ----------

# DBTITLE 1,Worst Performing Stock
# MAGIC %sql
# MAGIC SELECT
# MAGIC   ticker as stock,
# MAGIC   SUM(delta),
# MAGIC   min(delta),
# MAGIC   max(delta)
# MAGIC FROM
# MAGIC   group5_finalproject.pricing_info_raw
# MAGIC WHERE
# MAGIC     date > '{{ Date Range.start }}' AND date < '{{ Date Range.end }}'
# MAGIC GROUP BY
# MAGIC     stock
# MAGIC ORDER BY 
# MAGIC     MIN(delta) ASC
# MAGIC LIMIT 15

# COMMAND ----------

# MAGIC %sql

# COMMAND ----------

# MAGIC %sql

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Alert Queries

# COMMAND ----------

# DBTITLE 1,Bitcoin Low Sentiment
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   DATE(created_at),
# MAGIC   count(CASE WHEN sentiment = 'positive' THEN 1 END) - count(CASE WHEN sentiment = 'negative' THEN 1 END) AS net_sentiment
# MAGIC FROM
# MAGIC   group5_finalproject.gold
# MAGIC WHERE
# MAGIC     ticker = "BTC-USD"
# MAGIC GROUP BY
# MAGIC   ticker, DATE(created_at)
# MAGIC ORDER BY
# MAGIC     net_sentiment ASC
# MAGIC LIMIT 5

# COMMAND ----------

# DBTITLE 1,Bitcoin High Sentiment
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   DATE(created_at),
# MAGIC   count(CASE WHEN sentiment = 'positive' THEN 1 END) - count(CASE WHEN sentiment = 'negative' THEN 1 END) AS net_sentiment
# MAGIC FROM
# MAGIC   group5_finalproject.gold
# MAGIC WHERE
# MAGIC     ticker = "BTC-USD"
# MAGIC GROUP BY
# MAGIC   ticker, DATE(created_at)
# MAGIC ORDER BY
# MAGIC     net_sentiment DESC
# MAGIC LIMIT 5

# COMMAND ----------

# DBTITLE 1,Bitcoin Price Drop by 2%
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   created_at,
# MAGIC   delta
# MAGIC FROM
# MAGIC   group5_finalproject.gold
# MAGIC WHERE
# MAGIC     ticker = "BTC-USD"
# MAGIC GROUP BY
# MAGIC   ticker, delta, created_at
# MAGIC ORDER BY
# MAGIC     delta ASC
# MAGIC LIMIT 5

# COMMAND ----------

# DBTITLE 1,Bitcoin Jump up by 2%
# MAGIC %sql
# MAGIC SELECT
# MAGIC   created_at,
# MAGIC   delta
# MAGIC FROM
# MAGIC   group5_finalproject.gold
# MAGIC WHERE
# MAGIC     ticker = "BTC-USD"
# MAGIC GROUP BY
# MAGIC   ticker, delta, created_at
# MAGIC ORDER BY
# MAGIC     delta DESC

# COMMAND ----------

# DBTITLE 1,Aggregated Data - Price change by Retweet and Trading Volume
# MAGIC %sql
# MAGIC SELECT
# MAGIC   crypto.QUERY,
# MAGIC   price.date,
# MAGIC   crypto.retweet_count,
# MAGIC   price.delta,
# MAGIC   price.volume
# MAGIC  
# MAGIC FROM
# MAGIC   group5_finalproject.crypto_df_tf AS crypto
# MAGIC JOIN group5_finalproject.ticker_list_tbl AS ticker_list 
# MAGIC     ON RIGHT(ticker_list.twitter_hashtag_1, length(ticker_list.twitter_hashtag_1) - 1) = RIGHT(crypto.QUERY, length(crypto.QUERY) - 1)
# MAGIC JOIN group5_finalproject.pricing_info_raw AS price
# MAGIC     ON price.ticker = ticker_list.yfinance_ticker
# MAGIC WHERE
# MAGIC     date > '{{ Date Range.start }}' AND date < '{{ Date Range.end }}'
# MAGIC AND  CASE
# MAGIC     WHEN is_member('group-a') THEN ticker IN ('SHIB-USD', 'DOT1-USD', 'LTC-USD', 'BTC-USD')
# MAGIC     WHEN is_member('group-b') THEN ticker IN ('AVAX-USD', 'ALGO-USD', 'ETH-USD', 'LUNA1-USD')
# MAGIC     WHEN is_member('group-c') THEN ticker IN ('ADA-USD', 'XLM-USD', 'SOL1-USD', 'BCH-USD','SHIB-USD', 'DOT1-USD', 'LTC-USD', 'BTC-USD')
# MAGIC     WHEN is_member('group-d') THEN ticker IN ('DOGE-USD', 'UNI3-USD', 'XRP-USD')
# MAGIC     ELSE TRUE 
# MAGIC   END
# MAGIC GROUP BY 
# MAGIC     price.date, crypto.QUERY, price.delta, price.volume, crypto.retweet_count
# MAGIC LIMIT 20000

# COMMAND ----------

# MAGIC %sql

# COMMAND ----------

# MAGIC %sql

# COMMAND ----------

# MAGIC %sql

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Other Queries

# COMMAND ----------

# DBTITLE 1,Ticker List
# MAGIC %sql
# MAGIC SELECT
# MAGIC     ticker
# MAGIC FROM group5_finalproject.gold
# MAGIC GROUP BY ticker

# COMMAND ----------

# DBTITLE 1,Silver Table - User Sentiment only
# MAGIC %sql
# MAGIC SELECT
# MAGIC     QUERY,
# MAGIC     sentiment,
# MAGIC     COUNT(QUERY) AS total
# MAGIC FROM 
# MAGIC     group5_finalproject.twitter_silver
# MAGIC GROUP BY
# MAGIC     sentiment, QUERY
# MAGIC ORDER BY
# MAGIC     total DESC

# COMMAND ----------

# DBTITLE 1,Twitter UserID @tags
# MAGIC %sql
# MAGIC SELECT
# MAGIC   DISTINCT(QUERY)
# MAGIC FROM
# MAGIC   group5_finalproject.crypto_df_tf
# MAGIC LIMIT
# MAGIC   15

# COMMAND ----------

# DBTITLE 1,Aggregated Data - All 15 cryptos, with filter for single crypto (Not in Dashboard due to security groups, used for BI exploration)
# MAGIC %sql
# MAGIC SELECT
# MAGIC   crypto.QUERY AS `Crypto::filter`, COUNT(0) AS tweets,
# MAGIC   price.date,
# MAGIC   crypto.retweet_count,
# MAGIC   price.delta,
# MAGIC   price.volume
# MAGIC 
# MAGIC FROM
# MAGIC   group5_finalproject.crypto_df_tf AS crypto
# MAGIC JOIN group5_finalproject.ticker_list_tbl AS ticker_list 
# MAGIC     ON RIGHT(ticker_list.twitter_hashtag_1, length(ticker_list.twitter_hashtag_1) - 1) = RIGHT(crypto.QUERY, length(crypto.QUERY) - 1)
# MAGIC JOIN group5_finalproject.pricing_info_raw AS price
# MAGIC     ON price.ticker = ticker_list.yfinance_ticker
# MAGIC GROUP BY 
# MAGIC     price.date, crypto.QUERY, price.delta, price.volume, crypto.retweet_count

# COMMAND ----------

# DBTITLE 1,Tweet count by Filtered Crypto - Used to attempt to build smart alarms to email based on dynamic sql input
# MAGIC %sql
# MAGIC SELECT
# MAGIC   QUERY AS `Crypto::filter`, COUNT(0)
# MAGIC FROM
# MAGIC   group5_finalproject.crypto_df_tf
# MAGIC GROUP BY
# MAGIC   QUERY
