// Databricks notebook source
//This notebook differs from the previous flow by
//changing timestamps to unix
//adding locale
//backfilling meetme user data?

// COMMAND ----------

// MAGIC %sql
// MAGIC --refresh source tables. We should stream these into delta tables to avoid the lengthy partition load times. Can add 20+ minutes to jobs
// MAGIC MSCK REPAIR TABLE datalake_protected.s_tmg_search_broadcast;
// MAGIC MSCK REPAIR TABLE datalake_protected.s_tmg_search_user_browse;
// MAGIC MSCK REPAIR TABLE datalake_protected.s_tmg_search_user_match;
// MAGIC MSCK REPAIR TABLE datalake_protected.s_tmg_broadcast_view;
// MAGIC MSCK REPAIR TABLE datalake_protected.s_tmg_broadcast_end_view;
// MAGIC MSCK REPAIR TABLE datalake_protected.s_tmg_user_profile;
// MAGIC MSCK REPAIR TABLE datalake_protected.s_tmg_economy_order_fulfilled; 
// MAGIC MSCK REPAIR TABLE datalake_protected.s_tmg_notification; 

// COMMAND ----------

// MAGIC %scala
// MAGIC import org.apache.spark.sql.functions.to_timestamp
// MAGIC //set variables for processing 7 days of event history and latest user data
// MAGIC //activity counts dates should be 24 hours from the first notification pushtime
// MAGIC //opens should ideally come in 24 hours after the last push time.
// MAGIC //user profile data we want to look at last update date
// MAGIC //val end_ts=spark.sql("select date_trunc('hour', current_timestamp()) - interval '1' hour as end_ts").as[(String)].first
// MAGIC //val start_ts=spark.sql("select date_trunc('hour', current_timestamp()) - interval '7' day as start_ts").as[(String)].first
// MAGIC //val start_user_ts=spark.sql("select date_trunc('hour', max(update_ts)) - interval '2' day from ml_push.temp_user_profile_snapshot").as[(String)].first
// MAGIC 
// MAGIC 
// MAGIC 
// MAGIC //val starting_timestamp= to_timestamp("01/18/2023 00:00:00", "MM/dd/yyyy HH:mm:ss")
// MAGIC 
// MAGIC 
// MAGIC val push_send_start_ts=spark.sql("select date_trunc('hour', current_timestamp()) - interval '193' hour as push_start_ts").as[(String)].first //8 days plus 1 hour
// MAGIC val push_send_end_ts=spark.sql("select date_trunc('hour', current_timestamp()) - interval '25' hour as push_send_end_ts").as[(String)].first //1 days plus 1 hour
// MAGIC val push_open_end_ts=spark.sql("select date_trunc('hour', current_timestamp()) - interval '1' hour as push_open_end_ts").as[(String)].first // 1 hour, wait i day for pushes to come in
// MAGIC 
// MAGIC val push_activity_start_ts=spark.sql("select date_trunc('hour', current_timestamp()) - interval '217' hour as push_activity_start_ts").as[(String)].first // 9 days 1 hour. 
// MAGIC val push_activity_end_ts=spark.sql("select date_trunc('hour', current_timestamp()) - interval '25' hour as push_activity_end_ts").as[(String)].first //1 days plus 1 hour
// MAGIC  
// MAGIC 
// MAGIC val start_user_ts=spark.sql("select date_trunc('hour', max(update_ts)) - interval '2' day from ml_push.temp_user_profile_snapshot").as[(String)].first
// MAGIC val end_user_ts=spark.sql("select date_trunc('hour', current_timestamp()) - interval '1' hour as end_ts").as[(String)].first

// COMMAND ----------

// MAGIC %sql
// MAGIC --drop the temp tables
// MAGIC DROP TABLE IF EXISTS ml_push.push_demographics_v3;
// MAGIC DROP TABLE IF EXISTS ml_push.temp_push;
// MAGIC DROP TABLE IF EXISTS ml_push.temp_push_activity_counts;

// COMMAND ----------

// MAGIC %python
// MAGIC ##clear out temp tables
// MAGIC dbutils.fs.rm("dbfs:/mnt/tmg-prod-ml-outputs/push_data/temp/push_demographics_v3/",True)
// MAGIC dbutils.fs.rm("dbfs:/mnt/tmg-prod-ml-outputs/push_data/temp/temp_push/",True)
// MAGIC dbutils.fs.rm("dbfs:/mnt/tmg-prod-ml-outputs/push_data/temp/temp_push_activity_counts/",True)

// COMMAND ----------

// MAGIC %sql
// MAGIC CREATE OR REPLACE TABLE ml_push.push_demographics_v3(
// MAGIC     utc_hour BIGINT,
// MAGIC     send_ts BIGINT,
// MAGIC     user_id STRING,
// MAGIC     network_user_id STRING,
// MAGIC     common__network STRING,
// MAGIC     event_type STRING,
// MAGIC     event_status STRING,
// MAGIC     notification_type STRING,
// MAGIC     notification_name STRING,
// MAGIC     from_user_id STRING,
// MAGIC     from_user_network STRING,
// MAGIC     service_name STRING,
// MAGIC     device_type STRING,
// MAGIC     open_flag INT,
// MAGIC     open_ts BIGINT,
// MAGIC     registration_ts BIGINT,
// MAGIC     age INT,
// MAGIC     gender string,
// MAGIC     country string,
// MAGIC     locale string,
// MAGIC     broadcast_search_count_24h BIGINT,
// MAGIC     search_user_count_24h BIGINT,
// MAGIC     search_match_count_24h BIGINT,
// MAGIC     view_count_24h BIGINT,
// MAGIC     view_end_count_24h BIGINT,
// MAGIC     gift_count_24h BIGINT,
// MAGIC     from_user_gender string,
// MAGIC     from_user_age INT,
// MAGIC     from_user_country string,
// MAGIC     from_user_locale string,
// MAGIC     source_correlation_id string
// MAGIC     )
// MAGIC USING DELTA
// MAGIC  PARTITIONED BY (utc_hour)  
// MAGIC     LOCATION  'dbfs:/mnt/tmg-prod-ml-outputs/push_data/temp/push_demographics_v3/'

// COMMAND ----------

// MAGIC %sql
// MAGIC CREATE OR REPLACE TABLE ml_push.temp_push(
// MAGIC     utc_hour TIMESTAMP,
// MAGIC     send_ts TIMESTAMP,
// MAGIC     user_id STRING,
// MAGIC     network_user_id STRING,
// MAGIC     common__network STRING,
// MAGIC     event_type STRING,
// MAGIC     event_status STRING,
// MAGIC     notification_type STRING,
// MAGIC     notification_name STRING,
// MAGIC     from_user_id STRING,
// MAGIC     from_user_network STRING,
// MAGIC     service_name STRING,
// MAGIC     device_type STRING,
// MAGIC     open_flag INT,
// MAGIC     open_ts TIMESTAMP,
// MAGIC     source_correlation_id STRING 
// MAGIC     )
// MAGIC USING DELTA
// MAGIC  PARTITIONED BY (utc_hour)  
// MAGIC     LOCATION  'dbfs:/mnt/tmg-prod-ml-outputs/push_data/temp/temp_push/'
// MAGIC    

// COMMAND ----------

// MAGIC %sql
// MAGIC create table ml_push.temp_push_activity_counts(
// MAGIC utc_hour timestamp,
// MAGIC send_ts timestamp, 
// MAGIC user_id string,
// MAGIC device string,
// MAGIC activity_type string,
// MAGIC counts long,
// MAGIC source_correlation_id string
// MAGIC )
// MAGIC USING DELTA
// MAGIC  PARTITIONED BY (utc_hour)  
// MAGIC     LOCATION  'dbfs:/mnt/tmg-prod-ml-outputs/push_data/temp/temp_push_activity_counts/'

// COMMAND ----------

// MAGIC 
// MAGIC %scala
// MAGIC //update the user profile table snapshot data
// MAGIC import org.apache.spark.sql.expressions.Window
// MAGIC import org.apache.spark.sql.functions._
// MAGIC import io.delta.tables._
// MAGIC 
// MAGIC val df_user= spark.sql("""
// MAGIC select 
// MAGIC     from_unixtime(body.message_timestamp/1000) message_timestamp,
// MAGIC     COALESCE(body.data.user_id.member0, body.data.user_id.member1) member_id,
// MAGIC     body.data.user_network network,
// MAGIC     body.data.birth_date birth_date,
// MAGIC     body.data.gender gender,
// MAGIC     body.data.location.country country, 
// MAGIC     body.data.locale locale,
// MAGIC     from_unixtime(body.data.registration_ts/1000) registration_ts
// MAGIC   from 
// MAGIC     datalake_protected.s_tmg_user_profile  
// MAGIC   where
// MAGIC     _processing_timestamp>= timestamp('""" + start_user_ts + "') and date(_processing_timestamp)< timestamp('" + end_user_ts +"')").as("users")
// MAGIC 
// MAGIC val partitionWindow1 = Window.partitionBy($"member_id", $"network")
// MAGIC val partitionWindow2 = Window.partitionBy($"member_id", $"network").orderBy($"birth_date", $"gender", $"country", $"locale", $"registration_ts")
// MAGIC 
// MAGIC 
// MAGIC val max_users=df_user
// MAGIC .withColumn("max_ts", max($"message_timestamp") over (partitionWindow1))
// MAGIC .filter($"message_timestamp"===$"max_ts")
// MAGIC .drop($"message_timestamp")
// MAGIC .withColumnRenamed("max_ts", "update_ts").distinct()
// MAGIC 
// MAGIC val max_demos=max_users
// MAGIC .withColumn("ranking", row_number().over(partitionWindow2))
// MAGIC .filter($"ranking"===1).drop($"ranking")
// MAGIC 
// MAGIC 
// MAGIC val deltaTableUsers = DeltaTable.forPath(spark, "dbfs:/mnt/tmg-prod-ml-outputs/push_data/temp/temp_user_profile_snapshot/")
// MAGIC 
// MAGIC deltaTableUsers
// MAGIC   .as("users")
// MAGIC   .merge(
// MAGIC     max_demos.as("updates"),
// MAGIC     "users.member_id = updates.member_id and users.network = updates.network")
// MAGIC   .whenMatched
// MAGIC   .updateExpr(
// MAGIC     Map(
// MAGIC       "member_id" -> "updates.member_id",
// MAGIC       "network" -> "updates.network",
// MAGIC       "birth_date" -> "updates.birth_date",
// MAGIC       "gender" -> "updates.gender",
// MAGIC       "country" -> "updates.country",
// MAGIC       "locale" -> "updates.locale",
// MAGIC       "registration_ts" -> "updates.registration_ts",
// MAGIC       "update_ts" -> "updates.update_ts"
// MAGIC     ))
// MAGIC   .whenNotMatched
// MAGIC   .insertExpr(
// MAGIC     Map(
// MAGIC       "member_id" -> "updates.member_id",
// MAGIC       "network" -> "updates.network",
// MAGIC       "birth_date" -> "updates.birth_date",
// MAGIC       "gender" -> "updates.gender",
// MAGIC       "country" -> "updates.country",
// MAGIC       "locale" -> "updates.locale",
// MAGIC       "registration_ts" -> "updates.registration_ts",
// MAGIC        "update_ts" -> "updates.update_ts"
// MAGIC     ))
// MAGIC   .execute()

// COMMAND ----------

// MAGIC %scala
// MAGIC import org.apache.spark.sql.functions._
// MAGIC 
// MAGIC val send_data=spark.sql(s"""SELECT 
// MAGIC                     date_trunc('hour', From_unixtime(common__timestamp / 1000)) utc_hour,
// MAGIC                     date_trunc('minute', From_unixtime(common__timestamp / 1000)) send_ts,
// MAGIC                     user_id,
// MAGIC                     network_user_id,
// MAGIC                     common__network,
// MAGIC                     event_type,
// MAGIC                     event_status,
// MAGIC                     notification_type,
// MAGIC                     notification_name,
// MAGIC                     from_user_id,
// MAGIC                     from_user_network,
// MAGIC                     service_name,
// MAGIC                     device_type,
// MAGIC                     metadata.source_correlation_id 
// MAGIC                  FROM   datalake_protected.s_tmg_notification 
// MAGIC                  WHERE  event_status = 'success' 
// MAGIC                  AND    network = 'meetme' 
// MAGIC                  AND    event_type = 'send' 
// MAGIC                  AND  _processing_timestamp>= timestamp('""" + push_send_start_ts + "') and date(_processing_timestamp)< timestamp('" + push_send_end_ts +"""')
// MAGIC                  AND notification_name IN ('broadcastStartWithDescriptionPush', 
// MAGIC                                               'broadcastStartPush') 
// MAGIC """
// MAGIC ).withColumn("max_time",  $"send_ts" + expr("INTERVAL 24 HOUR")).as("send")
// MAGIC 
// MAGIC val open_data= spark.sql(s"""
// MAGIC  SELECT date_trunc('hour', From_unixtime(common__timestamp / 1000)) utc_hour,
// MAGIC                     date_trunc('minute', From_unixtime(common__timestamp / 1000)) open_ts,
// MAGIC                     user_id,
// MAGIC                     network_user_id,
// MAGIC                     common__network,
// MAGIC                     event_type,
// MAGIC                     event_status,
// MAGIC                     notification_type,
// MAGIC                     notification_name,
// MAGIC                     from_user_id,
// MAGIC                     from_user_network,
// MAGIC                     service_name,
// MAGIC                     device_type,
// MAGIC                     metadata.source_correlation_id open_source_correlation_id
// MAGIC                  FROM   datalake_protected.s_tmg_notification 
// MAGIC                  WHERE  event_status = 'success' 
// MAGIC                  AND    network = 'meetme' 
// MAGIC                  AND    event_type = 'open' 
// MAGIC                  AND  _processing_timestamp>= timestamp('""" + push_send_start_ts + "') and date(_processing_timestamp)< timestamp('" + push_open_end_ts +"""')
// MAGIC                  AND notification_name IN ('broadcastStartWithDescriptionPush', 
// MAGIC                                               'broadcastStartPush')""").as('open)
// MAGIC 
// MAGIC //AND _processing_timestamp >= date('2022-11-24') AND _processing_timestamp >= date('2022-11-24') interval '7' days
// MAGIC // AND _processing_timestamp >= date('2022-11-24') AND _processing_timestamp >= date('2022-11-24') interval '8' days
// MAGIC 
// MAGIC 
// MAGIC val joined=send_data.join(open_data, 
// MAGIC                           $"send.network_user_id" === $"open.network_user_id" && 
// MAGIC                           $"source_correlation_id"===$"open_source_correlation_id",
// MAGIC                            "leftouter").distinct()  
// MAGIC 
// MAGIC 
// MAGIC val push_joined=joined
// MAGIC .withColumn("open_flag", when($"open.network_user_id".isNotNull,1)
// MAGIC             .otherwise(0))
// MAGIC .drop($"open.network_user_id")
// MAGIC 
// MAGIC val pushes=push_joined
// MAGIC .select($"send.utc_hour", 
// MAGIC         $"send.send_ts",
// MAGIC         $"send.user_id", 
// MAGIC         $"send.network_user_id", 
// MAGIC         $"send.common__network", 
// MAGIC         $"send.event_type",
// MAGIC         $"send.event_status",
// MAGIC         $"send.notification_type", 
// MAGIC         $"send.notification_name", 
// MAGIC         $"send.from_user_id", 
// MAGIC         $"send.from_user_network", 
// MAGIC         $"send.service_name", 
// MAGIC         $"send.device_type",
// MAGIC         $"open_flag",
// MAGIC         $"open.open_ts",
// MAGIC         $"send.source_correlation_id")
// MAGIC 
// MAGIC pushes.write.mode("append").format("delta").save("dbfs:/mnt/tmg-prod-ml-outputs/push_data/temp/temp_push/")

// COMMAND ----------

// MAGIC 
// MAGIC %scala
// MAGIC // Get the counts of search broadcast for look back window of 24 hours of the send
// MAGIC 
// MAGIC import org.apache.spark.sql.functions._
// MAGIC import org.apache.spark.sql.expressions.Window
// MAGIC 
// MAGIC spark.conf.set("spark.sql.shuffle.partitions", 10000)
// MAGIC 
// MAGIC 
// MAGIC 
// MAGIC val df_search=spark.sql(s"""select
// MAGIC     date_trunc('hour', from_unixtime(request_time/1000)) utc_hour,
// MAGIC     date_trunc('minute', from_unixtime(request_time / 1000)) utc_time,
// MAGIC     user.member_id,
// MAGIC     device_info.name device
// MAGIC   from
// MAGIC     datalake_protected.s_tmg_search_broadcast
// MAGIC   where   _processing_timestamp>= timestamp('""" + push_activity_start_ts + "') and date(_processing_timestamp)< timestamp('" + push_activity_end_ts +"""')
// MAGIC     and common__network = 'meetme'
// MAGIC """
// MAGIC ).as("search")
// MAGIC 
// MAGIC // _processing_timestamp >= date('2022-11-24') - interval '25' hour and _processing_timestamp < date('2022-11-24') + interval '7' days
// MAGIC 
// MAGIC 
// MAGIC val partitionWindow = Window.partitionBy($"member_id", $"send_ts").orderBy($"utc_time".desc)
// MAGIC 
// MAGIC 
// MAGIC val df_push=spark.sql("Select distinct user_id, send_ts, utc_hour from ml_push.temp_push").withColumn("prev_hour",  $"send_ts" - expr("INTERVAL 24 HOUR")).as("push")
// MAGIC 
// MAGIC 
// MAGIC val joined=df_push.hint("range_join", 1000).join(df_search, $"push.user_id" === $"search.member_id" && $"push.send_ts" > $"search.utc_time" && $"search.utc_time" > $"push.prev_hour").withColumn("device", first($"device", true) over (partitionWindow)) 
// MAGIC 
// MAGIC val cnts=joined.groupBy($"send_ts", $"member_id", $"device").count().withColumn("activity_type", lit("broadcast_search")).withColumnRenamed("count", "counts").withColumn("utc_hour", date_trunc("hour", $"send_ts")).withColumnRenamed("member_id", "user_id")
// MAGIC      
// MAGIC cnts.select("utc_hour", "send_ts", "user_id", "device", "activity_type", "counts").write.mode("append").format("delta").save("dbfs:/mnt/tmg-prod-ml-outputs/push_data/temp/temp_push_activity_counts/")

// COMMAND ----------

// MAGIC %scala
// MAGIC //Get the counts of search match for look back window of 24 hours of the send
// MAGIC 
// MAGIC val df_search=spark.sql(s"""select
// MAGIC     date_trunc('hour', from_unixtime(request_time/1000)) utc_hour,
// MAGIC     date_trunc('minute', from_unixtime(request_time / 1000)) utc_time,
// MAGIC     user.member_id,
// MAGIC     device_info.name device
// MAGIC   from
// MAGIC     datalake_protected.s_tmg_search_user_match
// MAGIC   where
// MAGIC      _processing_timestamp>= timestamp('""" + push_activity_start_ts + "') and date(_processing_timestamp)< timestamp('" + push_activity_end_ts +"""')
// MAGIC     and common__network = 'meetme'"""
// MAGIC ).as("search")
// MAGIC 
// MAGIC //   _processing_timestamp >= date('2022-11-24') - interval '25' hour and _processing_timestamp < date('2022-11-24') + interval '7' days
// MAGIC val partitionWindow = Window.partitionBy($"member_id", $"send_ts").orderBy($"utc_time".desc)
// MAGIC 
// MAGIC 
// MAGIC 
// MAGIC val df_push=spark.sql(s"Select distinct user_id, send_ts, utc_hour from ml_push.temp_push").withColumn("prev_hour",  $"send_ts" - expr("INTERVAL 24 HOUR")).as("push")
// MAGIC 
// MAGIC val joined=df_push.hint("range_join", 1000).join(df_search, $"push.user_id" === $"search.member_id" && $"push.send_ts" > $"search.utc_time" && $"search.utc_time" > $"push.prev_hour").withColumn("device", first($"device", true) over (partitionWindow)) 
// MAGIC 
// MAGIC val cnts=joined.groupBy($"send_ts", $"member_id", $"device").count().withColumn("activity_type", lit("match_search")).withColumnRenamed("count", "counts").withColumn("utc_hour", date_trunc("hour", $"send_ts")).withColumnRenamed("member_id", "user_id")
// MAGIC      
// MAGIC cnts.select("utc_hour", "send_ts", "user_id", "device", "activity_type", "counts").write.mode("append").format("delta").save("dbfs:/mnt/tmg-prod-ml-outputs/push_data/temp/temp_push_activity_counts/")

// COMMAND ----------

// MAGIC %scala
// MAGIC //Get the counts of search user browse for look back window of 24 hours of the send
// MAGIC spark.conf.set("spark.sql.shuffle.partitions", 10000)
// MAGIC 
// MAGIC 
// MAGIC val df_search=spark.sql(s"""select
// MAGIC     date_trunc('hour', from_unixtime(request_time/1000)) utc_hour,
// MAGIC     date_trunc('minute', from_unixtime(request_time / 1000)) utc_time,
// MAGIC     user.member_id,
// MAGIC     device_info.name device
// MAGIC   from
// MAGIC     datalake_protected.s_tmg_search_user_browse
// MAGIC   where
// MAGIC    _processing_timestamp>= timestamp('""" + push_activity_start_ts + "') and date(_processing_timestamp)< timestamp('" + push_activity_end_ts +"""')
// MAGIC     and common__network = 'meetme'"""
// MAGIC ).as("search")
// MAGIC 
// MAGIC //  _processing_timestamp >= date('2022-11-24') - interval '25' hour and _processing_timestamp < date('2022-11-24') + interval '7' days
// MAGIC 
// MAGIC val partitionWindow = Window.partitionBy($"member_id", $"send_ts").orderBy($"utc_time".desc)
// MAGIC 
// MAGIC 
// MAGIC val df_push=spark.sql(s"Select distinct user_id, send_ts, utc_hour from ml_push.temp_push").withColumn("prev_hour",  $"send_ts" - expr("INTERVAL 24 HOUR")).as("push")
// MAGIC 
// MAGIC val joined=df_push.hint("range_join", 1000).join(df_search, $"push.user_id" === $"search.member_id" && $"push.send_ts" > $"search.utc_time" && $"search.utc_time" > $"push.prev_hour").withColumn("device", first($"device", true) over (partitionWindow))  
// MAGIC 
// MAGIC val cnts=joined.groupBy($"send_ts", $"member_id", $"device").count().withColumn("activity_type", lit("user_browse_search")).withColumnRenamed("count", "counts").withColumn("utc_hour", date_trunc("hour", $"send_ts")).withColumnRenamed("member_id", "user_id")
// MAGIC      
// MAGIC cnts.select("utc_hour", "send_ts", "user_id", "device", "activity_type", "counts").write.mode("append").format("delta").save("dbfs:/mnt/tmg-prod-ml-outputs/push_data/temp/temp_push_activity_counts/")

// COMMAND ----------

// MAGIC %scala
// MAGIC //Get the counts of search views for look back window of 24 hours of the send
// MAGIC import org.apache.spark.sql.functions._
// MAGIC import org.apache.spark.sql.expressions.Window
// MAGIC 
// MAGIC 
// MAGIC spark.conf.set("spark.sql.shuffle.partitions", 10000)
// MAGIC 
// MAGIC 
// MAGIC val df_search=spark.sql(s"""select
// MAGIC     date_trunc('hour', from_unixtime(body.publishedtime)) utc_hour,
// MAGIC     date_trunc('minute', from_unixtime(body.publishedtime)) utc_time,
// MAGIC     body.socialnetwork.memberid member_id,
// MAGIC     body.deviceinfo.name device
// MAGIC   from
// MAGIC     datalake_protected.s_tmg_broadcast_view
// MAGIC   where
// MAGIC      _processing_timestamp>= timestamp('""" + push_activity_start_ts + "') and date(_processing_timestamp)< timestamp('" + push_activity_end_ts +"""')
// MAGIC     and body.socialnetwork.name = 'meetme'"""
// MAGIC ).as("search")
// MAGIC 
// MAGIC val partitionWindow = Window.partitionBy($"member_id", $"send_ts").orderBy($"utc_time".desc)
// MAGIC 
// MAGIC //   _processing_timestamp >= date('2022-11-24') - interval '25' hour and _processing_timestamp < date('2022-11-24') + interval '7' days
// MAGIC 
// MAGIC val df_push=spark.sql(s"Select distinct user_id, send_ts, utc_hour from ml_push.temp_push").withColumn("prev_hour",  $"send_ts" - expr("INTERVAL 24 HOUR")).as("push")
// MAGIC 
// MAGIC val joined=df_push.hint("range_join", 1000).join(df_search, $"push.user_id" === $"search.member_id" && $"push.send_ts" > $"search.utc_time" && $"search.utc_time" > $"push.prev_hour").withColumn("device", first($"device", true) over (partitionWindow))  
// MAGIC 
// MAGIC 
// MAGIC 
// MAGIC val cnts=joined.groupBy($"send_ts", $"member_id", $"device").count().withColumn("activity_type", lit("broadcast_view")).withColumnRenamed("count", "counts").withColumn("utc_hour", date_trunc("hour", $"send_ts")).withColumnRenamed("member_id", "user_id")
// MAGIC      
// MAGIC cnts.select("utc_hour", "send_ts", "user_id", "device", "activity_type", "counts").write.mode("append").format("delta").save("dbfs:/mnt/tmg-prod-ml-outputs/push_data/temp/temp_push_activity_counts/")

// COMMAND ----------

// MAGIC %scala
// MAGIC //Get the counts of view end for look back window of 24 hours of the send
// MAGIC spark.conf.set("spark.sql.shuffle.partitions", 10000)
// MAGIC 
// MAGIC 
// MAGIC val df_search=spark.sql(s"""select
// MAGIC      date_trunc('hour', from_unixtime(body.publishedtime)) utc_hour,
// MAGIC      date_trunc('minute', from_unixtime(body.publishedtime )) utc_time,
// MAGIC      body.socialnetwork.memberid member_id,
// MAGIC      body.deviceinfo.name device
// MAGIC   from
// MAGIC     datalake_protected.s_tmg_broadcast_end_view
// MAGIC   where
// MAGIC      _processing_timestamp>= timestamp('""" + push_activity_start_ts + "') and date(_processing_timestamp)< timestamp('" + push_activity_end_ts +"""')
// MAGIC     and body.socialnetwork.name = 'meetme'"""
// MAGIC ).as("search")
// MAGIC 
// MAGIC // _processing_timestamp >= date('2022-11-24') - interval '25' hour and _processing_timestamp < date('2022-11-24') + interval '7' days
// MAGIC 
// MAGIC val partitionWindow = Window.partitionBy($"member_id", $"send_ts").orderBy($"utc_time".desc)
// MAGIC 
// MAGIC 
// MAGIC val df_push=spark.sql(s"Select distinct user_id, send_ts, utc_hour from ml_push.temp_push").withColumn("prev_hour",  $"send_ts" - expr("INTERVAL 24 HOUR")).as("push")
// MAGIC 
// MAGIC val joined=df_push.hint("range_join", 1000).join(df_search, $"push.user_id" === $"search.member_id" && $"push.send_ts" > $"search.utc_time" && $"search.utc_time" > $"push.prev_hour").withColumn("device", first($"device", true) over (partitionWindow))    
// MAGIC 
// MAGIC val cnts=joined.groupBy($"send_ts", $"member_id", $"device").count().withColumn("activity_type", lit("broadcast_view_end")).withColumnRenamed("count", "counts").withColumn("utc_hour", date_trunc("hour", $"send_ts")).withColumnRenamed("member_id", "user_id")
// MAGIC      
// MAGIC cnts.select("utc_hour", "send_ts", "user_id", "device", "activity_type", "counts").write.mode("append").format("delta").save("dbfs:/mnt/tmg-prod-ml-outputs/push_data/temp/temp_push_activity_counts/")

// COMMAND ----------

// MAGIC %scala
// MAGIC //Get the counts of video gifts for look back window of 24 hours of the send
// MAGIC 
// MAGIC import org.apache.spark.sql.functions._
// MAGIC spark.conf.set("spark.sql.shuffle.partitions", 10000)
// MAGIC 
// MAGIC 
// MAGIC val df_orders=spark.sql("""select
// MAGIC     date_trunc('hour', from_unixtime(body.order.orderdate/1000)) utc_hour,
// MAGIC     date_trunc('minute', from_unixtime(body.order.orderdate/1000)) utc_time,
// MAGIC     split(body.order.purchaser, ':')[0] purchaser_network,
// MAGIC     split(body.order.purchaser, ':')[2] purchaser_member_id,
// MAGIC     body.order.orderstatus orderstatus,
// MAGIC     body.order.products
// MAGIC   from
// MAGIC   datalake_protected.s_tmg_economy_order_fulfilled
// MAGIC    where  
// MAGIC      _processing_timestamp>= timestamp('""" + push_activity_start_ts + "') and date(_processing_timestamp)< timestamp('" + push_activity_end_ts +"""')
// MAGIC    and body.order.purchaser like '%meetme%'
// MAGIC    """)
// MAGIC 
// MAGIC   //   _processing_timestamp >= date('2022-11-24') - interval '25' hour and _processing_timestamp < date('2022-11-24') + interval '7' days 
// MAGIC //get orders with exploded prodcucts
// MAGIC val df_products=df_orders.select($"utc_hour", $"utc_time", $"purchaser_network", $"purchaser_member_id", $"orderstatus", explode($"products.product").as("product"))
// MAGIC 
// MAGIC //get video gifts with filters
// MAGIC val df_filtered=df_products
// MAGIC .select($"utc_hour", 
// MAGIC         $"utc_time", 
// MAGIC         $"purchaser_network", 
// MAGIC         $"purchaser_member_id", 
// MAGIC         $"orderstatus", 
// MAGIC         $"product.producttype".as("producttype"),
// MAGIC         $"product.exchange.currency".as("exchange_currency"),
// MAGIC         $"product.purchase.currency".as("purchase_currency"),
// MAGIC         $"product.categories".as("categories")
// MAGIC        ).filter($"producttype"==="video-gift" && $"orderstatus"==="FULFILLED" && $"exchange_currency"==="DMD" && $"purchase_currency"=!="USD")
// MAGIC .where(not(array_contains($"categories","transfer")))
// MAGIC .as("gift")
// MAGIC 
// MAGIC //get pushes 
// MAGIC 
// MAGIC //get pushes 
// MAGIC val df_push=spark
// MAGIC .sql(s"Select distinct user_id, common__network as network, from_user_id, from_user_network, send_ts, utc_hour from ml_push.temp_push")
// MAGIC .withColumn("prev_hour",  $"utc_hour" - expr("INTERVAL 24 HOUR"))
// MAGIC .as("push")
// MAGIC 
// MAGIC //join gift and pushes
// MAGIC val joined=df_push.hint("range_join", 1000).join(df_filtered, $"push.user_id" === $"purchaser_member_id" && $"push.network" === $"purchaser_network" &&  $"push.send_ts" > $"gift.utc_time" && $"gift.utc_time" > $"push.prev_hour") 
// MAGIC 
// MAGIC 
// MAGIC //aggregate counts
// MAGIC val cnts=joined.groupBy($"send_ts", $"user_id").count().withColumn("activity_type", lit("gifts")).withColumnRenamed("count", "counts").withColumn("utc_hour", date_trunc("hour", $"send_ts")).withColumn("device", lit(null))  
// MAGIC 
// MAGIC cnts.select("utc_hour", "send_ts", "user_id", "device", "activity_type", "counts").write.mode("append").format("delta").save("dbfs:/mnt/tmg-prod-ml-outputs/push_data/temp/temp_push_activity_counts/")

// COMMAND ----------

// MAGIC %sql
// MAGIC --TO DO. add QA checks for counts and raise errors if doenst pass

// COMMAND ----------

// MAGIC %sql
// MAGIC CREATE OR REPLACE FUNCTION get_age(event_date timestamp, birth_date date)
// MAGIC   RETURNS INT
// MAGIC   COMMENT 'calculates a user age based on event_date and birth_date'
// MAGIC   
// MAGIC   RETURN CAST((datediff(date_trunc('day',event_date), birth_date)/365.25) as int) 

// COMMAND ----------

// MAGIC %sql
// MAGIC CREATE OR REPLACE FUNCTION get_account_age(event_date timestamp, registration_date date)
// MAGIC   RETURNS long
// MAGIC   COMMENT 'calculates a user account age based on event_date and registration time'
// MAGIC   
// MAGIC   RETURN CAST((datediff(date_trunc('day',event_date), registration_date)) as long) 

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC --- do do, convert from sql code
// MAGIC with push_data as (
// MAGIC SELECT 
// MAGIC   a.utc_hour,
// MAGIC   a.send_ts ,
// MAGIC   a.user_id ,
// MAGIC   a.network_user_id ,
// MAGIC   a.common__network ,
// MAGIC   a.event_type ,
// MAGIC   a.event_status ,
// MAGIC   a.notification_type ,
// MAGIC   a.notification_name ,
// MAGIC   a.from_user_id ,
// MAGIC   a.from_user_network,
// MAGIC   a.service_name ,
// MAGIC   COALESCE(a.device_type , b.device, c.device, e.device, f.device) device_type,   --- find out what device cyrus is using, and see what we need for serving time, he will probably use ios.
// MAGIC   a.open_flag ,
// MAGIC   a.open_ts,
// MAGIC   h.registration_ts, -- look at using this to calculate account age.
// MAGIC   get_age(a.send_ts, from_unixtime(h.birth_date/1000)) age,
// MAGIC   h.gender,
// MAGIC   h.country,
// MAGIC   h.locale,
// MAGIC   coalesce(b.counts, 0) broadcast_search_count_24h,
// MAGIC   coalesce(c.counts, 0) search_user_count_24h,
// MAGIC   coalesce(d.counts, 0) search_match_count_24h,
// MAGIC   coalesce(e.counts, 0) view_count_24h,
// MAGIC   coalesce(f.counts, 0) view_end_count_24h,
// MAGIC   coalesce(g.counts, 0) gift_count_24h,
// MAGIC   i.gender from_user_gender,
// MAGIC   get_age(a.send_ts, from_unixtime(i.birth_date/1000)) from_user_age,
// MAGIC   i.country from_user_country,
// MAGIC   i.locale from_user_locale,
// MAGIC   a.source_correlation_id
// MAGIC from 
// MAGIC  ml_push.temp_push a
// MAGIC  left join ml_push.temp_push_activity_counts b on a.user_id=b.user_id and a.send_ts =b.send_ts and b.activity_type="broadcast_search"
// MAGIC  left join ml_push.temp_push_activity_counts c on a.user_id=c.user_id and a.send_ts =c.send_ts and c.activity_type="user_browse_search"
// MAGIC  left join ml_push.temp_push_activity_counts d on a.user_id=d.user_id and a.send_ts =d.send_ts and d.activity_type="match_search"
// MAGIC  left join ml_push.temp_push_activity_counts e on a.user_id=e.user_id and a.send_ts =e.send_ts and e.activity_type="broadcast_view"
// MAGIC  left join ml_push.temp_push_activity_counts f on a.user_id=f.user_id and a.send_ts =f.send_ts and f.activity_type="broadcast_view_end"
// MAGIC  left join ml_push.temp_push_activity_counts g on a.user_id=g.user_id and a.send_ts =g.send_ts and g.activity_type="gifts"
// MAGIC  left join ml_push.temp_user_profile_snapshot h on  a.user_id=h.member_id and a.common__network=h.network 
// MAGIC  left join ml_push.temp_user_profile_snapshot i on  a.from_user_id=i.member_id and a.from_user_network=i.network 
// MAGIC )
// MAGIC 
// MAGIC  INSERT INTO 
// MAGIC ml_push.push_demographics_v3(
// MAGIC     utc_hour,
// MAGIC     send_ts ,
// MAGIC     user_id ,
// MAGIC     network_user_id ,
// MAGIC     common__network ,
// MAGIC     event_type ,
// MAGIC     event_status ,
// MAGIC     notification_type ,
// MAGIC     notification_name ,
// MAGIC     from_user_id ,
// MAGIC     from_user_network ,
// MAGIC     service_name ,
// MAGIC     device_type ,
// MAGIC     open_flag ,
// MAGIC     open_ts ,
// MAGIC     registration_ts ,
// MAGIC     age ,
// MAGIC     gender ,
// MAGIC     country ,
// MAGIC     locale,
// MAGIC     broadcast_search_count_24h ,
// MAGIC     search_user_count_24h ,
// MAGIC     search_match_count_24h ,
// MAGIC     view_count_24h ,
// MAGIC     view_end_count_24h ,
// MAGIC     gift_count_24h ,
// MAGIC     from_user_gender ,
// MAGIC     from_user_age ,
// MAGIC     from_user_country,
// MAGIC     from_user_locale,
// MAGIC     source_correlation_id) 
// MAGIC  select 
// MAGIC     unix_timestamp(utc_hour) * 1000 utc_hour,
// MAGIC     unix_timestamp(send_ts) * 1000 send_ts,
// MAGIC     user_id ,
// MAGIC     network_user_id ,
// MAGIC     common__network ,
// MAGIC     event_type ,
// MAGIC     event_status ,
// MAGIC     notification_type ,
// MAGIC     notification_name ,
// MAGIC     from_user_id ,
// MAGIC     from_user_network ,
// MAGIC     service_name ,
// MAGIC     device_type ,
// MAGIC     open_flag ,
// MAGIC     unix_timestamp(open_ts) * 1000 open_ts,
// MAGIC     unix_timestamp(registration_ts) * 1000 registration_ts,
// MAGIC     age ,
// MAGIC     gender ,
// MAGIC     country ,
// MAGIC     locale,
// MAGIC     broadcast_search_count_24h ,
// MAGIC     search_user_count_24h ,
// MAGIC     search_match_count_24h ,
// MAGIC     view_count_24h ,
// MAGIC     view_end_count_24h ,
// MAGIC     gift_count_24h ,
// MAGIC     from_user_gender ,
// MAGIC     from_user_age ,
// MAGIC     from_user_country ,
// MAGIC     from_user_locale,
// MAGIC     source_correlation_id
// MAGIC   from push_data  
// MAGIC   
// MAGIC  

// COMMAND ----------

// MAGIC %sql
// MAGIC select count(*) from  ml_push.push_demographics_v3
// MAGIC where source_correlation_id is null

// COMMAND ----------

// MAGIC %sql
// MAGIC select 
// MAGIC to_date(from_unixtime(utc_hour/1000)) send_date,
// MAGIC utc_hour,
// MAGIC send_ts, 
// MAGIC user_id, 
// MAGIC network_user_id,
// MAGIC common__network,
// MAGIC event_type,event_status,
// MAGIC notification_type,
// MAGIC notification_name,
// MAGIC from_user_id,
// MAGIC from_user_network,
// MAGIC service_name,
// MAGIC device_type,
// MAGIC open_flag, 
// MAGIC open_ts,
// MAGIC registration_ts,
// MAGIC age ,
// MAGIC gender ,
// MAGIC country ,
// MAGIC locale,
// MAGIC broadcast_search_count_24h ,
// MAGIC search_user_count_24h ,
// MAGIC search_match_count_24h ,
// MAGIC view_count_24h ,
// MAGIC view_end_count_24h ,
// MAGIC gift_count_24h,
// MAGIC from_user_gender,
// MAGIC from_user_age ,
// MAGIC from_user_country ,
// MAGIC from_user_locale,
// MAGIC source_correlation_id
// MAGIC from 
// MAGIC ml_push.push_demographics_v3 
// MAGIC where to_date(from_unixtime(utc_hour/1000)) =date('2023-01-17')
// MAGIC group by 1,
// MAGIC utc_hour,
// MAGIC send_ts, 
// MAGIC user_id, 
// MAGIC network_user_id,
// MAGIC common__network,
// MAGIC event_type,
// MAGIC event_status,
// MAGIC notification_type,
// MAGIC notification_name,
// MAGIC from_user_id,
// MAGIC from_user_network,
// MAGIC service_name,
// MAGIC device_type,
// MAGIC open_flag, 
// MAGIC open_ts,
// MAGIC registration_ts,
// MAGIC age ,
// MAGIC gender ,
// MAGIC country ,
// MAGIC locale,
// MAGIC broadcast_search_count_24h ,
// MAGIC search_user_count_24h ,
// MAGIC search_match_count_24h ,
// MAGIC view_count_24h ,
// MAGIC view_end_count_24h ,
// MAGIC gift_count_24h,
// MAGIC from_user_gender,
// MAGIC from_user_age ,
// MAGIC from_user_country ,
// MAGIC from_user_locale,
// MAGIC source_correlation_id
// MAGIC having count(*)>1
// MAGIC limit 10
