// Databricks notebook source
// MAGIC %md
// MAGIC
// MAGIC ## Graph Analysis with GraphFrames
// MAGIC This notebook goes over basic graph analysis using [the GraphFrames package available on spark-packages.org](http://spark-packages.org/package/graphframes/graphframes). The goal of this notebook is to show you how to use GraphFrames to perform graph analysis. You're going to be doing this with Bay area bike share data from [Kaggle](https://www.kaggle.com/benhamner/sf-bay-area-bike-share/downloads/sf-bay-area-bike-share.zip).
// MAGIC
// MAGIC #### Graph Theory and Graph Processing
// MAGIC Graph processing is an important aspect of analysis that applies to a lot of use cases. Fundamentally graph theory and processing are about defining relationships between different nodes and edges. Nodes or vertices are the units while edges are the relationships that are defined between those. This works great for social network analysis and running algorithms like [PageRank](https://en.wikipedia.org/wiki/PageRank) to better understand and weigh relationships.
// MAGIC
// MAGIC Some business use cases could be to look at central people in social networks [who is most popular in a group of friends], importance of papers in bibliographic networks [which papers are most referenced], and of course ranking web pages!
// MAGIC
// MAGIC #### Graphs and Bike Trip Data
// MAGIC As mentioned, in this example you'll be using Bay area bike share data. The way you're going to orient your analysis is by making every vertex a station and each trip will become an edge connecting two stations. This creates a *directed* graph.
// MAGIC
// MAGIC **Further Reference:**
// MAGIC * [Graph Theory on Wikipedia](https://en.wikipedia.org/wiki/Graph_theory)
// MAGIC * [PageRank on Wikipedia](https://en.wikipedia.org/wiki/PageRank)
// MAGIC
// MAGIC #### **Table of Contents**
// MAGIC * **Create DataFames**
// MAGIC * **Imports**
// MAGIC * **Building the Graph**
// MAGIC * **PageRank**
// MAGIC * **Trips from Station to Station**
// MAGIC * **In Degrees and Out Degrees**

// COMMAND ----------

// MAGIC %md ### Create DataFrames

// COMMAND ----------

val bikeStations = spark.read.format("csv")
    .option("header", "true")
    .load("dbfs:/tmp/station.csv")

val tripData = spark.read.format("csv")
    .option("header", "true")
    .load("dbfs:/tmp/trip.csv")

// COMMAND ----------

display(bikeStations)

// COMMAND ----------

display(tripData)

// COMMAND ----------

// MAGIC %md It can often times be helpful to look at the exact schema to ensure that you have the right types associated with the right columns.

// COMMAND ----------

bikeStations.printSchema()
tripData.printSchema()

// COMMAND ----------

bikeStations.write.format("delta").mode("overwrite").save("dbfs:/tmp/delta/bike")

// COMMAND ----------

tripData.write.format("delta").mode("overwrite").save("dbfs:/tmp/delta/trip")

// COMMAND ----------

// MAGIC %md 
// MAGIC ### Imports
// MAGIC You're going to need to import several things before you can continue. You're going to import a variety of SQL functions that are going to make working with DataFrames much easier and you're going to import everything that you're going to need from GraphFrames.

// COMMAND ----------

import org.apache.spark.sql._
import org.apache.spark.sql.functions._

import org.graphframes._

// COMMAND ----------

// MAGIC %md
// MAGIC ### Build the Graph
// MAGIC Now that you've imported your data, you're going to need to build your graph. To do so you're going to do two things. You are going to build the structure of the vertices (or nodes) and you're going to build the structure of the edges. What's awesome about GraphFrames is that this process is incredibly simple. All that you need to do get the distinct **id** values in the Vertices table and rename the start and end stations to **src** and **dst** respectively for your edges tables. These are required conventions for vertices and edges in GraphFrames.

// COMMAND ----------

val bikeStations = spark.read.format("delta").load("dbfs:/tmp/delta/bike")
val tripData = spark.read.format("delta").load("dbfs:/tmp/delta/trip")

// COMMAND ----------

val stationVertices = bikeStations
  .distinct()

val tripEdges = tripData
  .withColumnRenamed("start_station_name", "src")
  .withColumnRenamed("end_station_name", "dst")

// COMMAND ----------

display(stationVertices)

// COMMAND ----------

display(tripEdges)

// COMMAND ----------

// MAGIC %md Now you can build your graph. 
// MAGIC
// MAGIC You're also going to cache the input DataFrames to your graph.

// COMMAND ----------

val stationGraph = GraphFrame(stationVertices, tripEdges)

tripEdges.cache()
stationVertices.cache()

// COMMAND ----------

println("Total Number of Stations: " + stationGraph.vertices.count)
println("Total Number of Trips in Graph: " + stationGraph.edges.count)
println("Total Number of Trips in Original Data: " + tripData.count)// sanity check

// COMMAND ----------

// MAGIC %md
// MAGIC ### Trips From Station to Station
// MAGIC One question you might ask is what are the most common destinations in the dataset from location to location. You can do this by performing a grouping operator and adding the edge counts together. This will yield a new graph except each edge will now be the sum of all of the semantically same edges. Think about it this way: you have a number of trips that are the exact same from station A to station B, you just want to count those up!
// MAGIC
// MAGIC In the below query you'll see that you're going to grab the station to station trips that are most common and print out the top 10.

// COMMAND ----------

val topTrips = stationGraph
  .edges
  .groupBy("src", "dst")
  .count()
  .orderBy(desc("count"))
  .limit(10)

display(topTrips)

// COMMAND ----------

// MAGIC %md You can see above that a given vertex being a Caltrain station seems to be significant! This makes sense as these are natural connectors and likely one of the most popular uses of these bike share programs to get you from A to B in a way that you don't need a car!

// COMMAND ----------

// MAGIC %md 
// MAGIC ### In Degrees and Out Degrees
// MAGIC Remember that in this instance you've got a directed graph. That means that your trips are directional - from one location to another. Therefore you get access to a wealth of analysis that you can use. You can find the number of trips that go into a specific station and leave from a specific station.
// MAGIC
// MAGIC Naturally you can sort this information and find the stations with lots of inbound and outbound trips! Check out this definition of [Vertex Degrees](http://mathworld.wolfram.com/VertexDegree.html) for more information.
// MAGIC
// MAGIC Now that you've defined that process, go ahead and find the stations that have lots of inbound and outbound traffic.

// COMMAND ----------

val inDeg = stationGraph.inDegrees
display(inDeg.orderBy(desc("inDegree")).limit(5))

// COMMAND ----------

val outDeg = stationGraph.outDegrees
display(outDeg.orderBy(desc("outDegree")).limit(5))

// COMMAND ----------

// MAGIC %md One interesting follow up question you could ask is what is the station with the highest ratio of in degrees but fewest out degrees. As in, what station acts as almost a pure trip sink. A station where trips end at but rarely start from.

// COMMAND ----------

val degreeRatio = inDeg.join(outDeg, inDeg.col("id") === outDeg.col("id"))
  .drop(outDeg.col("id"))
  .selectExpr("id", "double(inDegree)/double(outDegree) as degreeRatio")

degreeRatio.cache()
  
display(degreeRatio.orderBy(desc("degreeRatio")).limit(10))

// COMMAND ----------

// MAGIC %md 
// MAGIC You can do something similar by getting the stations with the lowest in degrees to out degrees ratios, meaning that trips start from that station but don't end there as often. This is essentially the opposite of what you have above.

// COMMAND ----------

display(degreeRatio.orderBy(asc("degreeRatio")).limit(10))

// COMMAND ----------

// MAGIC %md The conclusions of what you get from the above analysis should be relatively straightforward. If you have a higher value, that means many more trips come into that station than out, and a lower value means that many more trips leave from that station than come into it!
// MAGIC
// MAGIC Hopefully you've gotten some value out of this notebook! Graph stuctures are everywhere once you start looking for them and hopefully GraphFrames will make analyzing them easy!
