# Databricks notebook source
# MAGIC %md
# MAGIC ###KeplerGL Visualisation Examples

# COMMAND ----------

# MAGIC %md
# MAGIC #####Install Required Libs
# MAGIC
# MAGIC Here we are specifying a version for keplerGL. At the time of building these examples, the latest version did not provide the ability to export map configurations from the UI of a map. These are used throughout this example to make the notebook rerunable with the best configs.

# COMMAND ----------

# MAGIC %pip install h3

# COMMAND ----------

# MAGIC %pip install keplergl==0.2.2

# COMMAND ----------

from keplergl import KeplerGl
from pyspark.sql.functions import hex, col, lit
from pyspark.sql import Row
import h3
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #####Databricks notebook specific function
# MAGIC
# MAGIC Produces HTML that is formatted properly in Databricks notebook

# COMMAND ----------

def create_kepler_html(data, config, height):
  map_1 = KeplerGl(height=height, data=data, config=config)
  html = map_1._repr_html_().decode("utf-8")
  new_html = html + """<script>
      var targetHeight = "{height}px";
      var interval = window.setInterval(function() {{
        if (document.body && document.body.style && document.body.style.height !== targetHeight) {{
          document.body.style.height = targetHeight;
        }}
      }}, 250);</script>""".format(height=height)
  return new_html

# COMMAND ----------

# MAGIC %md
# MAGIC ###Example 1 - Understanding Hexagons and Hierarchies
# MAGIC
# MAGIC In this first example we show how the different resolutions of hexagons in the `h3` library fit into one other in a hierarchical manner. There are functions that allow us to zoom up and down the hierarchy and we can visualise this using KeplerGL. This forms a great basis for understanding the rest of the examples.

# COMMAND ----------

# MAGIC %md
# MAGIC #####Produce a hexagon to traverse into
# MAGIC
# MAGIC First we will use a set of coordinates located in New York City to produce a hex_id at resolution 6. We can then find all the children of this hexagon using a fairly fine-grained resolution, 11.

# COMMAND ----------

lat = 40.7831
lng = -73.9712
resolution = 6
parent_h3 = h3.geo_to_h3(lat, lng, resolution)

print(parent_h3)

# COMMAND ----------

res11 = [Row(x) for x in list(h3.h3_to_children(parent_h3, 11))]

print(res11[:10])

# COMMAND ----------

# MAGIC %md
# MAGIC #####Create Dataframe 
# MAGIC
# MAGIC Now we can create a dataframe with out resolution 11 hexagons and then add columns using UDFs to get the parent values for each step up. Below we capture the parent values at resolutions 10, 9, 8, and 7.

# COMMAND ----------

schema = StructType([       
    StructField('hex_id', StringType(), True)
])

sdf = spark.createDataFrame(data=res11, schema = schema)

display(sdf)

# COMMAND ----------

@udf
def getLat(h3_id):
  return h3.h3_to_geo(h3_id)[0]

@udf
def getLong(h3_id):
  return h3.h3_to_geo(h3_id)[1]

@udf
def getParent(h3_id, parent_res):
  return h3.h3_to_parent(h3_id, parent_res)

# COMMAND ----------

pdf = ( sdf
   .withColumn("h3_res10", getParent("hex_id", lit(10)))
       .withColumn("h3_res9", getParent("hex_id", lit(9)))
       .withColumn("h3_res8", getParent("hex_id", lit(8)))
       .withColumn("h3_res7", getParent("hex_id", lit(7)))
       .withColumnRenamed('hex_id', "h3_res11")
    .toPandas() 
      )

display(pdf)

# COMMAND ----------

# DBTITLE 1,Map configs for re-run
map_config={
  "version": "v1",
  "config": {
    "visState": {
      "filters": [],
      "layers": [
        {
          "id": "mg08ej3",
          "type": "hexagonId",
          "config": {
            "dataId": "hex_data",
            "label": "h3_res11",
            "color": [
              18,
              147,
              154
            ],
            "columns": {
              "hex_id": "h3_res11"
            },
            "isVisible": False,
            "visConfig": {
              "opacity": 0.8,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "coverage": 1,
              "enable3d": False,
              "sizeRange": [
                0,
                500
              ],
              "coverageRange": [
                0,
                1
              ],
              "elevationScale": 5
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": None,
            "colorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear",
            "coverageField": None,
            "coverageScale": "linear"
          }
        },
        {
          "id": "x1gdbyj",
          "type": "hexagonId",
          "config": {
            "dataId": "hex_data",
            "label": "h3_res10",
            "color": [
              221,
              178,
              124
            ],
            "columns": {
              "hex_id": "h3_res10"
            },
            "isVisible": False,
            "visConfig": {
              "opacity": 0.8,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "coverage": 1,
              "enable3d": False,
              "sizeRange": [
                0,
                500
              ],
              "coverageRange": [
                0,
                1
              ],
              "elevationScale": 5
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": None,
            "colorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear",
            "coverageField": None,
            "coverageScale": "linear"
          }
        },
        {
          "id": "xrdgt51",
          "type": "hexagonId",
          "config": {
            "dataId": "hex_data",
            "label": "h3_res9",
            "color": [
              136,
              87,
              44
            ],
            "columns": {
              "hex_id": "h3_res9"
            },
            "isVisible": False,
            "visConfig": {
              "opacity": 0.8,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "coverage": 1,
              "enable3d": False,
              "sizeRange": [
                0,
                500
              ],
              "coverageRange": [
                0,
                1
              ],
              "elevationScale": 5
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": None,
            "colorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear",
            "coverageField": None,
            "coverageScale": "linear"
          }
        },
        {
          "id": "etz6nbl",
          "type": "hexagonId",
          "config": {
            "dataId": "hex_data",
            "label": "h3_res8",
            "color": [
              125,
              177,
              227
            ],
            "columns": {
              "hex_id": "h3_res8"
            },
            "isVisible": True,
            "visConfig": {
              "opacity": 0.15,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "coverage": 0.95,
              "enable3d": False,
              "sizeRange": [
                0,
                500
              ],
              "coverageRange": [
                0,
                1
              ],
              "elevationScale": 5
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": None,
            "colorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear",
            "coverageField": None,
            "coverageScale": "linear"
          }
        },
        {
          "id": "jn27fi",
          "type": "hexagonId",
          "config": {
            "dataId": "hex_data",
            "label": "h3_res7",
            "color": [
              117,
              222,
              227
            ],
            "columns": {
              "hex_id": "h3_res7"
            },
            "isVisible": True,
            "visConfig": {
              "opacity": 0.31,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "coverage": 0.98,
              "enable3d": False,
              "sizeRange": [
                0,
                500
              ],
              "coverageRange": [
                0,
                1
              ],
              "elevationScale": 5
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": None,
            "colorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear",
            "coverageField": None,
            "coverageScale": "linear"
          }
        }
      ],
      "interactionConfig": {
        "tooltip": {
          "fieldsToShow": {
            "hex_data": [
              {
                "name": "h3_res11",
                "format": None
              },
              {
                "name": "h3_res10",
                "format": None
              },
              {
                "name": "h3_res9",
                "format": None
              },
              {
                "name": "h3_res8",
                "format": None
              },
              {
                "name": "h3_res7",
                "format": None
              }
            ]
          },
          "compareMode": False,
          "compareType": "absolute",
          "enabled": True
        },
        "brush": {
          "size": 0.5,
          "enabled": False
        },
        "geocoder": {
          "enabled": False
        },
        "coordinate": {
          "enabled": False
        }
      },
      "layerBlending": "subtractive",
      "splitMaps": [],
      "animationConfig": {
        "currentTime": None,
        "speed": 1
      }
    },
    "mapState": {
      "bearing": 0,
      "dragRotate": False,
      "latitude": 40.79223909866621,
      "longitude": -73.99527316762153,
      "pitch": 0,
      "zoom": 13.347945326288185,
      "isSplit": False
    },
    "mapStyle": {
      "styleType": "dark",
      "topLayerGroups": {},
      "visibleLayerGroups": {
        "label": True,
        "road": True,
        "border": False,
        "building": True,
        "water": True,
        "land": True,
        "3d building": False
      },
      "threeDBuildingColor": [
        9.665468314072013,
        17.18305478057247,
        31.1442867897876
      ],
      "mapStyles": {}
    }
  }
}

# COMMAND ----------

# MAGIC %md
# MAGIC ####Display Results for Example 1

# COMMAND ----------

example_1_html = create_kepler_html(data= {"hex_data": pdf }, config=map_config, height=600)

displayHTML(example_1_html)

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Example 2 - Polygon and Hexagon Relationships
# MAGIC
# MAGIC Now we can get into using the dataset that we worked with earlier to better understand the relationships in our data and how indexing libraries work with geospatial analysis. Below we will query some of our raw and index safegraph data to show the relationship of polygon and a h3_index.

# COMMAND ----------

# MAGIC %md
# MAGIC #####Query Data
# MAGIC
# MAGIC The following query pulls all polygons in Washington DC in a particular US ZIP code ('20001'). They key components that we are interested in displaying on the map are the ploygons for the different POIs as well as the cooresponding hex_ids that were computed at resolution 13. Supporting data points include attributes such as the location name and street address.
# MAGIC
# MAGIC Here we are breaking out polygons from each individual hex_id to cut the total amount of data going into the map. because there are several hex_ids per each polygon, if we separate the Hex_ids from the rest of the POI data, we can eliminate duplicated values. This means the final HTML that is displayed is acutally less bytes total.

# COMMAND ----------

sql = """select h3.safegraph_place_id,
h3.location_name,
h3.street_address,
h3.city,
h3.region,
h3.postal_code,
cast(h3.longitude as float),
cast(h3.latitude as float),
trim(h3.h3) as hex_id,
wkt.polygon_wkt as poly
from geospatial_lakehouse_blog_db.h3_indexed_safegraph_poi h3
INNER JOIN geospatial_lakehouse_blog_db.raw_safegraph_poi wkt
  ON h3.safegraph_place_id = wkt.safegraph_place_id
where h3.postal_code = '20001'
and h3.region = 'DC'
order by h3.h3
"""

polygons_df = spark.sql(sql).drop("hex_id").distinct().toPandas()
hex_df = spark.sql(sql).select("hex_id").distinct().toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #####View data
# MAGIC
# MAGIC Now we can take a look at the two datasets we will be rendering on the KeplerGL map

# COMMAND ----------

print(polygons_df.shape)
print(hex_df.shape)

# COMMAND ----------

display(polygons_df)

# COMMAND ----------

display(hex_df)

# COMMAND ----------

# DBTITLE 1,Map configs for re-run
map_config = {
  "version": "v1",
  "config": {
    "visState": {
      "filters": [],
      "layers": [
        {
          "id": "3x3ae5l",
          "type": "hexagonId",
          "config": {
            "dataId": "hex_data",
            "label": "hex_id",
            "color": [
              255,
              153,
              31
            ],
            "columns": {
              "hex_id": "hex_id"
            },
            "isVisible": True,
            "visConfig": {
              "opacity": 0.8,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "coverage": 1,
              "enable3d": False,
              "sizeRange": [
                0,
                500
              ],
              "coverageRange": [
                0,
                1
              ],
              "elevationScale": 5
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": None,
            "colorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear",
            "coverageField": None,
            "coverageScale": "linear"
          }
        },
        {
          "id": "xhm7gpn",
          "type": "point",
          "config": {
            "dataId": "polygon_data",
            "label": "Point",
            "color": [
              18,
              147,
              154
            ],
            "columns": {
              "lat": "latitude",
              "lng": "longitude",
              "altitude": None
            },
            "isVisible": True,
            "visConfig": {
              "radius": 6.7,
              "fixedRadius": False,
              "opacity": 0.8,
              "outline": False,
              "thickness": 2,
              "strokeColor": None,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "strokeColorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "radiusRange": [
                0,
                50
              ],
              "filled": True
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": None,
            "colorScale": "quantile",
            "strokeColorField": None,
            "strokeColorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear"
          }
        },
        {
          "id": "ig82fok",
          "type": "geojson",
          "config": {
            "dataId": "polygon_data",
            "label": "polygon_data",
            "color": [
              221,
              178,
              124
            ],
            "columns": {
              "geojson": "poly"
            },
            "isVisible": True,
            "visConfig": {
              "opacity": 0.8,
              "strokeOpacity": 0.8,
              "thickness": 1.5,
              "strokeColor": [
                206,
                64,
                170
              ],
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "strokeColorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "radius": 10,
              "sizeRange": [
                0,
                10
              ],
              "radiusRange": [
                0,
                50
              ],
              "heightRange": [
                0,
                500
              ],
              "elevationScale": 5,
              "stroked": True,
              "filled": True,
              "enable3d": False,
              "wireframe": False
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": None,
            "colorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear",
            "strokeColorField": None,
            "strokeColorScale": "quantile",
            "heightField": None,
            "heightScale": "linear",
            "radiusField": None,
            "radiusScale": "linear"
          }
        }
      ],
      "interactionConfig": {
        "tooltip": {
          "fieldsToShow": {
            "polygon_data": [
              {
                "name": "safegraph_place_id",
                "format": None
              },
              {
                "name": "location_name",
                "format": None
              },
              {
                "name": "street_address",
                "format": None
              },
              {
                "name": "city",
                "format": None
              },
              {
                "name": "region",
                "format": None
              }
            ],
            "hex_data": [
              {
                "name": "hex_id",
                "format": None
              }
            ]
          },
          "compareMode": False,
          "compareType": "absolute",
          "enabled": True
        },
        "brush": {
          "size": 0.5,
          "enabled": False
        },
        "geocoder": {
          "enabled": False
        },
        "coordinate": {
          "enabled": False
        }
      },
      "layerBlending": "normal",
      "splitMaps": [],
      "animationConfig": {
        "currentTime": None,
        "speed": 1
      }
    },
    "mapState": {
      "bearing": 0,
      "dragRotate": False,
      "latitude": 38.90230007078847,
      "longitude": -77.01366163411703,
      "pitch": 0,
      "zoom": 14.137157694462616,
      "isSplit": False
    },
    "mapStyle": {
      "styleType": "dark",
      "topLayerGroups": {},
      "visibleLayerGroups": {
        "label": True,
        "road": True,
        "border": False,
        "building": True,
        "water": True,
        "land": True,
        "3d building": False
      },
      "threeDBuildingColor": [
        9.665468314072013,
        17.18305478057247,
        31.1442867897876
      ],
      "mapStyles": {}
    }
  }
}

# COMMAND ----------

# MAGIC %md
# MAGIC #####Display Results for Example 2

# COMMAND ----------

html = create_kepler_html(data={"polygon_data": polygons_df, "hex_data": hex_df }, config=map_config, height=600)
displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Example 3 - Resolution Analysis
# MAGIC
# MAGIC In our final example we can hone in on a particular POI and examine the data points around it. By filter on the National Portrait Gallery we can get its polygon as well as the hexagons that fill it in at resolution 13. To fill in a polygon using H3 library functionality, the centroid of the hexagon must fall within the bounds of the polygon. 
# MAGIC
# MAGIC Next we add a few more columns to show larger parent hexagons of the original resolution `13` hexagons. The resulting hexagons at this higher resolution are not neccesarily contained within the the bounds of the polygon. In order to obtain these, we would need to resuse the `h3.poly_fill()` method for the resultions in question. But by looking at the map we can clearly see which polygons would fit inside and which ones would get dropped off. 
# MAGIC
# MAGIC You can also tell that the larger the resolution, the less precision you have in terms of the polygon definition using hex_ids. at larger resolutions, part of a hexagon may by hanging over the boundary or there may be parts of the polygon not accounted for by a hex_id. This understanding is important when deciding your precsion to performance tradeoffs.
# MAGIC
# MAGIC Feel free to play around with the opacity settings in each layer to get a better feel for this.

# COMMAND ----------

# MAGIC %md
# MAGIC #####Query Data

# COMMAND ----------

sql = """select h3.safegraph_place_id,
h3.location_name,
h3.street_address,
h3.city,
h3.region,
h3.postal_code,
cast(h3.longitude as float),
cast(h3.latitude as float),
trim(h3.h3) as hex_id,
wkt.polygon_wkt as poly
from geospatial_lakehouse_blog_db.h3_indexed_safegraph_poi h3
INNER JOIN geospatial_lakehouse_blog_db.raw_safegraph_poi wkt
  ON h3.safegraph_place_id = wkt.safegraph_place_id
where h3.postal_code = '20001'
and h3.region = 'DC'
and h3.location_name = 'National Portrait Gallery'
order by h3.h3
"""

single_polygons_df = spark.sql(sql).drop("hex_id").distinct().toPandas()


single_hex_df = (spark.sql(sql)
                 .select("hex_id")
                 .withColumn("res_11", getParent("hex_id", lit(11)))
                 .withColumn("res_12", getParent("hex_id", lit(12)))
                 .toPandas() )

# COMMAND ----------

# DBTITLE 1,Map configs for re-run
map_config = {
  "version": "v1",
  "config": {
    "visState": {
      "filters": [],
      "layers": [
        {
          "id": "ig82fok",
          "type": "geojson",
          "config": {
            "dataId": "polygon_data",
            "label": "polygon_data",
            "color": [
              221,
              178,
              124
            ],
            "columns": {
              "geojson": "poly"
            },
            "isVisible": True,
            "visConfig": {
              "opacity": 0.18,
              "strokeOpacity": 0.8,
              "thickness": 0.5,
              "strokeColor": [
                241,
                200,
                230
              ],
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "strokeColorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "radius": 10,
              "sizeRange": [
                0,
                10
              ],
              "radiusRange": [
                0,
                50
              ],
              "heightRange": [
                0,
                500
              ],
              "elevationScale": 5,
              "stroked": True,
              "filled": True,
              "enable3d": False,
              "wireframe": False
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": None,
            "colorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear",
            "strokeColorField": None,
            "strokeColorScale": "quantile",
            "heightField": None,
            "heightScale": "linear",
            "radiusField": None,
            "radiusScale": "linear"
          }
        },
        {
          "id": "pga8xz8",
          "type": "hexagonId",
          "config": {
            "dataId": "hex_data",
            "label": "Res_13",
            "color": [
              137,
              218,
              193
            ],
            "columns": {
              "hex_id": "hex_id"
            },
            "isVisible": True,
            "visConfig": {
              "opacity": 0.8,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "coverage": 1,
              "enable3d": False,
              "sizeRange": [
                0,
                500
              ],
              "coverageRange": [
                0,
                1
              ],
              "elevationScale": 5
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": None,
            "colorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear",
            "coverageField": None,
            "coverageScale": "linear"
          }
        },
        {
          "id": "yn86mtch",
          "type": "hexagonId",
          "config": {
            "dataId": "hex_data",
            "label": "Res_12",
            "color": [
              183,
              136,
              94
            ],
            "columns": {
              "hex_id": "res_12"
            },
            "isVisible": True,
            "visConfig": {
              "opacity": 0.8,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "coverage": 1,
              "enable3d": False,
              "sizeRange": [
                0,
                500
              ],
              "coverageRange": [
                0,
                1
              ],
              "elevationScale": 5
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": None,
            "colorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear",
            "coverageField": None,
            "coverageScale": "linear"
          }
        },
        {
          "id": "8u0kqtn",
          "type": "hexagonId",
          "config": {
            "dataId": "hex_data",
            "label": "Res_11",
            "color": [
              248,
              149,
              112
            ],
            "columns": {
              "hex_id": "res_11"
            },
            "isVisible": True,
            "visConfig": {
              "opacity": 0.8,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "coverage": 1,
              "enable3d": False,
              "sizeRange": [
                0,
                500
              ],
              "coverageRange": [
                0,
                1
              ],
              "elevationScale": 5
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": None,
            "colorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear",
            "coverageField": None,
            "coverageScale": "linear"
          }
        }
      ],
      "interactionConfig": {
        "tooltip": {
          "fieldsToShow": {
            "polygon_data": [
              {
                "name": "safegraph_place_id",
                "format": None
              },
              {
                "name": "location_name",
                "format": None
              },
              {
                "name": "street_address",
                "format": None
              },
              {
                "name": "city",
                "format": None
              },
              {
                "name": "region",
                "format": None
              }
            ],
            "hex_data": [
              {
                "name": "hex_id",
                "format": None
              }
            ]
          },
          "compareMode": False,
          "compareType": "absolute",
          "enabled": True
        },
        "brush": {
          "size": 0.5,
          "enabled": False
        },
        "geocoder": {
          "enabled": False
        },
        "coordinate": {
          "enabled": False
        }
      },
      "layerBlending": "normal",
      "splitMaps": [],
      "animationConfig": {
        "currentTime": None,
        "speed": 1
      }
    },
    "mapState": {
      "bearing": 0,
      "dragRotate": False,
      "latitude": 38.89797647747516,
      "longitude": -77.02325841464138,
      "pitch": 0,
      "zoom": 17.198676745164164,
      "isSplit": False
    },
    "mapStyle": {
      "styleType": "dark",
      "topLayerGroups": {},
      "visibleLayerGroups": {
        "label": True,
        "road": True,
        "border": False,
        "building": True,
        "water": True,
        "land": True,
        "3d building": False
      },
      "threeDBuildingColor": [
        9.665468314072013,
        17.18305478057247,
        31.1442867897876
      ],
      "mapStyles": {}
    }
  }
}

# COMMAND ----------

# MAGIC %md
# MAGIC #####Display Map

# COMMAND ----------

html = create_kepler_html(data={"polygon_data": single_polygons_df, "hex_data": single_hex_df }, config=map_config, height=600)
displayHTML(html)
