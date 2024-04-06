CREATE STREAMING LIVE TABLE raw_retail
COMMENT "This is the RAW bronze input data read in with Autoloader - no optimizations or expectations"
TBLPROPERTIES ("quality" = "bronze")
AS
SELECT * FROM cloud_files('/databricks-datasets/online_retail/data-001/', "csv", 
        map(
            "schema", "InvoiceNo STRING, StockCode STRING, Description STRING, Quantity FLOAT, InvoiceDate STRING, UnitPrice FLOAT, CustomerID STRING, Country STRING",
            "header", "true"
        )
);


CREATE OR REPLACE STREAMING LIVE TABLE cleaned_retail
PARTITIONED BY (Country)
COMMENT "This is the raw bronze table with data cleaned (dates, etc.), data partitioned, and optimized"
TBLPROPERTIES --Can be spark, delta, or DLT confs
("quality"="bronze",
"pipelines.autoOptimize.managed"="true",
"pipelines.autoOptimize.zOrderCols"="CustomerID,InvoiceNo",
"pipelines.trigger.interval"="1 hour"
 )
 AS 
 SELECT * 
 FROM STREAM(LIVE.raw_retail)
 ;


 CREATE OR REPLACE STREAMING LIVE TABLE quality_retail
(
  CONSTRAINT has_customer EXPECT (CustomerID IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT has_invoice EXPECT (InvoiceNo IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT valid_date_time EXPECT (CAST(InvoiceDatetime AS TIMESTAMP) IS NOT NULL) ON VIOLATION DROP ROW
)
PARTITIONED BY (Country)
COMMENT "This is the raw bronze table with data cleaned (dates, etc.), data partitioned, and optimized"
TBLPROPERTIES 
("quality"="silver",
"pipelines.autoOptimize.managed"="true",
"pipelines.autoOptimize.zOrderCols"="CustomerID,InvoiceNo",
"pipelines.trigger.interval"="1 hour"
 )
 AS (
   WITH step1 AS 
    (SELECT 
    *,
    split(InvoiceDate, " ") AS Timestamp_Parts
    FROM STREAM(LIVE.cleaned_retail)
    ),
      step2 AS (
      SELECT 
      *,
      split(Timestamp_Parts[0], "/") AS DateParts,
      Timestamp_Parts[1] AS RawTime
      FROM step1
      ),
          step3 AS (
          SELECT
          *,
          concat(lpad(DateParts[2], 4, "20"), "-", lpad(DateParts[0], 2, "0"),"-", lpad(DateParts[1], 2, "0")) AS CleanDate,
          lpad(RawTime, 5, '0') AS CleanTime
          FROM step2
          )
  SELECT
  InvoiceNo,
  StockCode,
  Description,
  Quantity,
  CleanDate AS InvoiceDate,
  CleanTime AS InvoiceTime,
  concat(CleanDate, " ", CleanTime)::timestamp AS InvoiceDatetime,
  UnitPrice,
  CustomerID,
  Country
  FROM step3
 )
 ;


CREATE OR REPLACE STREAMING LIVE TABLE quarantined_retail
(
  CONSTRAINT has_customer EXPECT (CustomerID IS NULL) ON VIOLATION DROP ROW,
  CONSTRAINT has_invoice EXPECT (InvoiceNo IS NULL) ON VIOLATION DROP ROW,
  CONSTRAINT valid_date_time EXPECT (InvoiceDate IS NULL) ON VIOLATION DROP ROW
)
TBLPROPERTIES 
("quality"="bronze",
"pipelines.autoOptimize.managed"="true",
"pipelines.autoOptimize.zOrderCols"="CustomerID,InvoiceNo",
"pipelines.trigger.interval"="1 hour"
 )
AS 
SELECT 
 InvoiceNo,
  StockCode,
  Description,
  Quantity,
  InvoiceDate,
  UnitPrice,
  CustomerID,
  Country
FROM STREAM(LIVE.cleaned_retail);
;


CREATE OR REPLACE LIVE TABLE distinct_countries_retail
AS
SELECT DISTINCT Country
FROM LIVE.quality_retail;


CREATE OR REFRESH STREAMING LIVE TABLE retail_sales_all_countries
TBLPROPERTIES 
("quality"="silver",
"delta.tuneFileSizesForRewrites"="true",
"pipelines.autoOptimize.managed"="true",
"pipelines.autoOptimize.zOrderCols"="CustomerID,InvoiceNo",
"pipelines.trigger.interval"="1 hour"
 );
APPLY CHANGES INTO LIVE.retail_sales_all_countries
FROM STREAM(LIVE.quality_retail)
KEYS (CustomerID, InvoiceNo)
SEQUENCE BY InvoiceDateTime
;

CREATE OR REPLACE LIVE TABLE sales_by_day
AS 
SELECT
date_trunc('day',InvoiceDatetime) AS Date,
SUM(Quantity) AS TotalSales
FROM (LIVE.retail_sales_all_countries)
GROUP BY date_trunc('day',InvoiceDatetime)
ORDER BY Date;


CREATE OR REPLACE LIVE TABLE sales_by_country
AS 
SELECT
Country,
SUM(Quantity) AS TotalSales
FROM (LIVE.retail_sales_all_countries)
GROUP BY Country
ORDER BY TotalSales DESC;


CREATE OR REPLACE LIVE TABLE top_ten_customers
AS 
SELECT
CustomerID,
SUM(Quantity) AS TotalSales
FROM LIVE.retail_sales_all_countries
GROUP BY CustomerID
ORDER BY TotalSales DESC
LIMIT 10;



CREATE OR REPLACE STREAMING LIVE TABLE quality_retail_split_by_country
(
  CONSTRAINT has_customer EXPECT (CustomerID IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT has_invoice EXPECT (InvoiceNo IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT valid_date_time EXPECT (CAST(InvoiceDatetime AS TIMESTAMP) IS NOT NULL) ON VIOLATION DROP ROW
)
PARTITIONED BY (Country)
COMMENT "This is the raw bronze table with data cleaned (dates, etc.), data partitioned, and optimized"
TBLPROPERTIES 
("quality"="silver",
"pipelines.autoOptimize.managed"="true",
"pipelines.autoOptimize.zOrderCols"="CustomerID,InvoiceNo",
"pipelines.trigger.interval"="1 hour"
 )
 AS 
 SELECT * FROM STREAM(LIVE.quality_retail)
 ;

