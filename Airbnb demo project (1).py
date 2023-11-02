# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/AB_NYC_2019.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC
# MAGIC select * from `AB_NYC_2019.csv`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "AB_NYC_2019.csv"

# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("AirbnbAnalysis").getOrCreate()

# Load CSV data into a DataFrame
nyc_df = spark.read.csv("dbfs:/FileStore/tables/AB_NYC_2019.csv", header=True, inferSchema=True)


# COMMAND ----------

# Display the first few rows of the DataFrame
display(nyc_df.head())



# COMMAND ----------

# Show basic statistics of numerical columns
nyc_df.describe().show()


# COMMAND ----------

from pyspark.sql.functions import col, count, lit

# Calculate missing value proportions for each column
missing_values = nyc_df.select(
    *[
        (count(col(c)) / count("*")).alias(c + "_missing") 
        for c in nyc_df.columns
    ]
)

# Add a row to display the results
missing_values_union = missing_values.union(
    nyc_df.agg(*[lit(1).alias(c + "_missing") for c in nyc_df.columns])
)

display(missing_values_union)


# COMMAND ----------

# Drop rows with missing values
nyc_df_cleaned = nyc_df.dropna()

# Show the count before and after dropping missing values
print("Count before dropping missing values:", nyc_df.count())
print("Count after dropping missing values:", nyc_df_cleaned.count())

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import seaborn as sns


# COMMAND ----------

spark = SparkSession.builder.appName("DataRelationships").getOrCreate()


# COMMAND ----------


nyc_df = spark.read.csv('dbfs:/FileStore/tables/AB_NYC_2019.csv', header=True, inferSchema=True)


# COMMAND ----------

#Exploration Data and Visualizing
import matplotlib.pyplot as plt
import seaborn as sns


# COMMAND ----------

# Create a histogram for price
plt.figure(figsize=(10, 6))
sns.histplot(data=nyc_df.toPandas(), x='price', bins=30, kde=True)
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# COMMAND ----------


nyc_df.printSchema()


# COMMAND ----------

# Load your data into a DataFrame (Replace 'your_file_path' with actual file path)
nyc_df = spark.read.csv('dbfs:/FileStore/tables/AB_NYC_2019.csv', header=True, inferSchema=True)


# COMMAND ----------

numeric_columns = ['price', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
for column in numeric_columns:
    nyc_df = nyc_df.withColumn(column, col(column).cast('float'))


# COMMAND ----------

nyc_df = nyc_df.dropna()


# COMMAND ----------

train_data, test_data = nyc_df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_data)


# COMMAND ----------

predictions = model.transform(test_data)


# COMMAND ----------

evaluator = RegressionEvaluator(labelCol='price', predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")


# COMMAND ----------

nyc_df = spark.read.csv('dbfs:/FileStore/tables/AB_NYC_2019.csv', header=True, inferSchema=True)

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Select the numerical columns for visualization
numerical_columns = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']

# Convert the DataFrame to Pandas for visualization
nyc_pd = nyc_df.select(numerical_columns).toPandas()

# Create histograms for numerical columns
nyc_pd.hist(bins=20, figsize=(12, 10))
plt.show()


# COMMAND ----------

#Box Plot by Room Type

room_price_data = nyc_df.select('room_type', 'price')

# Convert the DataFrame to Pandas for visualization
room_price_pd = room_price_data.toPandas()

# Create box plots for price by room type
plt.figure(figsize=(10, 6))
sns.boxplot(data=room_price_pd, x='room_type', y='price')
plt.xlabel('Room Type')
plt.ylabel('Price')
plt.title('Distribution of Prices by Room Type')
plt.show()


# COMMAND ----------

#latitude and longitude columns
lat_long_data = nyc_df.select('latitude', 'longitude')

# Convert the DataFrame to Pandas for visualization
lat_long_pd = lat_long_data.toPandas()

# Create a scatter plot of latitude and longitude
plt.figure(figsize=(10, 8))
sns.scatterplot(data=lat_long_pd, x='longitude', y='latitude', alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographical Distribution of Listings')
plt.show()


# COMMAND ----------

# piechart of room types
room_type_counts = nyc_df.groupBy('room_type').count()

# Convert the DataFrame to Pandas for visualization
room_type_counts_pd = room_type_counts.toPandas()

# Create a pie chart of room types
plt.figure(figsize=(8, 8))
plt.pie(room_type_counts_pd['count'], labels=room_type_counts_pd['room_type'], autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Room Types')
plt.show()


# COMMAND ----------

from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

# Initialize Spark session
spark = SparkSession.builder.appName("NeighbourhoodPrice").getOrCreate()

neighbourhood_price = nyc_df.groupBy("neighbourhood").mean("price").orderBy("neighbourhood")

# Convert DataFrame to Pandas for visualization
neighbourhood_price_pd = neighbourhood_price.toPandas()

# Create a bar plot
plt.figure(figsize=(14, 12))
plt.bar(neighbourhood_price_pd["neighbourhood"], neighbourhood_price_pd["avg(price)"])
plt.xlabel("Neighbourhood")
plt.ylabel("Mean Price")
plt.title("Mean Price vs Neighbourhood")
plt.xticks(rotation=120)
plt.tight_layout()
plt.show()



# COMMAND ----------

# Load your data into a DataFrame (Replace 'your_file_path' with actual file path)
nyc_df = spark.read.csv('dbfs:/FileStore/tables/AB_NYC_2019.csv', header=True, inferSchema=True)


# COMMAND ----------


# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming nyc_df is your DataFrame
room_type_counts = nyc_df.groupBy("room_type").count().orderBy("count")

# Convert DataFrame to Pandas for plotting
room_type_counts_pd = room_type_counts.toPandas()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=room_type_counts_pd, x="room_type", y="count", palette="pastel")
plt.xlabel("Room Type")
plt.ylabel("Number of Listings")
plt.title("Distribution of Listings by Room Type")
plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjust rotation and alignment
plt.tight_layout()
plt.show()


# COMMAND ----------

# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming nyc_df is your DataFrame
room_type_counts = nyc_df.groupBy("room_type").count().orderBy("count")

# Convert DataFrame to Pandas for plotting
room_type_counts_pd = room_type_counts.toPandas()
# Create a bar plot
plt.figure(figsize=(24, 6))  # Increase the figure size
sns.barplot(data=room_type_counts_pd, x="room_type", y="count", palette="pastel")
plt.xlabel("Room Type")
plt.ylabel("Number of Listings")
plt.title("Distribution of Listings by Room Type")
plt.xticks(rotation=45, ha='left', fontsize=12)
plt.tight_layout()
plt.show()


# COMMAND ----------

from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark session
spark = SparkSession.builder.appName("PriceDistribution").getOrCreate()

# Assuming nyc_df is your DataFrame
price_data = nyc_df.select("price").filter(nyc_df.price.isNotNull())  # Remove null values

# Convert DataFrame to Pandas for visualization
price_data_pd = price_data.toPandas()

# Create a histogram
plt.figure(figsize=(10, 6))
sns.histplot(price_data_pd, x="price", bins=30, kde=True, color="blue")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Price Distribution")
plt.show()


# COMMAND ----------

from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark session
spark = SparkSession.builder.appName("CategoricalAnalysis").getOrCreate()

# Assuming nyc_df is your DataFrame
room_type_counts = nyc_df.groupBy("room_type").count().orderBy("count", ascending=False)
borough_counts = nyc_df.groupBy("neighbourhood_group").count().orderBy("count", ascending=False)

# Convert DataFrames to Pandas for visualization
room_type_counts_pd = room_type_counts.toPandas()
borough_counts_pd = borough_counts.toPandas()

# Create bar plots for room types
plt.figure(figsize=(24, 6))
sns.barplot(data=room_type_counts_pd, x="room_type", y="count", palette="viridis")
plt.xlabel("Room Type")
plt.ylabel("Count")
plt.title("Room Type Distribution")
plt.xticks(rotation=45)
plt.show()

# Create bar plots for borough distributions
plt.figure(figsize=(24, 6))
sns.barplot(data=borough_counts_pd, x="neighbourhood_group", y="count", palette="viridis")
plt.xlabel("Borough")
plt.ylabel("Count")
plt.title("Borough Distribution")
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------


