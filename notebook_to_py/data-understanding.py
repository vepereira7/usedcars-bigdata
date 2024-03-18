from io import BytesIO, StringIO
from google.cloud import storage

import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf 
from pyspark.sql import types
from pyspark.sql import functions as F
from pyspark.ml.feature import Imputer, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.sql.types import StringType, IntegerType, StructType, StructField, DoubleType
from pyspark.sql.functions import when, upper, avg, year, to_date, sqrt, log, lower, col, row_number, asc, lit, count, expr, percentile_approx, monotonically_increasing_id, udf, skewness, regexp_extract
from pyspark.sql.window import Window
#from plotly.offline import iplot
#import plotly.graph_objs as go
from pyspark.sql.functions import round
#import plotly.express as px
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, IsotonicRegression, FMRegressor, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor, GeneralizedLinearRegression
from pyspark.ml import Pipeline

# Utilities
import os

# Numpy & Pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import pandas_profiling as pp
import time
import psutil

# Others (warnings etc)
from warnings import simplefilter

# Create a SparkSession
#spark = SparkSession.builder.config('...').master('yarn').appName('egd').getOrCreate()
spark = SparkSession.builder.config('spark.driver.memory', '1g').config('spark.executor.memory', '4g') \
.config('spark.executor.instances', '6').config('spark.executor.cores','2').config('spark.driver.maxResultSize', '1g') \
.master('yarn').appName('egd').getOrCreate()

file_path = 'gs://egd-project-vp-1/egd-project/notebooks_jupyter_notebooks_jupyter_vehicles.csv'
df = (spark.read.format('csv').option('header', 'true').option('inferSchema','true').load(file_path))

# Show the first three rows of the DataFrame
df.show(n=3, truncate=False)


df.printSchema()

# # ignore all future warnings
# simplefilter(action='ignore', category=FutureWarning)

# Define a function to show values on bar charts
def show_values_on_bars(axs, space=0.4):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.0f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", va="bottom") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


print('----------------------------------------')
print('---------- DATA UNDERSTANDING ----------')
print('----------------------------------------')
print('')

manufacturer_counts = (
    df.groupBy('manufacturer')
    .agg({'manufacturer': 'count', 'price': 'mean'})
    .withColumnRenamed('count(manufacturer)', 'num_listings')
    .withColumnRenamed('median(price)', 'avg_price')
    .orderBy('num_listings', ascending=False)
)
manufacturer_counts.show()

print('Visualize sum of car postings and average price per car manufacturer')
import pyspark.sql.functions as F
from pyspark.sql.functions import lit
import matplotlib.pyplot as plt

# create a bar chart of number of listings by manufacturer
# plt.figure(figsize=(25,6))
manufacturer_counts_filtered = manufacturer_counts.filter(F.col('manufacturer').isNotNull() & F.col('num_listings').isNotNull())
manufacturer_counts_updated = manufacturer_counts_filtered.select('manufacturer', 'num_listings').orderBy('num_listings', ascending=False).collect()
manufacturer = [row.manufacturer for row in manufacturer_counts_updated]
num_listings = [row.num_listings for row in manufacturer_counts_updated]
# plt.bar(manufacturer, num_listings)
# plt.title('Number of Listings by Manufacturer')
# plt.xlabel('Manufacturer')
# plt.ylabel('Number of Listings')
# plt.xticks(rotation=90)
# # save the plot
# # try:
# # 	bucket_name = 'egd-project-vp-1'
# # 	blob_name = 'egd-project/Number of Listings by Manufacturer.png'  # specify the folder name
# # 	client = storage.Client()
# # 	bucket = client.bucket(bucket_name)
# # 	blob = bucket.blob(blob_name)
# # 	with plt.rc_context({'figure.dpi': 300}):  # set DPI value to 300
# # 	    plt.savefig('/tmp/plot.png')
# # 	with open('/tmp/plot.png', 'rb') as f:
# # 	    blob.upload_from_file(f)
# # 	print('Plot saved successfully!')
# # except:
# # 	print('Plot was not saved!')
# plt.show()
# manufacturer.show(20)
# num_listings.show(20)

# create a scatterplot of average price vs. number of listings by manufacturer
# plt.figure(figsize=(25,6))
# df_scatter = manufacturer_counts.select('num_listings', 'avg(price)').collect()
# num_listings = [row['num_listings'] for row in df_scatter]
# avg_price = [row['avg(price)'] for row in df_scatter]
# plt.scatter(num_listings, avg_price)
# plt.title('Average Price vs. Number of Listings by Manufacturer')
# plt.xlabel('Number of Listings')
# plt.ylabel('Average Price')
# plt.show()

print('Trying to categorize the number of online dealership, physical dealership, and private party dealer to the best of my ability')

online_dealerships = ['carvana', 'vroom', 'shift', 'carMax']
physical_dealerships = ['finance', 'call', 'guaranteed', 'inspection', 'test drive', 'call us today', 'auction', 'visit our', 'automotive']

def categorize_description(description):
    if description is None:
        return 'Private party'
    elif any(keyword in description.lower() for keyword in online_dealerships):
        return 'Online dealership'
    elif any(keyword in description.lower() for keyword in physical_dealerships):
        return 'Physical dealership'
    else:
        return 'Private party'
    
categorize_description_udf = udf(categorize_description, StringType())

# apply the function to each row of the DataFrame
df = df.withColumn('category', categorize_description_udf('description'))

# calculate the percentage of descriptions in each category
category_counts = df.groupby('category').count()
category_counts = category_counts.withColumn('percentage', category_counts['count'] / df.count() * 100).orderBy('percentage', ascending=False)
category_counts.show(20)

# # collect the data
# category_counts_list = category_counts.collect()

# # extract the data into separate arrays
# categories = [row['category'] for row in category_counts_list]
# percentages = [row['percentage'] for row in category_counts_list]
# # create a figure and axis object
# fig, ax = plt.subplots()

# # set the x and y labels, title, and rotation of x-tick labels
# ax.set_xlabel('Category')
# ax.set_ylabel('Percentage')
# ax.set_title('Percentage of Car Listings by Category')
# ax.set_xticklabels(categories, rotation=0)

# # create the bar chart
# x_pos = np.arange(len(categories))
# ax.bar(x_pos, percentages)
# ax.set_xticks(x_pos)

# # show the chart
# plt.show()

print('What are the oldest cars?')

df = df.orderBy('year')
df.show(10)

# create a new DataFrame with the excluded rows removed
df_excluded = df.filter(~(
    lower(col("description")).like("%cash for%") |
    lower(col("description")).like("%provide photos%") |
    lower(col("description")).like("%buying%")
))

# extract the year from the description column
df_excluded = df_excluded.withColumn("year", regexp_extract(col("description"), r"\b(19[0-9][0-9]|20[0-2][0-9])\b", 0))

# convert year column to integer type
df_excluded = df_excluded.withColumn("year", df_excluded["year"].cast("integer"))

# sort the remaining rows by year and show the top 5 oldest cars
oldest_cars = df_excluded.filter(df_excluded.year.isNotNull()).sort("year").select('year', 'price', 'type', 'description').limit(5)
oldest_cars.show()

print('What is the average price and sum of listings per state - plot4')

# define a UDF to calculate the median
median_udf = expr('percentile_approx(price, 0.5)')

# group by state and calculate count of listings and median price
state_counts = (df.groupBy('state')
                .agg(count('state').alias('num_listings'), median_udf.alias('median_price'))
                .withColumnRenamed('count(state)', 'num_listings')
                .withColumnRenamed('median(price)', 'avg_price'))

# add a new column with row numbers
window = Window.orderBy(asc('state'))
state_counts = (state_counts.withColumn('row_num', row_number().over(window))
                .withColumn('state', upper('state'))  # uppercase state column
                .drop('row_num'))  # drop row_num column

state_counts.show()



print('What percentage of postings for each state is electric cars')

# Filter the dataframe for electric cars
electric_cars = df.filter(df['fuel'] == 'electric')

# Group the data by state and count occurrences
state_counts = electric_cars.groupBy('state').agg(count('*').alias('count')).orderBy('count', ascending=False)

# Calculate the total number of electric cars
total_electric_cars = state_counts.agg({'count': 'sum'}).collect()[0][0]

# Calculate the percentage of electric cars in each state
state_percentages = state_counts.withColumn('percentage', state_counts['count'] / total_electric_cars * 100)

# Display the result
state_percentages.show()


print('What percentage of postings for each state is salvaged cars')

# Filter the dataframe for salvage
salvage_cars = df.filter(df.title_status == 'salvage')

# Group the data by state and count occurrences
salvage_counts = salvage_cars.groupBy('state').agg(count('*').alias('count'))

# Calculate the total number of salvage cars
total_salvage_cars = salvage_counts.agg({'count': 'sum'}).collect()[0][0]

# Calculate the percentage of salvage cars in each state and round to 2 decimal places
state_percentages = (salvage_counts.withColumn('percentage', round(salvage_counts['count'] / total_salvage_cars * 100, 2))
                    .select('state', 'percentage'))

# Sort the data by percentage in descending order
state_percentages = state_percentages.orderBy('percentage', ascending=False)

# Display the result
print(state_percentages.show())