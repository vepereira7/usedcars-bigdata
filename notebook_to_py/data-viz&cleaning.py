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


print('-----------------------------------------')
print('---------- DATA VIZ & CLEANING ----------')
print('-----------------------------------------')
print('')


# Determine and remove the columns to drop based on the above graph
cols_to_drop = ['id','url', 'region', 'region_url', 'VIN', 'image_url', 'description', \
    'county', 'size', 'paint_color', 'drive', 'cylinders', 'state', 'lat','long']
vehicles_df = df.select([col(c) for c in df.columns if c not in cols_to_drop])

# Remove null values and duplicated rows
vehicles_df = vehicles_df.dropna().dropDuplicates()

# Get info of the new data frame
vehicles_df.printSchema()

# Preview the new dataframe
list(vehicles_df)

print('\nUpon reviewing the previews and descriptions of the continuous variables, it becomes clear that some initial cleaning is necessary. This will be continued accordingly.\n')

# Drop NaNs and duplicates
vehicles_df = vehicles_df.dropna().dropDuplicates()

# Add index column using monotonically_increasing_id() function
vehicles_df = vehicles_df.withColumn("index", monotonically_increasing_id())

# Change data type of year to string
vehicles_df = vehicles_df.withColumn("year", col("year").cast("string"))

# Reorder columns with index first
vehicles_df = vehicles_df.select("index", *vehicles_df.columns[:-1])

print('\nVisualizing the data reveals patterns that are not obvious to the human eye when reviewing raw data.')
print('Correlation matrices, histograms, category, scatter & box plots have helped identify relationships.\n')

print('Cleaned data based on visualizations:\n')

print('\t- Removed NaNs & duplicates')
print('\t- Price b/w 2k and 50k')
print('\t- Odometer b/w 100 and 200k, etc..')

print('\nIn the section below, features that would help with better prediction are identified.\n')

# Describing the dataset to get a basic idea of the non-categorical features
vehicles_df.select([col(c).cast("float") for c in vehicles_df.columns if c not in ['manufacturer', 'model', 'condition', 'fuel', 'title_status', 'transmission', 'type']]).describe().show()

# # Set seaborn style
# sns.set_style('whitegrid')

# # Looking at the target column "price" first
# f, ax = plt.subplots(figsize=(12, 8))
# ax.set_title('Price Distribution', pad=12)

# # Convert spark dataframe to pandas dataframe
# price_df = vehicles_df.select(col("price")).toPandas()

# # Plot histogram using pandas dataframe
# sns.histplot(data=price_df, x="price", stat='count', bins=5)
# show_values_on_bars(ax)


print('\nIt appears that the price ranges between 0 and an unrealistic $3.7B')
print('To keep things simple and realistic, making a subset of prices between 2k and 50k')


# Create a new Spark DataFrame with the filtered data
vehicles_prc = vehicles_df.filter((vehicles_df.price >= 2000) & (vehicles_df.price <= 50000))

# Extract the data from the DataFrame as a list
prices = vehicles_prc.select('price').rdd.flatMap(lambda x: x).collect()

# # Plot the histogram
# f, ax = plt.subplots(figsize=(12, 8))
# ax.set_title('Price Distribution', pad=12)
# ax.hist(prices, bins=20)
# show_values_on_bars(ax)

# Calculate skewness of odometer column
odometer_skew = vehicles_prc.select(skewness("odometer")).collect()[0][0]
print(f"\nSkewness for odometer: {odometer_skew:.2f}\n")

# # Create a Pandas DataFrame from the Spark DataFrame
# vehicles_prc_pd = vehicles_prc.select("odometer").toPandas()

# # Plot histogram with Seaborn
# sns.displot(data=vehicles_prc_pd, x="odometer", aspect=2, height=5, kde=True)

# # Show the plot
# plt.show()


print("""It's evident that the distribution is highly skewed and there's some bad data with max odometer readings of 10mil miles etc.

Let's work on cleaning up some of that data.

Doing some research, We found that Americans drive an average of 14,300 miles per year, according to the Federal Highway Administration.

Let's look at the entries for odometer = 0 and odometer > 200k.\n""")

# Describe vehicles with 0 odometer
vehicles_prc.filter(col("odometer") == 0).describe().show()

# Describe vehicles with odometer over 200,000
vehicles_prc.filter(col("odometer") > 200000).describe().show()


print('Based on the stats above, We can make a fair assumption that odometer readings be between 100 (CPO) to 200k (20 yo) will be a good dataset to continue with.\n')

# Filtering the dataset and verifying again
vehicles_odo = vehicles_prc.filter((col("odometer") > 100) & (col("odometer") <= 200000))

# Displaying statistics
vehicles_odo.select("odometer").describe().show()

# # Displaying skewness
# skewness = vehicles_prc.select(F.skewness("odometer")).collect()[0][0]
# print(f"\nSkewness for odometer:\t{skewness:.2f}")

# # Set the figure size
# plt.figure(figsize=(12, 6))

# # Displaying distribution plot
# sns.set(style="whitegrid")
# sns.histplot(data=vehicles_odo.toPandas(), x="odometer", kde=True)
# plt.show()

# # Displaying distribution plot using Spark's built-in visualization tools
# display(vehicles_odo.select('odometer'))

# Log
odo_log = vehicles_odo.select(log('odometer').alias('odometer_log'))
odo_log_skew = odo_log.select(F.skewness('odometer_log').alias('skewness')).collect()[0]['skewness']
print(f"\nSkewness for Log of Odometer Readings::\t: {odo_log_skew:.2f}")

# # Plot
# odo_log_pd = odo_log.toPandas()
# sns.displot(data=odo_log_pd, x='odometer_log', aspect=2, height=5, kde=True, legend=True)
# plt.xlabel('Log of Odometer Readings')
# plt.show()


# Square Root
odo_sqrt = vehicles_odo.select(sqrt('odometer').alias('odometer_sqrt'))
odo_sqrt_skew = odo_sqrt.select(F.skewness('odometer_sqrt').alias('skewness')).collect()[0]['skewness']
print(f"\nSkewness for Square Root of Odometer Readings::\t: {odo_sqrt_skew:.2f}")

# # Plot
# odo_sqrt_pd = odo_sqrt.toPandas()
# sns.displot(data=odo_sqrt_pd, x='odometer_sqrt', aspect=2, height=5, kde=True, legend=True)
# plt.xlabel('Square Root of Odometer Readings')
# plt.show()


# fig, ax = plt.subplots(figsize=(20, 10))
# ax.set_title('Price vs Year', pad=12)
# sns_df = vehicles_odo.select('year', 'price').toPandas()

# # Ordenando os anos
# sns_df = sns_df.sort_values(by='year')

# sns.boxplot(x='year', y='price', data=sns_df, ax=ax)
# plt.xticks(rotation=90)

print("""
	\n It appears that there is some inconsistency in the first 2/3rds of the dataset.\n

Price seems to consistently rise 2000 onwards until about 2021; and there seems to be some bad data for 2022 as well.\n

Filtering the dataset between 2000 and 2020 for further analysis. \n""")

year_list = list(range(2000, 2021))
vehicles_year = vehicles_odo.filter(col('year').cast(IntegerType()).isin(year_list))

# # Collect the year column as a list and pass it to the x parameter of boxplot
# year_collected = [row[0] for row in vehicles_year.select(col('year').cast(IntegerType())).collect()]
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.set_title('Price vs Year', pad=12)
# sns.boxplot(x=year_collected,
#             y='price',
#             data=vehicles_year.toPandas(),
#             ax=ax)
# plt.xticks(rotation=90);


vehicles_year.show()

print('\nWith this used 20 year set, next, trying to find how the three features come together and depict real-worl characteristics. Checking how price varies with mean odometer ratings over the age of the car posted.\n')


# Convert posting_date to a date type
vehicles_year = vehicles_year.withColumn('posting_date', to_date('posting_date'))

# Add a new column for age of cars
vehicles_year = vehicles_year.withColumn('age', year('posting_date') - vehicles_year['year'])

# Preview the changes
vehicles_year.show()

# Get mean of odometer readings by age
grp_df = vehicles_year.groupBy('age').agg(avg('price').alias('avg_price'), avg('odometer').alias('avg_odometer')).orderBy('age')

# # Visualize how odometer average readings vary with price over age of cars
# # Set axes and points 
# x = [row.avg_odometer for row in grp_df.collect()]
# y = [row.avg_price for row in grp_df.collect()]
# points = [row.age for row in grp_df.collect()]
# s = [30*n for n in range(len(y))]

# f, ax = plt.subplots(figsize=(12, 8))
# # Plot for each year
# plt.title(f"Mean of Odometer vs Price over the cars age")
# plt.xlabel("Odometer Readings (mean)")
# plt.ylabel("Avg Price ($)")
# ax.grid(False)

# # Add labels for weeks
# for i, week in enumerate(points):
#     plt.annotate(week, (x[i], y[i]), size=14, va="bottom", ha="center")
#     plt.scatter(x, y, s=s)

# plt.show()


print("""
	\n It's evident that cars that have been driven less are more expensive than older cars which have been driven more. 
	There seem to be a good chunk of cars under 10k that have been driven 120k and over and are 12 years and older - this is an interesting insight.\n""")


# Group by condition and title_status columns and aggregate the mean of price
grouped_df = vehicles_year.groupBy('condition', 'title_status').agg({'price': 'mean'})

# # Convert Spark DataFrame to Pandas DataFrame for plotting
# pandas_df = grouped_df.toPandas()

# # Create plot
# sns.catplot(x='condition', y='avg(price)', hue='title_status', data=pandas_df,
#             kind='bar', aspect=2, height=5)

# # Set plot labels and title
# plt.xlabel('Condition')
# plt.ylabel('Price')
# plt.title('Average Price by Condition and Title Status')

# # Show plot
# plt.show()



print("""
	Since we want to look at only used cars, ignoring new cars for the moment.\n

It also looks like there are only parts being sold - which might affect the price.\n

Removing both these attributes..\n""")


# Filter the Spark DataFrame to include only used cars
vehicles_used = vehicles_year.filter(col('condition') != 'new')

# Filter the Spark DataFrame to exclude cars with title_status 'parts only'
vehicles_used = vehicles_used.filter(col('title_status') != 'parts only')

# Group by condition and title_status columns and aggregate the mean of price
grouped_df = vehicles_used.groupBy('condition', 'title_status').agg({'price': 'mean'})

# # Convert Spark DataFrame to Pandas DataFrame for plotting
# pandas_df = grouped_df.toPandas()

# # Create plot
# sns.catplot(x='condition', y='avg(price)', hue='title_status', data=pandas_df,
#             kind='bar', aspect=2, height=5)

# # Set plot labels and title
# plt.xlabel('Condition')
# plt.ylabel('Price')
# plt.title('Average Price of Used Cars by Condition and Title Status')

# # Show plot
# plt.show()



print('On to the next, understanding how price of cars is affected by the fuel and trasmission features...\n')


# Filter the Spark DataFrame to include only used cars
vehicles_used = vehicles_year.filter(col('condition') != 'new')

vehicles_used.show()

# Filter the Spark DataFrame to exclude cars with title_status 'parts only'
vehicles_used = vehicles_used.filter(col('title_status') != 'parts only')

vehicles_used.show()
# # Create plot
# g = sns.catplot(x='type', y='price', hue='fuel', col='transmission', data=vehicles_used.toPandas(),
#                 kind='bar', aspect=3, height=4, palette='rocket', col_wrap=1)

# # Set plot labels and title
# g.set_axis_labels('Type', 'Price')
# g.set_titles('Transmission: {col_name}')
# plt.subplots_adjust(top=0.9)
# g.fig.suptitle('Price of Used Cars by Type, Fuel and Transmission')

# # Show plot
# plt.show()

print("""
	It's noted that "other" values for type of fuels and trasmissions contribute to a considerable volume of data.\n

These, which are not a lot of value might affect the overall accuracy - hence removing them..\n""")

# Filter the Spark DataFrame to include only used cars
vehicles_used = vehicles_year.filter(col('condition') != 'new')

# Filter the Spark DataFrame to exclude cars with title_status 'parts only'
vehicles_used = vehicles_used.filter(col('title_status') != 'parts only')

# Filter the Spark DataFrame to exclude fuel types 'other'
vehicles_used = vehicles_used.filter(col('fuel') != 'other')

# Filter the Spark DataFrame to exclude transmission types 'other'
vehicles_used = vehicles_used.filter(col('transmission') != 'other')

vehicles_used.show()

# # Create plot
# g = sns.catplot(x='type', y='price', hue='fuel', col='transmission', data=vehicles_used.toPandas(),
#                 kind='bar', aspect=3, height=4, palette='rocket', col_wrap=1)

# # Set plot labels and title
# g.set_axis_labels('Type', 'Price')
# g.set_titles('Transmission: {col_name}')
# plt.subplots_adjust(top=0.9)
# g.fig.suptitle('Price of Used Cars by Type, Fuel and Transmission')

# # Show plot
# plt.show()


print('Next, we see how price is related to different kinds of manufacturers and the models they produce.\n')

# Visualize the relationship of average price by manufacturer
grp_man_df = vehicles_used.groupBy('manufacturer').agg(avg('price').alias('avg_price')).orderBy('manufacturer')

grp_man_df.show()

x = [row.manufacturer for row in grp_man_df.collect()]
y = [row.avg_price for row in grp_man_df.collect()]
y_mean = [vehicles_used.select(avg('price')).collect()[0][0]]*len(grp_man_df.collect())

# f, ax = plt.subplots(figsize=(12, 8))
# ax.set_facecolor('white') # Define a cor de fundo do gráfico como branco
# ax.scatter(x, y, s=[p/100 for p in y])
# ax.plot(x, y_mean, label='Average price', linestyle='--')
# ax.grid(False)

# plt.title(f"Average Prices by Manufacturer")
# plt.ylabel("Price ($)")
# plt.xlabel("Manufacturer")
# plt.xticks(rotation=90)
# plt.legend()
# plt.show()


print("""
	It's obviously evident that luxury brands have a higer price, but except a couple outliers, the median price lies near most points\n

Finally, we explore the "model" feature which I imagine has the highest cardinality amongst all the features we've seen so far..\n""")


# Add a field for row numbers
vehicles_used = vehicles_used.withColumn("row_num", row_number().over(Window.orderBy(col("model"))))

# Get counts of models
model_df = vehicles_used.groupBy("model").count().withColumnRenamed("count", "count_model")

model_df.show()

# Get only 10 frequent models and how much the other account to
lar10_df = model_df.orderBy(col("count_model").desc()).limit(10).toPandas()
other_val_sum = model_df.selectExpr("sum(count_model) as other_count").collect()[0].other_count - lar10_df["count_model"].sum()
lar10_df.loc[10] = ['Other Models', other_val_sum]

# # Plot what the counts of models look like
# import matplotlib.pyplot as plt
# import seaborn as sns

# f, ax = plt.subplots(figsize=(12, 8))
# ax.set_title('Count Distribution of Car Models', pad=12)
# sns.barplot(x="model", y="count_model",  palette="icefire",  data=lar10_df)
# for i, row in lar10_df.iterrows():
#     ax.text(i, row["count_model"] + 2, row["count_model"], ha="center")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()










