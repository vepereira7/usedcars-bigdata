# IMPORTS #
from io import BytesIO, StringIO
from google.cloud import storage

import time
import psutil
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf 
from pyspark.sql import types
from pyspark.sql import functions as F
from pyspark.ml.feature import Imputer, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.sql.types import StringType, IntegerType, StructType, StructField, DoubleType
from pyspark.sql.functions import when, upper, avg, year, to_date, sqrt, log, lower, col, row_number, asc, lit, count, expr, percentile_approx, monotonically_increasing_id, udf, skewness, regexp_extract
from pyspark.sql.window import Window
from pyspark.sql.functions import round
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, IsotonicRegression, FMRegressor, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor, GeneralizedLinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

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
#######


print('-----------------------------------------------')
print('----------- 2 cores & 18 instances ------------')
print('-----------------------------------------------')
print('')
print('')


print('')
print('')
print('-----------------------------------------------')
print('----------- Sample size: 100 MB ---------------')
print('-----------------------------------------------')
print('')
print('')

# Create spark session and define cores & instances
spark = SparkSession.builder.config('spark.driver.memory', '1g').config('spark.executor.memory', '4g') \
.config('spark.executor.instances', '18').config('spark.executor.cores','2').config('spark.driver.maxResultSize', '1g') \
.master('yarn').appName('egd').getOrCreate()


# Dataset loading
# client = storage.Client()
# bucket = client.get_bucket('dataproc-temp-us-central1-1012015907918-5s0m0iet')
# # Then do other things...
# #blob = bucket.get_blob('notebooks/jupyter/vehicles.csv')
# #bucket_path=f"gs://egd-project-bucket-2/notebooks/jupyter/notebooks_jupyter_notebooks_jupyter_vehicles.parquet"
# bucket_path = 'gs://egd-project-bucket-2/notebooks/jupyter/notebooks_jupyter_notebooks_jupyter_vehicles.csv'
# df=spark.read.csv(bucket_path, header=True)
# df.show()

file_path = 'gs://egd-bucket/perfomance/notebooks_jupyter_notebooks_jupyter_vehicles.csv'
df = (spark.read.format('csv').option('header', 'true').option('inferSchema','true').load(file_path))
df.show()

performance_df = []
performance_df_queries = []
performance_df_model = []

start_time = time.time()
# Set the desired sample size in megabytes
sample_size_mb = 100

# Calculate the fraction of the total dataset that corresponds to the desired sample size
total_size_mb = df.rdd.map(lambda row: len(str(row))).sum() / (1024*1024)  # Size of full dataset in MB
fraction = sample_size_mb / total_size_mb

# Load a sample of the desired size
df = df.sample(withReplacement=False, fraction=fraction, seed=42)
df.show(10)

end_time = time.time()
elapsed_time = end_time-start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Sample loading',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_queries.append(performance_dict)

print('')
print('')
print('-----------------------------------------------')
print('------------- Queries Testing -----------------')
print('-----------------------------------------------')
print('')
print('')

print('-----------------------------------------------')
print('----------------- Query 1 ---------------------')
print('-----------------------------------------------')
# Query 1 - What's the average price and total number of listings for each car manufacturer?
print("\nWhat's the average price and total number of listings for each car manufacturer?\n")
start_time = time.time()
manufacturer_counts = (
    df.groupBy('manufacturer')
    .agg({'manufacturer': 'count', 'price': 'mean'})
    .withColumnRenamed('count(manufacturer)', 'num_listings')
    .withColumnRenamed('median(price)', 'avg_price')
   .orderBy('num_listings', ascending=False)
)

manufacturer_counts.show()

end_time = time.time()
elapsed_time = end_time-start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Query 1',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_queries.append(performance_dict)

print('-----------------------------------------------')
print('----------------- Query 2 ---------------------')
print('-----------------------------------------------')
# Query 2 - What is the average price and sum of listings per state?
print("\nWhat is the average price and sum of listings per state?\n")
start_time = time.time()
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

end_time = time.time()
elapsed_time = end_time-start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Query 2',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_queries.append(performance_dict)

print('-----------------------------------------------')
print('----------------- Query 3 ---------------------')
print('-----------------------------------------------')
# Query 3 - What percentage of postings for each state is salvaged cars?
print("\nWhat percentage of postings for each state is salvaged cars?\n")

start_time = time.time()
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

end_time = time.time()
elapsed_time = end_time-start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Query 3',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_queries.append(performance_dict)

print('-----------------------------------------------')
print('----------------- Results ---------------------')
print('-----------------------------------------------')

queries_performance = spark.createDataFrame(performance_df_queries)
#queries_performance.show()
# queries_pandas = queries_performance.toPandas()
# queries_pandas.to_csv('2_6_100_queries.csv')

queries_performance = queries_performance.select('Cores & Instances', 'Query/Model', 'Dataset size', 'Execution Time', 'CPU Usage')
queries_performance.show()

print('')
print('')
print('-----------------------------------------------')
print('-------------- Models Testing -----------------')
print('-----------------------------------------------')
print('')
print('')

# CLEANING SIMPLIFIED # 

# Determine and remove the columns to drop based on the above graph
cols_to_drop = ['id','url', 'region', 'region_url', 'VIN', 'image_url', 'description', \
    'county', 'size', 'paint_color', 'drive', 'cylinders', 'state', 'lat','long']
vehicles_df = df.select([col(c) for c in df.columns if c not in cols_to_drop])

# Remove null values and duplicated rows
vehicles_df = vehicles_df.dropna().dropDuplicates()

# Drop NaNs and duplicates
vehicles_df = vehicles_df.dropna().dropDuplicates()

# Add index column using monotonically_increasing_id() function
vehicles_df = vehicles_df.withColumn("index", monotonically_increasing_id())

# Change data type of year to string
vehicles_df = vehicles_df.withColumn("year", col("year").cast("string"))

# Reorder columns with index first
vehicles_df = vehicles_df.select("index", *vehicles_df.columns[:-1])

# Describing the dataset to get a basic idea of the non-categorical features
vehicles_df.select([col(c).cast("float") for c in vehicles_df.columns if c not in ['manufacturer', 'model', 'condition', 'fuel', 'title_status', 'transmission', 'type']])

# Create a new Spark DataFrame with the filtered data
vehicles_prc = vehicles_df.filter((vehicles_df.price >= 2000) & (vehicles_df.price <= 50000))

vehicles_odo = vehicles_prc.filter((col("odometer") > 100) & (col("odometer") <= 200000))

year_list = list(range(2000, 2021))
vehicles_year = vehicles_odo.filter(col('year').cast(IntegerType()).isin(year_list))

# Convert posting_date to a date type
vehicles_year = vehicles_year.withColumn('posting_date', to_date('posting_date'))

# Add a new column for age of cars
vehicles_year = vehicles_year.withColumn('age', year('posting_date') - vehicles_year['year'])

# Group by condition and title_status columns and aggregate the mean of price
grouped_df = vehicles_year.groupBy('condition', 'title_status').agg({'price': 'mean'})

# Filter the Spark DataFrame to include only used cars
vehicles_used = vehicles_year.filter(col('condition') != 'new')

# Filter the Spark DataFrame to exclude cars with title_status 'parts only'
vehicles_used = vehicles_used.filter(col('title_status') != 'parts only')

# Group by condition and title_status columns and aggregate the mean of price
grouped_df = vehicles_used.groupBy('condition', 'title_status').agg({'price': 'mean'})

# Filter the Spark DataFrame to include only used cars
vehicles_used = vehicles_year.filter(col('condition') != 'new')

# Filter the Spark DataFrame to exclude cars with title_status 'parts only'
vehicles_used = vehicles_used.filter(col('title_status') != 'parts only')

# Filter the Spark DataFrame to include only used cars
vehicles_used = vehicles_year.filter(col('condition') != 'new')

# Filter the Spark DataFrame to exclude cars with title_status 'parts only'
vehicles_used = vehicles_used.filter(col('title_status') != 'parts only')

# Filter the Spark DataFrame to exclude fuel types 'other'
vehicles_used = vehicles_used.filter(col('fuel') != 'other')

# Filter the Spark DataFrame to exclude transmission types 'other'
vehicles_used = vehicles_used.filter(col('transmission') != 'other')

# Add a field for row numbers
vehicles_used = vehicles_used.withColumn("row_num", row_number().over(Window.orderBy(col("model"))))

# MODELS #

# Get current information of the dataset
vehicles_used.printSchema()

# Drop columns populated during clean-up or not required
vehicles_used = vehicles_used.drop('posting_date', 'row_num')


# Make a copy of the data frame for encoding
vehicles_used_enc = vehicles_used

# Print schema of the encoded DataFrame
vehicles_used_enc.printSchema()
vehicles_used_enc2 = vehicles_used_enc


vehicles_used_enc2.show()

# Cast the string column to double
vehicles_used_enc = vehicles_used_enc.withColumn("price", col("price").cast("long"))

vehicles_used_enc = vehicles_used_enc.withColumn("odometer", col("odometer").cast("double"))


# Get fields that are categorical and remove only "model"
cat_features = [c for c, dtype in vehicles_used_enc.dtypes if dtype == 'string']
print(f'Categorical features: {cat_features}\n\n')

# Encode using StringIndexer
for c in cat_features:
    indexer = StringIndexer(inputCol=c, outputCol=c+"_indexed")
    model = indexer.fit(vehicles_used_enc)
    vehicles_used_enc = model.transform(vehicles_used_enc).drop(c)
    vehicles_used_enc = vehicles_used_enc.withColumnRenamed(c+"_indexed", c)


# drop row number column
vehicles_used_enc = vehicles_used_enc.drop("index")
vehicles_used_enc.show()

vehicles_used_enc.printSchema()



features = VectorAssembler(inputCols = [
 'odometer',
 'age',
 'year',
 'manufacturer',
 'model',
 'condition',
 'fuel',
 'title_status',
 'transmission',
 'type'],outputCol='features', handleInvalid = 'skip')

training_features = features.transform(vehicles_used_enc)
training_features = training_features.select('price','features')
print('\n--- ML dataset ---\n')
training_features.show(5)
#split ML dataset in train (.8) and test (.2)
train_data, test_data = training_features.randomSplit([0.8,0.2])
print('\n--- Train Data ---\n')
train_data.show(5)
print('\n--- Test Data ---\n')
test_data.show(5)
print('\n--- Data Used for the Prediction---\n')
# amostrar aleatoriamente 5% dos dados para teste
test_data_sample = test_data.sample(fraction=0.05, seed=123)

# exibir as primeiras linhas do DataFrame
test_data_sample.show(5)


# FUNCTION TO TRAIN ALL MODELS
def reg_metrics(model, train_data, test_data, algo):
    """ Function takes in training and testing sets, prediction model, 
    and ouputs the below metrics:
    1. R² or Coefficient of Determination.
    2. Adjusted R²
    3. Mean Squared Error(MSE)
    4. Root-Mean-Squared-Error(RMSE).
    5. Mean-Absolute-Error(MAE).
    """
    # Get predicted values on test_data
    test_pred = model.transform(test_data)
       
    #1 & 2 Coefficient of Determination (R² & Adjusted R²)
    print("\n\t--- Coefficient of Determination (R² & Adjusted R²) ---")
    evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="r2")
    r2 = evaluator.evaluate(test_pred)
    adj_r2 = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="r2adj")
    adj_r2 = evaluator.evaluate(test_pred)

    print(f"R²\t\t: {r2:.5f}")
    print(f"Adjusted R²\t: {adj_r2:.5f}")

    #3 & 4. MSE and RMSE
    print("\n\t--- Mean Squared Error (MSE & RMSE) ---")
    mse_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="mse")
    mse = mse_evaluator.evaluate(test_pred)
    rmse_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
    rmse = rmse_evaluator.evaluate(test_pred)
    
    print(f"MSE\t: {mse:.4f}")
    print(f"RMSE\t: {rmse:.2f}")

    #5. MAE
    print("\n\t--- Mean Absolute Error (MAE) ---")
    mae_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="mae")
    mae = mae_evaluator.evaluate(test_pred)
    print(f"MAE\t: {mae:.2f}")
    
    # Return metrics as a dictionary
    metrics_dict = {
        'Algorithm': algo,
        'R²': r2,
        'Adjusted R²': adj_r2,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae
    }
    #'Adjusted R²': adj_r2_formatted,
    return metrics_dict


print('-----------------------------------------------')
print('-------------- Linear Regression --------------')
print('-----------------------------------------------')

start_time = time.time()
lr = LinearRegression(
    featuresCol='features',
    labelCol='price',
    maxIter=100,  # increase maxIter
    regParam=0.1,  # try different values for regParam
    elasticNetParam=0.7  # try different values for elasticNetParam
)
lr_model = lr.fit(train_data)
metrics_dict_lr = reg_metrics(lr_model, train_data, test_data, 'Linear Regression')
print("\n\t--- Predictions ---")
pred_results = lr_model.evaluate(test_data_sample)
pred_results.predictions.show(5)

end_time = time.time()
elapsed_time = end_time - start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Liner Regression',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_model.append(performance_dict)

print('-----------------------------------------------')
print('---------------- Decision Tree ----------------')
print('-----------------------------------------------')


# DECISION TREE #
start_time = time.time()
dt = DecisionTreeRegressor(featuresCol = 'features', labelCol = 'price', maxDepth=5, maxBins=40000)
dt_model = dt.fit(train_data)
metrics_dict_dt = reg_metrics(dt_model, train_data, test_data, 'Decision Tree')
print("\n\t--- Predictions ---")
pred_results = dt_model.transform(test_data_sample)
pred_results.show(5)

end_time = time.time()
elapsed_time = end_time - start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Decison Tree',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_model.append(performance_dict)

print('-----------------------------------------------')
print('--------- Random Forest Regression ------------')
print('-----------------------------------------------')


# RANDOM FOREST REGRESSION #
start_time = time.time()
rf = RandomForestRegressor(featuresCol = 'features', labelCol = 'price', numTrees=2, maxDepth=2, maxBins=40000)
rf_model = rf.fit(train_data)
metrics_dict_rf = reg_metrics(rf_model, train_data, test_data, 'Random Forest Regression')
print("\n\t--- Predictions ---")
pred_results = rf_model.transform(test_data_sample)
pred_results.show(5)

end_time = time.time()
elapsed_time = end_time - start_time
# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Random Forest Regression',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_model.append(performance_dict)

print('-----------------------------------------------')
print('----------------- Results ---------------------')
print('-----------------------------------------------')

models_performance = spark.createDataFrame(performance_df_model)
models_performance.show()
# queries_pandas = queries_performance.toPandas()
# queries_pandas.to_csv('2_6_100_queries.csv'
models_performance = models_performance.select('Cores & Instances', 'Query/Model', 'Dataset size', 'Execution Time', 'CPU Usage')
models_performance.show()

performance_df_queries = []
performance_df_model = []



print('')
print('')
print('-----------------------------------------------')
print('----------- Sample size: 700 MB ---------------')
print('-----------------------------------------------')
print('')
print('')

# Dataset loading
# client = storage.Client()
# bucket = client.get_bucket('dataproc-temp-us-central1-1012015907918-5s0m0iet')
# # Then do other things...
# #blob = bucket.get_blob('notebooks/jupyter/vehicles.csv')
# #bucket_path=f"gs://egd-project-bucket-2/notebooks/jupyter/notebooks_jupyter_notebooks_jupyter_vehicles.parquet"
# bucket_path = 'gs://egd-project-bucket-2/notebooks/jupyter/notebooks_jupyter_notebooks_jupyter_vehicles.csv'
# df=spark.read.csv(bucket_path, header=True)
# df.show()

df = (spark.read.format('csv').option('header', 'true').option('inferSchema','true').load(file_path))
df.show()

start_time = time.time()
# Set the desired sample size in megabytes
sample_size_mb = 700

# Calculate the fraction of the total dataset that corresponds to the desired sample size
total_size_mb = df.rdd.map(lambda row: len(str(row))).sum() / (1024*1024)  # Size of full dataset in MB
fraction = sample_size_mb / total_size_mb

# Load a sample of the desired size
df = df.sample(withReplacement=False, fraction=fraction, seed=42)
df.show(10)

end_time = time.time()
elapsed_time = end_time-start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Sample loading',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_queries.append(performance_dict)

print('')
print('')
print('-----------------------------------------------')
print('------------- Queries Testing -----------------')
print('-----------------------------------------------')
print('')
print('')

print('-----------------------------------------------')
print('----------------- Query 1 ---------------------')
print('-----------------------------------------------')
# Query 1 - What's the average price and total number of listings for each car manufacturer?
print("\nWhat's the average price and total number of listings for each car manufacturer?\n")

start_time = time.time()
manufacturer_counts = (
    df.groupBy('manufacturer')
    .agg({'manufacturer': 'count', 'price': 'mean'})
    .withColumnRenamed('count(manufacturer)', 'num_listings')
    .withColumnRenamed('median(price)', 'avg_price')
   .orderBy('num_listings', ascending=False)
)

manufacturer_counts.show()

end_time = time.time()
elapsed_time = end_time-start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Query 1',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_queries.append(performance_dict)

print('-----------------------------------------------')
print('----------------- Query 2 ---------------------')
print('-----------------------------------------------')
# Query 2 - What is the average price and sum of listings per state?
print("\nWhat is the average price and sum of listings per state?\n")

start_time = time.time()
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

end_time = time.time()
elapsed_time = end_time-start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Query 2',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_queries.append(performance_dict)

print('-----------------------------------------------')
print('----------------- Query 3 ---------------------')
print('-----------------------------------------------')
# Query 3 - What percentage of postings for each state is salvaged cars?
print("\nWhat percentage of postings for each state is salvaged cars?\n")

start_time = time.time()
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

end_time = time.time()
elapsed_time = end_time-start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Query 3',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_queries.append(performance_dict)

print('-----------------------------------------------')
print('----------------- Results ---------------------')
print('-----------------------------------------------')

queries_performance = spark.createDataFrame(performance_df_queries)
queries_performance = queries_performance.select('Cores & Instances', 'Query/Model', 'Dataset size', 'Execution Time', 'CPU Usage')
queries_performance.show()
# queries_pandas = queries_performance.toPandas()
# queries_pandas.to_csv('2_6_100_queries.csv')


print('')
print('')
print('-----------------------------------------------')
print('-------------- Models Testing -----------------')
print('-----------------------------------------------')
print('')
print('')

# CLEANING SIMPLIFIED # 

# Determine and remove the columns to drop based on the above graph
cols_to_drop = ['id','url', 'region', 'region_url', 'VIN', 'image_url', 'description', \
    'county', 'size', 'paint_color', 'drive', 'cylinders', 'state', 'lat','long']
vehicles_df = df.select([col(c) for c in df.columns if c not in cols_to_drop])

# Remove null values and duplicated rows
vehicles_df = vehicles_df.dropna().dropDuplicates()

# Drop NaNs and duplicates
vehicles_df = vehicles_df.dropna().dropDuplicates()

# Add index column using monotonically_increasing_id() function
vehicles_df = vehicles_df.withColumn("index", monotonically_increasing_id())

# Change data type of year to string
vehicles_df = vehicles_df.withColumn("year", col("year").cast("string"))

# Reorder columns with index first
vehicles_df = vehicles_df.select("index", *vehicles_df.columns[:-1])

# Describing the dataset to get a basic idea of the non-categorical features
vehicles_df.select([col(c).cast("float") for c in vehicles_df.columns if c not in ['manufacturer', 'model', 'condition', 'fuel', 'title_status', 'transmission', 'type']])

# Create a new Spark DataFrame with the filtered data
vehicles_prc = vehicles_df.filter((vehicles_df.price >= 2000) & (vehicles_df.price <= 50000))

vehicles_odo = vehicles_prc.filter((col("odometer") > 100) & (col("odometer") <= 200000))

year_list = list(range(2000, 2021))
vehicles_year = vehicles_odo.filter(col('year').cast(IntegerType()).isin(year_list))

# Convert posting_date to a date type
vehicles_year = vehicles_year.withColumn('posting_date', to_date('posting_date'))

# Add a new column for age of cars
vehicles_year = vehicles_year.withColumn('age', year('posting_date') - vehicles_year['year'])

# Group by condition and title_status columns and aggregate the mean of price
grouped_df = vehicles_year.groupBy('condition', 'title_status').agg({'price': 'mean'})

# Filter the Spark DataFrame to include only used cars
vehicles_used = vehicles_year.filter(col('condition') != 'new')

# Filter the Spark DataFrame to exclude cars with title_status 'parts only'
vehicles_used = vehicles_used.filter(col('title_status') != 'parts only')

# Group by condition and title_status columns and aggregate the mean of price
grouped_df = vehicles_used.groupBy('condition', 'title_status').agg({'price': 'mean'})

# Filter the Spark DataFrame to include only used cars
vehicles_used = vehicles_year.filter(col('condition') != 'new')

# Filter the Spark DataFrame to exclude cars with title_status 'parts only'
vehicles_used = vehicles_used.filter(col('title_status') != 'parts only')

# Filter the Spark DataFrame to include only used cars
vehicles_used = vehicles_year.filter(col('condition') != 'new')

# Filter the Spark DataFrame to exclude cars with title_status 'parts only'
vehicles_used = vehicles_used.filter(col('title_status') != 'parts only')

# Filter the Spark DataFrame to exclude fuel types 'other'
vehicles_used = vehicles_used.filter(col('fuel') != 'other')

# Filter the Spark DataFrame to exclude transmission types 'other'
vehicles_used = vehicles_used.filter(col('transmission') != 'other')

# Add a field for row numbers
vehicles_used = vehicles_used.withColumn("row_num", row_number().over(Window.orderBy(col("model"))))

# MODELS #

# Get current information of the dataset
vehicles_used.printSchema()

# Drop columns populated during clean-up or not required
vehicles_used = vehicles_used.drop('posting_date', 'row_num')


# Make a copy of the data frame for encoding
vehicles_used_enc = vehicles_used

# Print schema of the encoded DataFrame
vehicles_used_enc.printSchema()
vehicles_used_enc2 = vehicles_used_enc


vehicles_used_enc2.show()

# Cast the string column to double
vehicles_used_enc = vehicles_used_enc.withColumn("price", col("price").cast("long"))

vehicles_used_enc = vehicles_used_enc.withColumn("odometer", col("odometer").cast("double"))


# Get fields that are categorical and remove only "model"
cat_features = [c for c, dtype in vehicles_used_enc.dtypes if dtype == 'string']
print(f'Categorical features: {cat_features}\n\n')

# Encode using StringIndexer
for c in cat_features:
    indexer = StringIndexer(inputCol=c, outputCol=c+"_indexed")
    model = indexer.fit(vehicles_used_enc)
    vehicles_used_enc = model.transform(vehicles_used_enc).drop(c)
    vehicles_used_enc = vehicles_used_enc.withColumnRenamed(c+"_indexed", c)


# drop row number column
vehicles_used_enc = vehicles_used_enc.drop("index")
vehicles_used_enc.show()

vehicles_used_enc.printSchema()



features = VectorAssembler(inputCols = [
 'odometer',
 'age',
 'year',
 'manufacturer',
 'model',
 'condition',
 'fuel',
 'title_status',
 'transmission',
 'type'],outputCol='features', handleInvalid = 'skip')

training_features = features.transform(vehicles_used_enc)
training_features = training_features.select('price','features')
print('\n--- ML dataset ---\n')
training_features.show(5)
#split ML dataset in train (.8) and test (.2)
train_data, test_data = training_features.randomSplit([0.8,0.2])
print('\n--- Train Data ---\n')
train_data.show(5)
print('\n--- Test Data ---\n')
test_data.show(5)
print('\n--- Data Used for the Prediction---\n')
# amostrar aleatoriamente 5% dos dados para teste
test_data_sample = test_data.sample(fraction=0.05, seed=123)

# exibir as primeiras linhas do DataFrame
test_data_sample.show(5)


print('-----------------------------------------------')
print('-------------- Linear Regression --------------')
print('-----------------------------------------------')

start_time = time.time()
lr = LinearRegression(
    featuresCol='features',
    labelCol='price',
    maxIter=100,  # increase maxIter
    regParam=0.1,  # try different values for regParam
    elasticNetParam=0.7  # try different values for elasticNetParam
)
lr_model = lr.fit(train_data)
metrics_dict_lr = reg_metrics(lr_model, train_data, test_data, 'Linear Regression')
print("\n\t--- Predictions ---")
pred_results = lr_model.evaluate(test_data_sample)
pred_results.predictions.show(5)

end_time = time.time()
elapsed_time = end_time - start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Liner Regression',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_model.append(performance_dict)

print('-----------------------------------------------')
print('---------------- Decision Tree ----------------')
print('-----------------------------------------------')


# DECISION TREE #
start_time = time.time()
dt = DecisionTreeRegressor(featuresCol = 'features', labelCol = 'price', maxDepth=5, maxBins=40000)
dt_model = dt.fit(train_data)
metrics_dict_dt = reg_metrics(dt_model, train_data, test_data, 'Decision Tree')
print("\n\t--- Predictions ---")
pred_results = dt_model.transform(test_data_sample)
pred_results.show(5)

end_time = time.time()
elapsed_time = end_time - start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Decision Tree',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_model.append(performance_dict)

print('-----------------------------------------------')
print('--------- Random Forest Regression ------------')
print('-----------------------------------------------')


# RANDOM FOREST REGRESSION #
start_time = time.time()
rf = RandomForestRegressor(featuresCol = 'features', labelCol = 'price', numTrees=2, maxDepth=2, maxBins=40000)
rf_model = rf.fit(train_data)
metrics_dict_rf = reg_metrics(rf_model, train_data, test_data, 'Random Forest Regression')
print("\n\t--- Predictions ---")
pred_results = rf_model.transform(test_data_sample)
pred_results.show(5)

end_time = time.time()
elapsed_time = end_time - start_time
# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Random Forest Regression',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_model.append(performance_dict)

print('-----------------------------------------------')
print('----------------- Results ---------------------')
print('-----------------------------------------------')

models_performance = spark.createDataFrame(performance_df_model)
models_performance = models_performance.select('Cores & Instances', 'Query/Model', 'Dataset size', 'Execution Time', 'CPU Usage')
models_performance.show()
# queries_pandas = queries_performance.toPandas()
# queries_pandas.to_csv('2_6_100_queries.csv'


performance_df_queries = []
performance_df_model = []



print('')
print('')
print('-----------------------------------------------')
print('----------- Sample size: 1.3 GB ---------------')
print('-----------------------------------------------')
print('')
print('')

# Dataset loading
# client = storage.Client()
# bucket = client.get_bucket('dataproc-temp-us-central1-1012015907918-5s0m0iet')
# # Then do other things...
# #blob = bucket.get_blob('notebooks/jupyter/vehicles.csv')
# #bucket_path=f"gs://egd-project-bucket-2/notebooks/jupyter/notebooks_jupyter_notebooks_jupyter_vehicles.parquet"
# start_time = time.time()
# bucket_path = 'gs://egd-project-bucket-2/notebooks/jupyter/notebooks_jupyter_notebooks_jupyter_vehicles.csv'
# df=spark.read.csv(bucket_path, header=True)

df = (spark.read.format('csv').option('header', 'true').option('inferSchema','true').load(file_path))
df.show()

sample_size_mb = 1300


end_time = time.time()
elapsed_time = end_time-start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Sample loading',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_queries.append(performance_dict)

print('')
print('')
print('-----------------------------------------------')
print('------------- Queries Testing -----------------')
print('-----------------------------------------------')
print('')
print('')

print('-----------------------------------------------')
print('----------------- Query 1 ---------------------')
print('-----------------------------------------------')
# Query 1 - What's the average price and total number of listings for each car manufacturer?

print("\nWhat's the average price and total number of listings for each car manufacturer?\n")
start_time = time.time()
manufacturer_counts = (
    df.groupBy('manufacturer')
    .agg({'manufacturer': 'count', 'price': 'mean'})
    .withColumnRenamed('count(manufacturer)', 'num_listings')
    .withColumnRenamed('median(price)', 'avg_price')
   .orderBy('num_listings', ascending=False)
)

manufacturer_counts.show()

end_time = time.time()
elapsed_time = end_time-start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Query 1',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_queries.append(performance_dict)

print('-----------------------------------------------')
print('----------------- Query 2 ---------------------')
print('-----------------------------------------------')
# Query 2 - What is the average price and sum of listings per state?
print("\nWhat is the average price and sum of listings per state?\n")
start_time = time.time()
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

end_time = time.time()
elapsed_time = end_time-start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Query 2',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_queries.append(performance_dict)

print('-----------------------------------------------')
print('----------------- Query 3 ---------------------')
print('-----------------------------------------------')
# Query 2 - What percentage of postings for each state is salvaged cars?
print("\nWhat percentage of postings for each state is salvaged cars?\n")

start_time = time.time()
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

end_time = time.time()
elapsed_time = end_time-start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Query 3',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_queries.append(performance_dict)

print('-----------------------------------------------')
print('----------------- Results ---------------------')
print('-----------------------------------------------')

queries_performance = spark.createDataFrame(performance_df_queries)
queries_performance = queries_performance.select('Cores & Instances', 'Query/Model', 'Dataset size', 'Execution Time', 'CPU Usage')
queries_performance.show()
# queries_pandas = queries_performance.toPandas()
# queries_pandas.to_csv('2_6_100_queries.csv')


print('')
print('')
print('-----------------------------------------------')
print('-------------- Models Testing -----------------')
print('-----------------------------------------------')
print('')
print('')

# CLEANING SIMPLIFIED # 

# Determine and remove the columns to drop based on the above graph
cols_to_drop = ['id','url', 'region', 'region_url', 'VIN', 'image_url', 'description', \
    'county', 'size', 'paint_color', 'drive', 'cylinders', 'state', 'lat','long']
vehicles_df = df.select([col(c) for c in df.columns if c not in cols_to_drop])

# Remove null values and duplicated rows
vehicles_df = vehicles_df.dropna().dropDuplicates()

# Drop NaNs and duplicates
vehicles_df = vehicles_df.dropna().dropDuplicates()

# Add index column using monotonically_increasing_id() function
vehicles_df = vehicles_df.withColumn("index", monotonically_increasing_id())

# Change data type of year to string
vehicles_df = vehicles_df.withColumn("year", col("year").cast("string"))

# Reorder columns with index first
vehicles_df = vehicles_df.select("index", *vehicles_df.columns[:-1])

# Describing the dataset to get a basic idea of the non-categorical features
vehicles_df.select([col(c).cast("float") for c in vehicles_df.columns if c not in ['manufacturer', 'model', 'condition', 'fuel', 'title_status', 'transmission', 'type']])

# Create a new Spark DataFrame with the filtered data
vehicles_prc = vehicles_df.filter((vehicles_df.price >= 2000) & (vehicles_df.price <= 50000))

vehicles_odo = vehicles_prc.filter((col("odometer") > 100) & (col("odometer") <= 200000))

year_list = list(range(2000, 2021))
vehicles_year = vehicles_odo.filter(col('year').cast(IntegerType()).isin(year_list))

# Convert posting_date to a date type
vehicles_year = vehicles_year.withColumn('posting_date', to_date('posting_date'))

# Add a new column for age of cars
vehicles_year = vehicles_year.withColumn('age', year('posting_date') - vehicles_year['year'])

# Group by condition and title_status columns and aggregate the mean of price
grouped_df = vehicles_year.groupBy('condition', 'title_status').agg({'price': 'mean'})

# Filter the Spark DataFrame to include only used cars
vehicles_used = vehicles_year.filter(col('condition') != 'new')

# Filter the Spark DataFrame to exclude cars with title_status 'parts only'
vehicles_used = vehicles_used.filter(col('title_status') != 'parts only')

# Group by condition and title_status columns and aggregate the mean of price
grouped_df = vehicles_used.groupBy('condition', 'title_status').agg({'price': 'mean'})

# Filter the Spark DataFrame to include only used cars
vehicles_used = vehicles_year.filter(col('condition') != 'new')

# Filter the Spark DataFrame to exclude cars with title_status 'parts only'
vehicles_used = vehicles_used.filter(col('title_status') != 'parts only')

# Filter the Spark DataFrame to include only used cars
vehicles_used = vehicles_year.filter(col('condition') != 'new')

# Filter the Spark DataFrame to exclude cars with title_status 'parts only'
vehicles_used = vehicles_used.filter(col('title_status') != 'parts only')

# Filter the Spark DataFrame to exclude fuel types 'other'
vehicles_used = vehicles_used.filter(col('fuel') != 'other')

# Filter the Spark DataFrame to exclude transmission types 'other'
vehicles_used = vehicles_used.filter(col('transmission') != 'other')

# Add a field for row numbers
vehicles_used = vehicles_used.withColumn("row_num", row_number().over(Window.orderBy(col("model"))))

# MODELS #

# Get current information of the dataset
vehicles_used.printSchema()

# Drop columns populated during clean-up or not required
vehicles_used = vehicles_used.drop('posting_date', 'row_num')


# Make a copy of the data frame for encoding
vehicles_used_enc = vehicles_used

# Print schema of the encoded DataFrame
vehicles_used_enc.printSchema()
vehicles_used_enc2 = vehicles_used_enc


vehicles_used_enc2.show()

# Cast the string column to double
vehicles_used_enc = vehicles_used_enc.withColumn("price", col("price").cast("long"))

vehicles_used_enc = vehicles_used_enc.withColumn("odometer", col("odometer").cast("double"))


# Get fields that are categorical and remove only "model"
cat_features = [c for c, dtype in vehicles_used_enc.dtypes if dtype == 'string']
print(f'Categorical features: {cat_features}\n\n')

# Encode using StringIndexer
for c in cat_features:
    indexer = StringIndexer(inputCol=c, outputCol=c+"_indexed")
    model = indexer.fit(vehicles_used_enc)
    vehicles_used_enc = model.transform(vehicles_used_enc).drop(c)
    vehicles_used_enc = vehicles_used_enc.withColumnRenamed(c+"_indexed", c)


# drop row number column
vehicles_used_enc = vehicles_used_enc.drop("index")
vehicles_used_enc.show()

vehicles_used_enc.printSchema()



features = VectorAssembler(inputCols = [
 'odometer',
 'age',
 'year',
 'manufacturer',
 'model',
 'condition',
 'fuel',
 'title_status',
 'transmission',
 'type'],outputCol='features', handleInvalid = 'skip')

training_features = features.transform(vehicles_used_enc)
training_features = training_features.select('price','features')
print('\n--- ML dataset ---\n')
training_features.show(5)
#split ML dataset in train (.8) and test (.2)
train_data, test_data = training_features.randomSplit([0.8,0.2])
print('\n--- Train Data ---\n')
train_data.show(5)
print('\n--- Test Data ---\n')
test_data.show(5)
print('\n--- Data Used for the Prediction---\n')
# amostrar aleatoriamente 5% dos dados para teste
test_data_sample = test_data.sample(fraction=0.05, seed=123)

# exibir as primeiras linhas do DataFrame
test_data_sample.show(5)


print('-----------------------------------------------')
print('-------------- Linear Regression --------------')
print('-----------------------------------------------')

start_time = time.time()
lr = LinearRegression(
    featuresCol='features',
    labelCol='price',
    maxIter=100,  # increase maxIter
    regParam=0.1,  # try different values for regParam
    elasticNetParam=0.7  # try different values for elasticNetParam
)
lr_model = lr.fit(train_data)
metrics_dict_lr = reg_metrics(lr_model, train_data, test_data, 'Linear Regression')
print("\n\t--- Predictions ---")
pred_results = lr_model.evaluate(test_data_sample)
pred_results.predictions.show(5)

end_time = time.time()
elapsed_time = end_time - start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Liner Regression',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_model.append(performance_dict)

print('-----------------------------------------------')
print('---------------- Decision Tree ----------------')
print('-----------------------------------------------')


# DECISION TREE #
start_time = time.time()
dt = DecisionTreeRegressor(featuresCol = 'features', labelCol = 'price', maxDepth=5, maxBins=40000)
dt_model = dt.fit(train_data)
metrics_dict_dt = reg_metrics(dt_model, train_data, test_data, 'Decision Tree')
print("\n\t--- Predictions ---")
pred_results = dt_model.transform(test_data_sample)
pred_results.show(5)

end_time = time.time()
elapsed_time = end_time - start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Decision Tree',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_model.append(performance_dict)

print('-----------------------------------------------')
print('--------- Random Forest Regression ------------')
print('-----------------------------------------------')


# RANDOM FOREST REGRESSION #
start_time = time.time()
rf = RandomForestRegressor(featuresCol = 'features', labelCol = 'price', numTrees=2, maxDepth=2, maxBins=40000)
rf_model = rf.fit(train_data)
metrics_dict_rf = reg_metrics(rf_model, train_data, test_data, 'Random Forest Regression')
print("\n\t--- Predictions ---")
pred_results = rf_model.transform(test_data_sample)
pred_results.show(5)

end_time = time.time()
elapsed_time = end_time - start_time
# Measure CPU usage
cpu_usage = psutil.cpu_percent()

performance_dict = {
    'Cores & Instances': '2 cores & 18 instances',
    'Dataset size': sample_size_mb,
    'Query/Model': 'Random Forest Regression',
    'Execution Time': elapsed_time,
    'CPU Usage': cpu_usage
}

performance_df.append(performance_dict)
performance_df_model.append(performance_dict)

print('-----------------------------------------------')
print('----------------- Results ---------------------')
print('-----------------------------------------------')

models_performance = spark.createDataFrame(performance_df_model)
models_performance = models_performance.select('Cores & Instances', 'Query/Model', 'Dataset size', 'Execution Time', 'CPU Usage')
models_performance.show()

print('')
print('-----------------------------------------------')
print('--------------- All Results -------------------')
print('-----------------------------------------------')
print('')
final_df_spark = spark.createDataFrame(performance_df)
final_df_spark = final_df_spark.select('Cores & Instances', 'Query/Model', 'Dataset size', 'Execution Time', 'CPU Usage')
final_df_spark.show()

# Save to csv
output_path = "gs://egd-bucket/perfomance/csv"

# Write DataFrame to CSV file
final_df_spark.write.format("csv").option("header", "true").mode("overwrite").save(output_path)

