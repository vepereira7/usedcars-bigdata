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
spark = SparkSession.builder.config('...').master('yarn').appName('egd').getOrCreate()
# spark = SparkSession.builder.config('spark.driver.memory', '1g').config('spark.executor.memory', '4g') \
# .config('spark.executor.instances', '2').config(conf=conf).config('spark.executor.cores','2').config('spark.driver.maxResultSize', '1g') \
# .master('yarn').appName('egd').getOrCreate()

file_path = 'gs://egd-project-vp-1/egd-project/notebooks_jupyter_notebooks_jupyter_vehicles.csv'
df = (spark.read.format('csv').option('header', 'true').option('inferSchema','true').load(file_path))

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


###########


print('-----------------------------------------')
print('----------- Price Prediction ------------')
print('-----------------------------------------')
print('')






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


from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

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

model_metrics = []
model_metrics_performance = []

# LINEAR REGRESSION #
# Measure execution time
start_time = time.time()
print("\t------- Linear Regression -------")
lr = LinearRegression(
    featuresCol='features',
    labelCol='price',
    maxIter=100,  # increase maxIter
    regParam=0.1,  # try different values for regParam
    elasticNetParam=0.7  # try different values for elasticNetParam
)
lr_model = lr.fit(train_data)
metrics_dict_lr = reg_metrics(lr_model, train_data, test_data, 'Linear Regression')
model_metrics.append(metrics_dict_lr)
print("\n\t--- Predictions ---")
pred_results = lr_model.evaluate(test_data_sample)
pred_results.predictions.show(5)

end_time = time.time()
elapsed_time = end_time - start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

metrics_dict_time = {
        'Algorithm':'Linear Regression',
        'Execution Time': elapsed_time,
        'CPU usage': cpu_usage
    }
model_metrics_performance.append(metrics_dict_time)


# LASSO REGRESSION #
start_time = time.time()
print("\t------- Lasso Regression -------")
lasso = LinearRegression(featuresCol='features', labelCol='price', maxIter=100, regParam=0.1, elasticNetParam=1)
lasso_model = lasso.fit(train_data)
metrics_dict_lasso = reg_metrics(lasso_model, train_data, test_data, 'Lasso Regression')
model_metrics.append(metrics_dict_lasso)
print("\n\t--- Predictions ---")
pred_results = lasso_model.transform(test_data_sample)
pred_results.show(5)

end_time = time.time()
elapsed_time = end_time - start_time
# Measure CPU usage
cpu_usage = psutil.cpu_percent()

metrics_dict_time = {
        'Algorithm':'Lasso Regression',
        'Execution Time': elapsed_time,
        'CPU usage': cpu_usage
    }
model_metrics_performance.append(metrics_dict_time)


# LASSO REGRESSION #
start_time = time.time()
print("\t------- Ridge Regression -------")
ridge = LinearRegression(featuresCol = 'features', labelCol = 'price',maxIter=100, regParam=0.3, elasticNetParam=0.1)
ridge_model = ridge.fit(train_data)
metrics_dict_ridge = reg_metrics(ridge_model, train_data, test_data, 'Ridge Regression')
model_metrics.append(metrics_dict_ridge)
print("\n\t--- Predictions ---")
pred_results = ridge_model.evaluate(test_data_sample)
pred_results.predictions.show(5)

end_time = time.time()
elapsed_time = end_time - start_time
# Measure CPU usage
cpu_usage = psutil.cpu_percent()


metrics_dict_time = {
        'Algorithm':'Ridge Regression',
        'Execution Time': elapsed_time,
        'CPU usage': cpu_usage
    }
model_metrics_performance.append(metrics_dict_time)


# ISOTONIC REGRESSION #
start_time = time.time()
print("\t------- Isotonic Regression -------")
iso = IsotonicRegression(labelCol="price", featuresCol="features")
iso_model = iso.fit(train_data)
metrics_dict_iso = reg_metrics(iso_model, train_data, test_data, 'Isotonic Regression')
model_metrics.append(metrics_dict_iso)
print("\n\t--- Predictions ---")
pred_results = iso_model.transform(test_data_sample)
pred_results.show(5)

end_time = time.time()
elapsed_time = end_time - start_time
# Measure CPU usage
cpu_usage = psutil.cpu_percent()

metrics_dict_time = {
        'Algorithm':'Isotonic Regression',
        'Execution Time': elapsed_time,
        'CPU usage': cpu_usage
    }
model_metrics_performance.append(metrics_dict_time)


# FACTORIZATION MACHINES REGRESSION #
start_time = time.time()
print("\t------- Factorization Machines Regression -------")
# Define the FMRegressor model
fm = FMRegressor(featuresCol="features", labelCol="price", stepSize=0.01)
# Train the model on the training data
fm_model = fm.fit(train_data)

# Evaluate the model on the training and test data
metrics_dict_fm = reg_metrics(fm_model, train_data, test_data, 'Factorization Machines Regression')
model_metrics.append(metrics_dict_fm)
print("\n\t--- Predictions ---")
# Make predictions on the test data
pred_results = fm_model.transform(test_data_sample)
pred_results.show(5)

end_time = time.time()
elapsed_time = end_time - start_time
# Measure CPU usage
cpu_usage = psutil.cpu_percent()

metrics_dict_time = {
        'Algorithm':'Factorization Machines',
        'Execution Time': elapsed_time,
        'CPU usage': cpu_usage
    }
model_metrics_performance.append(metrics_dict_time)




# DECISION TREE #
start_time = time.time()
print("\t------- Decision Tree -------")
dt = DecisionTreeRegressor(featuresCol = 'features', labelCol = 'price', maxDepth=5, maxBins=40000)
dt_model = dt.fit(train_data)
metrics_dict_dt = reg_metrics(dt_model, train_data, test_data, 'Decision Tree')
model_metrics.append(metrics_dict_dt)
print("\n\t--- Predictions ---")
pred_results = dt_model.transform(test_data_sample)
pred_results.show(5)

end_time = time.time()
elapsed_time = end_time - start_time

# Measure CPU usage
cpu_usage = psutil.cpu_percent()

metrics_dict_time = {
        'Algorithm':'Decision Tree',
        'Execution Time': elapsed_time,
        'CPU usage': cpu_usage
    }
model_metrics_performance.append(metrics_dict_time)


# RANDOM FOREST REGRESSION #
start_time = time.time()
print("\t------- Random Forest Regression -------")
rf = RandomForestRegressor(featuresCol = 'features', labelCol = 'price', numTrees=2, maxDepth=2, maxBins=40000)
rf_model = rf.fit(train_data)
metrics_dict_rf = reg_metrics(rf_model, train_data, test_data, 'Random Forest Regression')
model_metrics.append(metrics_dict_rf)
print("\n\t--- Predictions ---")
pred_results = rf_model.transform(test_data_sample)
pred_results.show(5)

end_time = time.time()
elapsed_time = end_time - start_time
# Measure CPU usage
cpu_usage = psutil.cpu_percent()

metrics_dict_time = {
        'Algorithm':'Random Forest',
        'Execution Time': elapsed_time,
        'CPU usage': cpu_usage
    }
model_metrics_performance.append(metrics_dict_time)


# GRADIENT BOOSTING REGRESSION #
start_time = time.time()
print("\t------- Gradient Boosting Regression -------")
gb = GBTRegressor(featuresCol = 'features', labelCol = 'price', maxIter=10, maxDepth=5, seed=42, maxBins=40000)
gb_model = gb.fit(train_data)
metrics_dict_gb = reg_metrics(gb_model, train_data, test_data, 'Gradient Boost Regression')
model_metrics.append(metrics_dict_gb)
print("\n\t--- Predictions ---")
pred_results = gb_model.transform(test_data_sample)
pred_results.show(5)

end_time = time.time()
elapsed_time = end_time - start_time
# Measure CPU usage
cpu_usage = psutil.cpu_percent()

metrics_dict_time = {
        'Algorithm':'Gradient Boosting',
        'Execution Time': elapsed_time,
        'CPU usage': cpu_usage
    }
model_metrics_performance.append(metrics_dict_time)


# GENERALIZED LINEAR REGRESSION #
start_time = time.time()
print("\t------- Generalized Linear Regression -------")
glr = GeneralizedLinearRegression(featuresCol = 'features', labelCol = 'price',family="gaussian", link="identity", maxIter=10, regParam=0.3)
glr_model = glr.fit(train_data)
metrics_dict_glr = reg_metrics(glr_model, train_data, test_data, 'Generalized Linear Regression')
model_metrics.append(metrics_dict_glr)
print("\n\t--- Predictions ---")
pred_results = glr_model.transform(test_data_sample)
pred_results.show(5)

end_time = time.time()
elapsed_time = end_time - start_time
# Measure CPU usage
cpu_usage = psutil.cpu_percent()


metrics_dict_time = {
        'Algorithm':'Generalized Linear Regression',
        'Execution Time': elapsed_time,
        'CPU usage': cpu_usage
    }
model_metrics_performance.append(metrics_dict_time)


print('------------------------------------------')
print('----------- Models Comparison ------------')
print('------------------------------------------')
print('')


# Create a Spark DataFrame from the list of dictionaries
df_models1 = spark.createDataFrame(model_metrics)

# Select columns in desired order and round the values to two decimal places
df_models = df_models1.withColumn("MSE", col("MSE").cast("decimal(20,2)")).select("Algorithm", 
                              round("R²", 2).alias("R²"), 
                              round("Adjusted R²", 2).alias("Adjusted R²"), 
                              "MSE", 
                              round("RMSE", 2).alias("RMSE"), 
                              round("MAE", 2).alias("MAE"))

# Set the option to display the full column width
df_models.show()



# Define the schema of the DataFrame
schema = StructType([
    StructField("Algorithm", StringType(), True),
    StructField("Execution Time", DoubleType(), True),
    StructField("CPU usage", DoubleType(), True),

])

# Create the DataFrame using the schema
df_models_performance = spark.createDataFrame(model_metrics_performance, schema=schema)
df_models_performance.show()
