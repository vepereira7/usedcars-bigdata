from io import BytesIO, StringIO
from google.cloud import storage

import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf 
from pyspark.sql import types
from pyspark.sql import functions as F
from pyspark.ml.feature import Imputer, VectorAssembler, StringIndexer, OneHotEncoder, RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, Normalizer
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
from pyspark.ml.linalg import SparseVector
from pyspark.ml.feature import Bucketizer, SQLTransformer, IndexToString, VectorIndexer
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import pyspark.sql.functions as F
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


print('-----------------------------------------')
print('--------- Recomendation System ----------')
print('-----------------------------------------')
print('')



selected_colors = ['white', 'silver', 'yellow', 'orange', 'green', 'custom', 'black', 'red', 'blue', 'purple', 'grey', 'brown']
selected_types = ['sedan', 'convertible', 'coupe', 'hatchback', 'other', 'SUV', 'wagon', 'pickup', 'offroad', 'truck', 'van', 'mini-van', 'bus']

rec_df = df.filter((col('paint_color').isin(selected_colors)) & (col('type').isin(selected_types))).select(*df.columns)

rec_df.show()

print("\nCreate manufacturer countries, this means, the 'Made' column\n")

def country(manufacturer):
    american = ['harley-davidson', 'chevrolet', 'pontiac', 'ram', 'ford', 'gmc', 'tesla', 'jeep', 'dodge',
                'cadillac', 'chrysler', 'lincoln', 'buick', 'saturn', 'mercury']
    japanese = ['lexus', 'nissan', 'toyota', 'acura', 'honda', 'infiniti', 'subaru', 'mitsubishi', 'datsun', 'mazda']
    german = ['volkswagen', 'mercedes-benz', 'bmw', 'audi', 'porsche']
    italian = ['ferrari', 'fiat', 'alfa-romeo']
    korean = ['kia', 'hyundai']
    swedish = ['volvo']
    english = ['rover', 'mini', 'land rover', 'jaguar']
    
    return when(col('manufacturer').isin(american), 'American') \
        .when(col('manufacturer').isin(japanese), 'Japanese') \
        .when(col('manufacturer').isin(german), 'German') \
        .when(col('manufacturer').isin(italian), 'Italian') \
        .when(col('manufacturer').isin(korean), 'Korean') \
        .when(col('manufacturer').isin(swedish), 'Swedish') \
        .when(col('manufacturer').isin(english), 'English') \
        .otherwise(None)

rec_df = rec_df.withColumn('Made', country(col('manufacturer')))

rec_df.select('Made', 'manufacturer').show()


print("\nCreate cars age, this means, the 'age' column\n")

# Add a new column for age of cars
rec_df = rec_df.withColumn('age', year('posting_date') - rec_df['year'])


print("\nCreate cars mil rating, this means, the 'mil_rating' column\n")

rec_df = rec_df.withColumn("avg_mil", col("odometer") / col("Age"))
rec_df = rec_df.withColumn("mil_rating", when(rec_df['avg_mil'] > 21500, "above average").otherwise("below average"))

print("\nCreate cars luxury division, this means, the 'type_group' column\n")

# Define UDF for the luxury function
luxury_udf = udf(lambda type_: "luxury_small" if type_ in ['sedan', 'convertible','coupe','hatchback','other']
                 else "luxury_large" if type_ in ['SUV','wagon']
                 else "non-luxury_small" if type_ in ['pickup','truck','offroad']
                 else "non-luxury_large" if type_ in ['van','mini-van','bus']
                 else None, StringType())

# Apply UDF to create new column in DataFrame
rec_df = rec_df.withColumn('type_group', luxury_udf('type'))

rec_df.select('type_group', 'type').show()

print("\nCreate cars colours division, this means, the 'color_group' column\n")

# create column 'color_group' based on 'paint_color'
rec_df = rec_df.withColumn('color_group', when(col('paint_color').isin(['white','silver','yellow','orange','green','custom']), 'light color').otherwise('dark color'))

rec_df.select('paint_color', 'color_group').show()

print("\nPrepare dataset to be used in the recognition system\n")

cols_to_drop = ['id','url', 'region', 'region_url', 'VIN', 'image_url', 'description', \
    'county', 'size', 'drive', 'cylinders', 'state', 'lat','long']
rec_df = rec_df.drop(*cols_to_drop)

# Remove null values and duplicated rows
rec_df = rec_df.drop_duplicates()

# Initial cleaning up
# Drop NaNs and duplicates
rec_df = rec_df.na.drop()

rec_df.printSchema()


# Function for recommending cars based on car manufacturer country. 
# It takes car manufacturer country, color group, type group  and price range as input.

def recommend(made, color_group, type_group, price_range):
    # Matching the type with the dataset
    data = rec_df.filter((rec_df.color_group == color_group) & 
                         (rec_df.type_group == type_group) &
                         (rec_df.price >= price_range[0]) &
                         (rec_df.price <= price_range[1]) &
                         (rec_df.Made == made))
    
    # Convert the car manufacturer country into vectors and used unigram
    tokenizer = RegexTokenizer(inputCol="Made", outputCol="words", pattern="\\W")
    stop_words_remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    cv = CountVectorizer(inputCol="filtered", outputCol="raw_features", vocabSize=2**16, minDF=1.0)
    idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=5)
    
    pipeline = Pipeline(stages=[tokenizer, stop_words_remover, cv, idf])
    pipeline_model = pipeline.fit(data)
    tfidf_matrix = pipeline_model.transform(data)
    
    # Calculating the similarity measures based on Cosine Similarity
    assembler = VectorAssembler(inputCols=["features"], outputCol="features_vec")
    assembled_data = assembler.transform(tfidf_matrix)
    normalizer = Normalizer(inputCol="features_vec", outputCol="features_norm", p=2.0)
    normalized_data = normalizer.transform(assembled_data)
    
    # Selecionando o top 5 de carros semelhantes
    idx = normalized_data.select('features_norm').collect()[0]
    sig = normalized_data.rdd.map(lambda row: (row.price, row.features_norm.dot(idx[0]))) \
                             .sortBy(lambda x: -x[1]) \
                             .take(6)[1:]

    # car indices
    car_indices = [i[0] for i in sig]
    
    # Top 5 car recommendations
    rec = data.select('price', 'Made', 'manufacturer', 'model', 'type', 'year', 'Age', 'condition', 'fuel',
                      'title_status', 'transmission', 'paint_color', 'mil_rating') \
              .filter(F.col('price').isin(car_indices)) \
              .orderBy('price') \
              .drop('id')
    rec_splits = rec.randomSplit([0.1, 0.1], seed=42)
    rec_sample = rec_splits[0].limit(5)
    return rec_sample.show()



print('')
print('')
print('-----------------------------------------------')
print('-------------- Recommendations ----------------')
print('-----------------------------------------------')
print('')
print('')


print('Recommendation 1\n')
recommend("Japanese", "light color", "luxury_small", (5000, 6000))

print('\nRecommendation 2\n')
recommend("American", "dark color", "luxury_large", (1000, 20000))

print('\nRecommendation 3\n')
recommend("German", "light color", "luxury_small", (1000, 6000))

print('\nRecommendation 4\n')
recommend("Italian", "light color", "luxury_small", (1000, 5000000))

print('\nRecommendation 5\n')
recommend("Korean", "light color", "luxury_small", (3000, 20000))




