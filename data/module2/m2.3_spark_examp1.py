from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler, Normalizer
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

# 1. Инициализация SparkSession с настройками для Cloud.ru
spark = SparkSession.builder \
    .appName("feature_processing_example") \
    .enableHiveSupport() \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

try:
    # 2. Чтение данных с явной схемой (лучшая практика)
    customer_schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("age", DoubleType(), True),
        StructField("income", DoubleType(), True),
        StructField("region", StringType(), True)
    ])
    
    # Альтернатива: чтение из Hive metastore
    # df = spark.table("hive.clean.customers")
    df = spark.read.parquet("s3a://bucket-ml/tables/customers/")
    
    # 3. Предобработка данных 
    df_clean = df.filter(
        col("age").isNotNull() & 
        col("income").isNotNull() &
        (col("age") >= 0) & 
        (col("age") <= 100) &  # разумные границы возраста
        (col("income") >= 0)   # доход не может быть отрицательным
    )
    
    # 4. Замена выбросов (опционально, но рекомендуется)
    df_processed = df_clean.withColumn(
        "income", 
        when(col("income") > 1000000, 1000000).otherwise(col("income"))  # cap outliers
    )
    
    # 5. Векторизация признаков с обработкой ошибок
    assembler = VectorAssembler(
        inputCols=["age", "income"],
        outputCol="features"
    )
    
    df_with_features = assembler.transform(df_processed)
    
    # 6. Масштабирование StandardScaler
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withStd=True,
        withMean=True
    )
    
    scaler_model = scaler.fit(df_with_features)
    scaled_df = scaler_model.transform(df_with_features)
    
    # 7. Нормализация L2
    normalizer = Normalizer(
        inputCol="scaled_features",  # нормализуем уже масштабированные признаки
        outputCol="normalized_features",
        p=2.0
    )
    
    normalized_df = normalizer.transform(scaled_df)
    
    # 8. Сохранение результатов 
    normalized_df.write \
        .mode("overwrite") \
        .parquet("s3a://bucket-ml/tables/customers_processed/")
    
    # Альтернатива: сохранение как Hive таблицы
    # normalized_df.write \
    #     .mode("overwrite") \
    #     .saveAsTable("hive.clean.customers_processed")
    
    print("Feature processing completed successfully!")
    print(f"Processed {normalized_df.count()} records")
finally:
    spark.stop()