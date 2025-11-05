#!/usr/bin/env python3
"""
Основной ETL-пайплайн для расчёта признаков оттока клиентов - АДАПТИРОВАННЫЙ
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

def main():
    # Инициализация Spark
    spark = SparkSession.builder \
        .appName("Churn_ETL_Adapted") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # Настройка логирования
    spark.sparkContext.setLogLevel("WARN")
    
    print("=== Запуск АДАПТИРОВАННОГО ETL-пайплайна анализа оттока ===")
    
    # Шаг 1: Загрузка данных
    print("1. Загрузка данных...")
    
    # Схема транзакций
    tx_schema = StructType([
        StructField("customer_id", StringType()),
        StructField("tx_amount", DoubleType()),
        StructField("tx_date", DateType())
    ])
    
    # Схема клиентов - ТОЛЬКО СУЩЕСТВУЮЩИЕ ПОЛЯ
    cust_schema = StructType([
        StructField("customer_id", StringType()),
        StructField("registration_date", DateType()),
        StructField("region", StringType())
        # age и customer_segment отсутствуют
    ])
    
    # Чтение данных из S3
    try:
        df_tx = spark.read.schema(tx_schema).parquet("s3a://bucket-ml/processed/cleaned/transactions/")
        df_customers = spark.read.schema(cust_schema).parquet("s3a://bucket-ml/processed/cleaned/customers/")
        
        print(f"Загружено {df_tx.count()} транзакций для {df_tx.select('customer_id').distinct().count()} клиентов")
        print(f"Загружено {df_customers.count()} записей клиентов")
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        spark.stop()
        return
    
    # Шаг 2: Расчёт признаков оттока
    print("2. Расчёт признаков оттока...")
    
    # Агрегация транзакций по клиентам
    df_agg = df_tx.groupBy("customer_id").agg(
        max("tx_date").alias("last_tx_date"),
        min("tx_date").alias("first_tx_date"),
        mean("tx_amount").alias("avg_tx_amount"),
        stddev("tx_amount").alias("std_tx_amount"),
        count("*").alias("total_tx_count"),
        sum("tx_amount").alias("total_tx_amount")
    )
    
    # Расчёт дополнительных признаков
    df_features = df_agg.withColumn(
        "days_since_last_tx", datediff(current_date(), col("last_tx_date"))
    ).withColumn(
        "customer_lifetime_days", datediff(col("last_tx_date"), col("first_tx_date"))
    ).withColumn(
        "is_churned", when(col("days_since_last_tx") > 30, 1).otherwise(0)
    ).withColumn(
        "tx_frequency", col("total_tx_count") / greatest(col("customer_lifetime_days"), lit(1))
    )
    
    # Объединение с данными клиентов
    df_final = df_features.join(df_customers, "customer_id", "left")
    
    # Добавление временных меток
    df_final = df_final.withColumn("processing_date", current_date()) \
                      .withColumn("processing_timestamp", current_timestamp())
    
    # Шаг 3: Проверка данных
    print("3. Проверка данных...")
    print("Образец данных:")
    df_final.show(10, truncate=False)
    
    print("Статистика по числовым признакам:")
    df_final.select("avg_tx_amount", "days_since_last_tx", "total_tx_count", "tx_frequency").describe().show()
    
    print("Распределение оттока по регионам:")
    df_final.groupBy("region", "is_churned").count().orderBy("region", "is_churned").show()
    
    # Шаг 4: Сохранение результатов
    print("4. Сохранение результатов...")
    
    # Сохранение в Parquet с партиционированием по региону
    df_final.write \
        .mode("overwrite") \
        .partitionBy("region") \
        .parquet("s3a://bucket-ml/processed/churn_features/")
    
    # Сохранение в CSV для аналитиков
    df_final.write \
        .mode("overwrite") \
        .option("header", "true") \
        .option("delimiter", ";") \
        .csv("s3a://bucket-ml/processed/datamarts/churn_features_daily")
    
    # Сохранение агрегированной статистики по регионам
    df_regional_stats = df_final.groupBy("region").agg(
        count("*").alias("total_customers"),
        avg("avg_tx_amount").alias("avg_region_tx_amount"),
        avg("days_since_last_tx").alias("avg_days_since_last_tx"),
        sum("is_churned").alias("churned_customers"),
        (sum("is_churned") / count("*") * 100).alias("churn_rate_percent")
    )
    
    df_regional_stats.write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv("s3a://bucket-ml/processed/datamarts/regional_stats")
    
    print("5. Готово! Результаты сохранены в S3:")
    print("   - s3a://bucket-ml/processed/churn_features/ (Parquet)")
    print("   - s3a://bucket-ml/processed/datamarts/churn_features_daily (CSV)")
    print("   - s3a://bucket-ml/processed/datamarts/regional_stats (CSV)")
    
    # Завершение сессии
    spark.stop()

if __name__ == "__main__":
    main()