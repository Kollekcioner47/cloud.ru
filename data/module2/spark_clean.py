#!/usr/bin/env python3
"""
Скрипт для очистки и валидации сырых данных - АДАПТИРОВАННЫЙ ПОД СУЩЕСТВУЮЩИЕ ДАННЫЕ
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from datetime import datetime, timedelta

def clean_transactions_data(spark):
    """Очистка данных транзакций"""
    print("Очистка данных транзакций...")
    
    # Схема транзакций (соответствует существующим данным)
    tx_schema = StructType([
        StructField("transaction_id", StringType()),
        StructField("customer_id", StringType()),
        StructField("tx_amount", DoubleType()),
        StructField("tx_date", DateType())
    ])
    
    # Загрузка сырых данных
    df_tx = spark.read.schema(tx_schema).parquet("s3a://bucket-ml/raw/transactions/")
    
    original_count = df_tx.count()
    print(f"Исходное количество транзакций: {original_count}")
    
    # Очистка данных - упрощенная версия
    df_clean = df_tx.dropna(subset=["customer_id", "tx_amount", "tx_date"])
    
    # Фильтры по отдельности чтобы избежать проблем с приоритетом операторов
    df_clean = df_clean.filter(length(trim(col("customer_id"))) > 0)
    df_clean = df_clean.filter(col("tx_amount") > 0)
    df_clean = df_clean.filter(col("tx_amount") <= 1000000)
    
    # Фильтр по дате
    min_date = datetime.now() - timedelta(days=365*5)
    df_clean = df_clean.filter(col("tx_date") >= lit(min_date))
    df_clean = df_clean.filter(col("tx_date") <= current_date())
    
    # Удаление дубликатов
    df_clean = df_clean.dropDuplicates(["customer_id", "tx_date", "tx_amount"])
    
    cleaned_count = df_clean.count()
    removed_count = original_count - cleaned_count
    
    print(f"После очистки: {cleaned_count} транзакций")
    print(f"Удалено записей: {removed_count} ({removed_count/original_count*100:.2f}%)")
    
    # Статистика по очистке
    if cleaned_count > 0:
        print("Статистика сумм транзакций:")
        df_clean.select(
            mean("tx_amount").alias("avg_amount"),
            stddev("tx_amount").alias("std_amount"),
            min("tx_amount").alias("min_amount"),
            max("tx_amount").alias("max_amount")
        ).show()
    
    # Сохранение очищенных данных
    df_clean.write.mode("overwrite").parquet("s3a://bucket-ml/processed/cleaned/transactions/")
    
    return df_clean

def clean_customers_data(spark):
    """Очистка данных клиентов - АДАПТИРОВАННАЯ ВЕРСИЯ"""
    print("Очистка данных клиентов...")
    
    # Схема клиентов - ТОЛЬКО ТЕ ПОЛЯ, КОТОРЫЕ ЕСТЬ В ДАННЫХ
    cust_schema = StructType([
        StructField("customer_id", StringType()),
        StructField("registration_date", DateType()),
        StructField("region", StringType())
        # age и customer_segment отсутствуют в данных
    ])
    
    # Загрузка сырых данных
    df_customers = spark.read.schema(cust_schema).json("s3a://bucket-ml/raw/customers/")
    
    original_count = df_customers.count()
    print(f"Исходное количество клиентов: {original_count}")
    
    # Очистка данных - только доступные поля
    df_clean = df_customers.dropna(subset=["customer_id", "registration_date", "region"])
    
    # Фильтры по отдельности
    df_clean = df_clean.filter(length(trim(col("customer_id"))) > 0)
    df_clean = df_clean.filter(length(trim(col("region"))) > 0)
    
    # Фильтр по дате регистрации
    min_date = datetime.now() - timedelta(days=365*10)
    df_clean = df_clean.filter(col("registration_date") >= lit(min_date))
    df_clean = df_clean.filter(col("registration_date") <= current_date())
    
    # Удаление дубликатов клиентов
    df_clean = df_clean.dropDuplicates(["customer_id"])
    
    cleaned_count = df_clean.count()
    removed_count = original_count - cleaned_count
    
    print(f"После очистки: {cleaned_count} клиентов")
    print(f"Удалено записей: {removed_count} ({removed_count/original_count*100:.2f}%)")
    
    # Статистика по регионам
    if cleaned_count > 0:
        print("Распределение по регионам:")
        df_clean.groupBy("region").count().orderBy("count", ascending=False).show()
    
    # Сохранение очищенных данных
    df_clean.write.mode("overwrite").parquet("s3a://bucket-ml/processed/cleaned/customers/")
    
    return df_clean

def validate_data_quality(spark, df_tx, df_customers):
    """Валидация качества данных"""
    print("Валидация качества данных...")
    
    if df_tx.count() == 0 or df_customers.count() == 0:
        print("⚠️ Один из датасетов пуст, пропускаем валидацию")
        return
    
    # Проверка покрытия клиентов
    tx_customers = df_tx.select("customer_id").distinct()
    all_customers = df_customers.select("customer_id")
    
    missing_in_tx = all_customers.subtract(tx_customers)
    missing_in_customers = tx_customers.subtract(all_customers)
    
    print(f"Клиенты без транзакций: {missing_in_tx.count()}")
    print(f"Клиенты без профиля: {missing_in_customers.count()}")
    
    # Проверка временного диапазона
    date_stats = df_tx.agg(
        min("tx_date").alias("first_tx_date"),
        max("tx_date").alias("last_tx_date")
    ).collect()[0]
    
    print(f"Диапазон транзакций: {date_stats['first_tx_date']} - {date_stats['last_tx_date']}")
    
    # Проверка распределения по дням
    print("Транзакции по дням (топ-10):")
    df_tx.groupBy("tx_date").count().orderBy("count", ascending=False).show(10)

def main():
    """Основная функция"""
    spark = SparkSession.builder \
        .appName("Data_Cleaning_Adapted") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("=== ЗАПУСК АДАПТИРОВАННОЙ ОЧИСТКИ ДАННЫХ ===")
    
    try:
        # Очистка данных
        df_tx_clean = clean_transactions_data(spark)
        df_cust_clean = clean_customers_data(spark)
        
        # Валидация качества
        validate_data_quality(spark, df_tx_clean, df_cust_clean)
        
        print("✅ Очистка данных завершена успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка при очистке данных: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        spark.stop()

if __name__ == "__main__":
    main()