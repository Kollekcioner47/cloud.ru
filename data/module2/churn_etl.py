#!/usr/bin/env python3
"""
Основной ETL-пайплайн для расчёта признаков оттока клиентов - ИСПРАВЛЕННАЯ ВЕРСИЯ
Использует максимальную дату транзакций как точку отсчета вместо текущей даты
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

def main():
    # Инициализация Spark с оптимизациями
    spark = SparkSession.builder \
        .appName("Churn_ETL_Fixed") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # Настройка логирования - только предупреждения и ошибки
    spark.sparkContext.setLogLevel("WARN")
    
    print("=== Запуск ИСПРАВЛЕННОГО ETL-пайплайна анализа оттока ===")
    
    # ШАГ 1: EXTRACT - ЗАГРУЗКА ДАННЫХ
    print("1. Загрузка данных...")
    
    # Схема транзакций (упрощенная - только нужные поля)
    tx_schema = StructType([
        StructField("customer_id", StringType()),      # ID клиента
        StructField("tx_amount", DoubleType()),        # Сумма транзакции
        StructField("tx_date", DateType())             # Дата транзакции
    ])
    
    # Схема клиентов - ТОЛЬКО СУЩЕСТВУЮЩИЕ ПОЛЯ
    cust_schema = StructType([
        StructField("customer_id", StringType()),      # ID клиента
        StructField("registration_date", DateType()),  # Дата регистрации
        StructField("region", StringType())            # Регион
    ])
    
    # Чтение ОЧИЩЕННЫХ данных из S3
    try:
        # Загрузка транзакций из подготовленной зоны
        df_tx = spark.read.schema(tx_schema).parquet("s3a://bucket-ml/processed/cleaned/transactions/")
        # Загрузка данных клиентов из подготовленной зоны
        df_customers = spark.read.schema(cust_schema).parquet("s3a://bucket-ml/processed/cleaned/customers/")
        
        # ДИАГНОСТИКА: вывод статистики загрузки
        print(f"Загружено {df_tx.count()} транзакций для {df_tx.select('customer_id').distinct().count()} клиентов")
        print(f"Загружено {df_customers.count()} записей клиентов")
        
        # ОПРЕДЕЛЯЕМ МАКСИМАЛЬНУЮ ДАТУ ТРАНЗАКЦИЙ
        max_tx_date = df_tx.agg(max("tx_date")).collect()[0][0]
        print(f"Самая последняя транзакция в данных: {max_tx_date}")
        
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        spark.stop()
        return  # Завершаем выполнение при ошибке загрузки
    
    # ШАГ 2: TRANSFORM - РАСЧЁТ ПРИЗНАКОВ ОТТОКА ОТНОСИТЕЛЬНО ПОСЛЕДНЕЙ ТРАНЗАКЦИИ
    print("2. Расчёт признаков оттока...")
    
    # АГРЕГАЦИЯ ТРАНЗАКЦИЙ ПО КЛИЕНТАМ:
    df_agg = df_tx.groupBy("customer_id").agg(
        max("tx_date").alias("last_tx_date"),          # Дата последней транзакции
        min("tx_date").alias("first_tx_date"),         # Дата первой транзакции
        mean("tx_amount").alias("avg_tx_amount"),      # Средний чек
        stddev("tx_amount").alias("std_tx_amount"),    # Стандартное отклонение сумм
        count("*").alias("total_tx_count"),            # Общее количество транзакций
        sum("tx_amount").alias("total_tx_amount")      # Общая сумма всех транзакций
    )
    
    # РАСЧЁТ ДОПОЛНИТЕЛЬНЫХ ПРИЗНАКОВ С ФИКСИРОВАННОЙ ДАТОЙ ОТСЧЕТА:
    df_features = df_agg.withColumn(
        # ИСПРАВЛЕНИЕ: используем максимальную дату транзакций вместо current_date()
        "days_since_last_tx", datediff(lit(max_tx_date), col("last_tx_date"))
    ).withColumn(
        "customer_lifetime_days", datediff(col("last_tx_date"), col("first_tx_date"))
    ).withColumn(
        # ОПРЕДЕЛЕНИЕ ОТТОКА: если с последней транзакции прошло более 30 дней
        # ОТНОСИТЕЛЬНО МАКСИМАЛЬНОЙ ДАТЫ В ДАННЫХ
        "is_churned", when(col("days_since_last_tx") > 30, 1).otherwise(0)
    ).withColumn(
        # ЧАСТОТА ТРАНЗАКЦИЙ: среднее количество транзакций в день
        "tx_frequency", col("total_tx_count") / greatest(col("customer_lifetime_days"), lit(1))
    )
    
    # ОБЪЕДИНЕНИЕ С ДАННЫМИ КЛИЕНТОВ:
    df_final = df_features.join(df_customers, "customer_id", "left")
    
    # ДОБАВЛЕНИЕ ТЕХНИЧЕСКИХ ПОЛЕЙ И МЕТАДАННЫХ:
    df_final = df_final.withColumn("processing_date", current_date()) \
                      .withColumn("processing_timestamp", current_timestamp()) \
                      .withColumn("analysis_reference_date", lit(max_tx_date))  # Сохраняем дату отсчета
    
    # ШАГ 3: ПРОВЕРКА ДАННЫХ (DATA QUALITY CHECKS)
    print("3. Проверка данных...")
    
    # КРИТИЧЕСКАЯ ПРОВЕРКА: распределение оттока
    print("РАСПРЕДЕЛЕНИЕ ОТТОКА:")
    churn_stats = df_final.groupBy("is_churned").count().collect()
    for row in churn_stats:
        percentage = (row['count'] / df_final.count()) * 100
        print(f"  is_churned={row['is_churned']}: {row['count']} записей ({percentage:.1f}%)")
    
    # ПРОСМОТР ОБРАЗЦА ДАННЫХ:
    print("Образец данных:")
    df_final.select("customer_id", "last_tx_date", "days_since_last_tx", "is_churned").show(10, truncate=False)
    
    # СТАТИСТИКА ПО ЧИСЛОВЫМ ПРИЗНАКАМ:
    print("Статистика по числовым признакам:")
    df_final.select("avg_tx_amount", "days_since_last_tx", "total_tx_count", "tx_frequency").describe().show()
    
    # АНАЛИЗ РАСПРЕДЕЛЕНИЯ ОТТОКА ПО РЕГИОНАМ:
    print("Распределение оттока по регионам:")
    df_final.groupBy("region", "is_churned").count().orderBy("region", "is_churned").show()
    
    # ШАГ 4: LOAD - СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    print("4. Сохранение результатов...")
    
    # ВАРИАНТ 1: Parquet с партиционированием (для ML и аналитиков)
    df_final.write \
        .mode("overwrite") \
        .partitionBy("region") \
        .parquet("s3a://bucket-ml/processed/churn_features/")
    
    # ВАРИАНТ 2: CSV для аналитиков (легче открывать в Excel)
    df_final.write \
        .mode("overwrite") \
        .option("header", "true") \
        .option("delimiter", ";") \
        .csv("s3a://bucket-ml/processed/datamarts/churn_features_daily")
    
    # ДОПОЛНИТЕЛЬНО: АГРЕГИРОВАННАЯ СТАТИСТИКА ПО РЕГИОНАМ
    df_regional_stats = df_final.groupBy("region").agg(
        count("*").alias("total_customers"),
        avg("avg_tx_amount").alias("avg_region_tx_amount"),
        avg("days_since_last_tx").alias("avg_days_since_last_tx"),
        sum("is_churned").alias("churned_customers"),
        (sum("is_churned") / count("*") * 100).alias("churn_rate_percent")
    )
    
    # Сохранение региональной статистики
    df_regional_stats.write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv("s3a://bucket-ml/processed/datamarts/regional_stats")
    
    # ФИНАЛЬНЫЙ ОТЧЁТ:
    print("5. Готово! Результаты сохранены в S3:")
    print(f"   Дата отсчета анализа: {max_tx_date}")
    print("   - s3a://bucket-ml/processed/churn_features/ (Parquet) - для ML моделей")
    print("   - s3a://bucket-ml/processed/datamarts/churn_features_daily (CSV) - для аналитиков")
    print("   - s3a://bucket-ml/processed/datamarts/regional_stats (CSV) - агрегированная статистика")
    
    # Завершение сессии Spark (освобождение ресурсов)
    spark.stop()

# Точка входа при прямом запуске скрипта
if __name__ == "__main__":
    main()