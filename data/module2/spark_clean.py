#!/usr/bin/env python3
"""
Скрипт для очистки и валидации сырых данных - АДАПТИРОВАННЫЙ ПОД СУЩЕСТВУЮЩИЕ ДАННЫЕ

НАЗНАЧЕНИЕ СКРИПТА:
- Загрузка сырых данных из S3
- Очистка и валидация данных транзакций и клиентов
- Удаление некорректных записей, дубликатов
- Проверка качества данных
- Сохранение очищенных данных обратно в S3

ОСНОВНЫЕ ЭТАПЫ:
1. Очистка транзакций (clean_transactions_data)
2. Очистка данных клиентов (clean_customers_data) 
3. Валидация качества данных (validate_data_quality)
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from datetime import datetime, timedelta

def clean_transactions_data(spark):
    """
    ОЧИСТКА ДАННЫХ ТРАНЗАКЦИЙ
    
    Параметры:
    spark - SparkSession для работы с DataFrame
    
    Возвращает:
    df_clean - очищенный DataFrame с транзакциями
    """
    print("Очистка данных транзакций...")
    
    # Схема транзакций (соответствует существующим данным)
    # Определяем структуру данных заранее для контроля типов
    tx_schema = StructType([
        StructField("transaction_id", StringType()),    # ID транзакции
        StructField("customer_id", StringType()),       # ID клиента
        StructField("tx_amount", DoubleType()),         # Сумма транзакции
        StructField("tx_date", DateType())              # Дата транзакции
    ])
    
    # Загрузка сырых данных из S3 с указанной схемой
    # parquet - колоночный формат, эффективный для аналитических запросов
    df_tx = spark.read.schema(tx_schema).parquet("s3a://bucket-ml/raw/transactions/")
    
    # Сохраняем исходное количество для статистики
    original_count = df_tx.count()
    print(f"Исходное количество транзакций: {original_count}")
    
    # БАЗОВАЯ ОЧИСТКА: удаление записей с пропущенными значениями
    # subset - проверяем только ключевые поля
    df_clean = df_tx.dropna(subset=["customer_id", "tx_amount", "tx_date"])
    
    # ФИЛЬТРАЦИЯ ДАННЫХ (раздельно для лучшей читаемости):
    
    # 1. Удаляем пустые customer_id (после обрезки пробелов)
    df_clean = df_clean.filter(length(trim(col("customer_id"))) > 0)
    
    # 2. Оставляем только положительные суммы транзакций
    df_clean = df_clean.filter(col("tx_amount") > 0)
    
    # 3. Устанавливаем верхний предел для исключения аномалий
    df_clean = df_clean.filter(col("tx_amount") <= 1000000)
    
    # ФИЛЬТРАЦИЯ ПО ДАТЕ:
    # Исключаем транзакции старше 5 лет и будущие даты
    min_date = datetime.now() - timedelta(days=365*5)
    df_clean = df_clean.filter(col("tx_date") >= lit(min_date))
    df_clean = df_clean.filter(col("tx_date") <= current_date())
    
    # УДАЛЕНИЕ ДУБЛИКАТОВ:
    # Удаляем записи с одинаковым клиентом, датой и суммой
    df_clean = df_clean.dropDuplicates(["customer_id", "tx_date", "tx_amount"])
    
    # СТАТИСТИКА ОЧИСТКИ:
    cleaned_count = df_clean.count()
    removed_count = original_count - cleaned_count
    
    print(f"После очистки: {cleaned_count} транзакций")
    print(f"Удалено записей: {removed_count} ({removed_count/original_count*100:.2f}%)")
    
    # ДЕТАЛЬНАЯ СТАТИСТИКА ПО СУММАМ ТРАНЗАКЦИЙ:
    if cleaned_count > 0:
        print("Статистика сумм транзакций:")
        df_clean.select(
            mean("tx_amount").alias("avg_amount"),      # Среднее значение
            stddev("tx_amount").alias("std_amount"),    # Стандартное отклонение
            min("tx_amount").alias("min_amount"),       # Минимальная сумма
            max("tx_amount").alias("max_amount")        # Максимальная сумма
        ).show()  # show() выводит результат в консоль
    
    # СОХРАНЕНИЕ РЕЗУЛЬТАТА:
    # mode("overwrite") - перезаписываем предыдущие результаты
    df_clean.write.mode("overwrite").parquet("s3a://bucket-ml/processed/cleaned/transactions/")
    
    return df_clean

def clean_customers_data(spark):
    """
    ОЧИСТКА ДАННЫХ КЛИЕНТОВ - АДАПТИРОВАННАЯ ВЕРСИЯ
    
    Особенности:
    - Работает только с доступными полями
    - Данные загружаются из JSON (в отличие от Parquet для транзакций)
    
    Параметры:
    spark - SparkSession для работы с DataFrame
    
    Возвращает:
    df_clean - очищенный DataFrame с клиентами
    """
    print("Очистка данных клиентов...")
    
    # Схема клиентов - ТОЛЬКО ТЕ ПОЛЯ, КОТОРЫЕ ЕСТЬ В ДАННЫХ
    cust_schema = StructType([
        StructField("customer_id", StringType()),       # ID клиента
        StructField("registration_date", DateType()),   # Дата регистрации
        StructField("region", StringType())             # Регион
        
    ])
    
    # Загрузка сырых данных из JSON
    df_customers = spark.read.schema(cust_schema).json("s3a://bucket-ml/raw/customers/")
    
    original_count = df_customers.count()
    print(f"Исходное количество клиентов: {original_count}")
    
    # Очистка данных - только доступные поля
    df_clean = df_customers.dropna(subset=["customer_id", "registration_date", "region"])
    
    # ФИЛЬТРАЦИЯ:
    # 1. Убираем пустые идентификаторы клиентов
    df_clean = df_clean.filter(length(trim(col("customer_id"))) > 0)
    
    # 2. Убираем пустые регионы
    df_clean = df_clean.filter(length(trim(col("region"))) > 0)
    
    # ФИЛЬТРАЦИЯ ПО ДАТЕ РЕГИСТРАЦИИ:
    # Клиенты не могут быть зарегистрированы более 10 лет назад
    min_date = datetime.now() - timedelta(days=365*10)
    df_clean = df_clean.filter(col("registration_date") >= lit(min_date))
    df_clean = df_clean.filter(col("registration_date") <= current_date())
    
    # УДАЛЕНИЕ ДУБЛИКАТОВ КЛИЕНТОВ:
    # У каждого клиента должна быть только одна запись
    df_clean = df_clean.dropDuplicates(["customer_id"])
    
    # СТАТИСТИКА ОЧИСТКИ:
    cleaned_count = df_clean.count()
    removed_count = original_count - cleaned_count
    
    print(f"После очистки: {cleaned_count} клиентов")
    print(f"Удалено записей: {removed_count} ({removed_count/original_count*100:.2f}%)")
    
    # АНАЛИЗ РАСПРЕДЕЛЕНИЯ ПО РЕГИОНАМ:
    if cleaned_count > 0:
        print("Распределение по регионам:")
        df_clean.groupBy("region").count().orderBy("count", ascending=False).show()
    
    # СОХРАНЕНИЕ РЕЗУЛЬТАТА:
    df_clean.write.mode("overwrite").parquet("s3a://bucket-ml/processed/cleaned/customers/")
    
    return df_clean

def validate_data_quality(spark, df_tx, df_customers):
    """
    ВАЛИДАЦИЯ КАЧЕСТВА ДАННЫХ
    
    Проверяет:
    - Согласованность данных между транзакциями и клиентами
    - Временные диапазоны
    - Распределение данных
    
    Параметры:
    spark - SparkSession
    df_tx - DataFrame с транзакциями
    df_customers - DataFrame с клиентами
    """
    print("Валидация качества данных...")
    
    # ПРОВЕРКА НА ПУСТЫЕ ДАТАСЕТЫ:
    if df_tx.count() == 0 or df_customers.count() == 0:
        print("⚠️ Один из датасетов пуст, пропускаем валидацию")
        return
    
    # ПРОВЕРКА СОГЛАСОВАННОСТИ ДАННЫХ:
    
    # Уникальные клиенты в транзакциях
    tx_customers = df_tx.select("customer_id").distinct()
    # Все клиенты из справочника
    all_customers = df_customers.select("customer_id")
    
    # Находим расхождения:
    # Клиенты, которые есть в справочнике, но нет транзакций
    missing_in_tx = all_customers.subtract(tx_customers)
    # Клиенты с транзакциями, но без профиля в справочнике
    missing_in_customers = tx_customers.subtract(all_customers)
    
    print(f"Клиенты без транзакций: {missing_in_tx.count()}")
    print(f"Клиенты без профиля: {missing_in_customers.count()}")
    
    # АНАЛИЗ ВРЕМЕННОГО ДИАПАЗОНА ТРАНЗАКЦИЙ:
    date_stats = df_tx.agg(
        min("tx_date").alias("first_tx_date"),  # Первая транзакция
        max("tx_date").alias("last_tx_date")    # Последняя транзакция
    ).collect()[0]  # collect() возвращает список строк, берем первую
    
    print(f"Диапазон транзакций: {date_stats['first_tx_date']} - {date_stats['last_tx_date']}")
    
    # АНАЛИЗ РАСПРЕДЕЛЕНИЯ ТРАНЗАКЦИЙ ПО ДНЯМ:
    print("Транзакции по дням (топ-10):")
    df_tx.groupBy("tx_date").count().orderBy("count", ascending=False).show(10)

def main():
    """
    ОСНОВНАЯ ФУНКЦИЯ СКРИПТА
    
    Оркестрирует весь процесс очистки данных:
    1. Создание Spark сессии
    2. Запуск очистки транзакций
    3. Запуск очистки данных клиентов  
    4. Валидация качества
    5. Обработка ошибок
    """
    # СОЗДАНИЕ SPARK СЕССИИ:
    spark = SparkSession.builder \
        .appName("Data_Cleaning_Adapted") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()                                # Создает или возвращает существующую
    
    # НАСТРОЙКА УРОВНЯ ЛОГИРОВАНИЯ:
    spark.sparkContext.setLogLevel("WARN")  # Показывать только предупреждения и ошибки
    
    print("=== ЗАПУСК АДАПТИРОВАННОЙ ОЧИСТКИ ДАННЫХ ===")
    
    try:
        # ОСНОВНОЙ ПРОЦЕСС ОЧИСТКИ:
        
        # 1. Очистка данных транзакций
        df_tx_clean = clean_transactions_data(spark)
        
        # 2. Очистка данных клиентов
        df_cust_clean = clean_customers_data(spark)
        
        # 3. Валидация качества данных
        validate_data_quality(spark, df_tx_clean, df_cust_clean)
        
        print("✅ Очистка данных завершена успешно!")
        
    except Exception as e:
        # ОБРАБОТКА ОШИБОК:
        print(f"❌ Ошибка при очистке данных: {e}")
        import traceback
        traceback.print_exc()  # Подробный вывод стека вызовов
        raise  # Повторно вызываем исключение для завершения скрипта
    
    finally:
        # ЗАВЕРШЕНИЕ SPARK СЕССИИ:
        # Важно всегда закрывать сессию для освобождения ресурсов
        spark.stop()

# ТОЧКА ВХОДА: скрипт выполняется только при прямом запуске
if __name__ == "__main__":
    main()