#!/usr/bin/env python3
"""
Скрипт для обучения ML-модели предсказания оттока - ИСПРАВЛЕННАЯ ВЕРСИЯ
"""

import subprocess
import sys
import os
import tempfile
from datetime import datetime  # Добавляем импорт

def install_and_import_packages():
    """Установка и импорт необходимых пакетов"""
    # Создаем временную директорию для установки пакетов
    temp_dir = tempfile.mkdtemp()
    print(f"Временная директория для пакетов: {temp_dir}")
    
    # Устанавливаем переменные окружения для pip
    env = os.environ.copy()
    env['PYTHONUSERBASE'] = temp_dir
    
    required_packages = {
        'numpy': 'numpy==1.21.0',
        'matplotlib': 'matplotlib'
    }
    
    # Устанавливаем пакеты
    for package_name, pip_name in required_packages.items():
        try:
            __import__(package_name)
            print(f"✓ {package_name} уже установлен")
        except ImportError:
            print(f"Установка {package_name}...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "--user", "--no-cache-dir", "--no-warn-script-location",
                    pip_name
                ], env=env)
                print(f"✓ {package_name} успешно установлен")
            except subprocess.CalledProcessError as e:
                print(f"✗ Ошибка установки {package_name}: {e}")
    
    # Добавляем пути к установленным пакетам в sys.path
    possible_paths = [
        os.path.join(temp_dir, 'lib', 'python3.10', 'site-packages'),
        os.path.join(temp_dir, 'lib', 'python3.9', 'site-packages'),
        os.path.join(temp_dir, 'lib', 'python3.8', 'site-packages'),
        os.path.join(temp_dir, 'lib', 'python3.7', 'site-packages'),
        os.path.join(temp_dir, 'lib', 'python', 'site-packages'),
        os.path.join(temp_dir, 'site-packages'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            print(f"Добавлен путь: {path}")

# Установка и настройка путей перед импортом
install_and_import_packages()

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import *
    from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
    from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml import Pipeline
except ImportError as e:
    print(f"Ошибка импорта PySpark: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✓ NumPy успешно импортирован")
except ImportError:
    print("✗ NumPy не доступен, создаем заглушку")
    class MockNumpy:
        def randint(self, max_val):
            import random
            return random.randint(0, max_val-1)
    np = MockNumpy()

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib успешно импортирован")
except ImportError:
    print("✗ Matplotlib не доступен, визуализация отключена")
    plt = None

def prepare_features_adapted(df):
    """Подготовка признаков для ML - АДАПТИРОВАННАЯ ВЕРСИЯ"""
    print("Подготовка признаков...")

    # Используем только существующие признаки
    base_features = [
        "avg_tx_amount", "total_tx_count", "days_since_last_tx",
        "customer_lifetime_days", "tx_frequency", "std_tx_amount"
    ]
    
    # Проверяем какие признаки существуют
    existing_features = [col for col in base_features if col in df.columns]
    print(f"Используемые признаки: {existing_features}")
    
    stages = []
    
    # Обработка региона если он есть
    if "region" in df.columns:
        region_indexer = StringIndexer(inputCol="region", outputCol="region_index")
        stages.append(region_indexer)
        existing_features.append("region_index")

    # Векторизация
    assembler = VectorAssembler(
        inputCols=existing_features,
        outputCol="features",
        handleInvalid="skip"
    )
    stages.append(assembler)
    
    feature_pipeline = Pipeline(stages=stages)
    return feature_pipeline, existing_features

def main():
    """Основная функция"""
    spark = SparkSession.builder \
        .appName("Churn_ML_Adapted") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print("=== Запуск АДАПТИРОВАННОГО ML-пайплайна ===")

    try:
        # 1. Загрузка данных
        print("Загрузка данных для ML...")
        df = spark.read.parquet("s3a://bucket-ml/processed/churn_features/")
        
        # Базовая фильтрация
        df_ml = df.filter(
            col("avg_tx_amount").isNotNull() &
            col("total_tx_count").isNotNull() &
            col("days_since_last_tx").isNotNull() &
            col("customer_lifetime_days").isNotNull()
        )
        
        print(f"Загружено {df_ml.count()} записей для обучения")
        print(f"Распределение целевой переменной:")
        df_ml.groupBy("is_churned").count().show()
        
        if df_ml.count() == 0:
            print("❌ Нет данных для обучения")
            spark.stop()
            return

        # 2. Разделение на train/test
        df_train, df_test = df_ml.randomSplit([0.8, 0.2], seed=42)
        print(f"Размер train: {df_train.count()}, test: {df_test.count()}")

        # 3. Подготовка признаков
        feature_pipeline, feature_cols = prepare_features_adapted(df_train)
        feature_pipeline_model = feature_pipeline.fit(df_train)
        df_train_processed = feature_pipeline_model.transform(df_train)
        df_test_processed = feature_pipeline_model.transform(df_test)

        # 4. Обучение простой модели
        print("Обучение логистической регрессии...")
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="is_churned",
            maxIter=50,
            regParam=0.01
        )
        
        lr_model = lr.fit(df_train_processed)
        predictions = lr_model.transform(df_test_processed)
        
        # 5. Оценка модели
        evaluator_auc = BinaryClassificationEvaluator(labelCol="is_churned")
        auc = evaluator_auc.evaluate(predictions)
        
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="is_churned",
            predictionCol="prediction",
            metricName="f1"
        )
        f1 = evaluator_f1.evaluate(predictions)
        
        print(f"✅ Модель обучена. AUC: {auc:.4f}, F1: {f1:.4f}")
        
        # 6. Сохранение модели
        lr_model.write().overwrite().save("s3a://bucket-ml/models/churn_model_adapted/")
        
        # ИСПРАВЛЕННЫЙ БЛОК: Сохранение метрик
        # Используем Python datetime вместо current_timestamp()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        metrics_data = [{
            'model': 'logistic_regression_adapted',
            'auc': float(auc),
            'f1': float(f1),
            'timestamp': current_time  # Используем строковое представление времени
        }]
        
        # Создаем схему для DataFrame с метриками
        from pyspark.sql.types import StructType, StructField, StringType, FloatType
        
        metrics_schema = StructType([
            StructField("model", StringType(), True),
            StructField("auc", FloatType(), True),
            StructField("f1", FloatType(), True),
            StructField("timestamp", StringType(), True)
        ])
        
        metrics_df = spark.createDataFrame(metrics_data, schema=metrics_schema)
        metrics_df.write.mode("append").json("s3a://bucket-ml/models/model_metrics/")
        
        print("✅ ML-пайплайн успешно завершён!")

    except Exception as e:
        print(f"❌ Ошибка в ML-пайплайне: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        spark.stop()

if __name__ == "__main__":
    main()