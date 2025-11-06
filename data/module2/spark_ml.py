#!/usr/bin/env python3
"""
Скрипт для обучения ML-модели предсказания оттока - ИСПРАВЛЕННАЯ ВЕРСИЯ

НАЗНАЧЕНИЕ СКРИПТА:
- Подготовка данных для машинного обучения
- Обучение модели классификации для предсказания оттока клиентов
- Оценка качества модели и сохранение результатов
- Управление зависимостями в Spark-окружении

ОСНОВНЫЕ ЭТАПЫ:
1. Установка и настройка зависимостей
2. Загрузка и подготовка данных
3. Инжиниринг признаков (Feature Engineering)
4. Обучение модели
5. Оценка и сохранение результатов
"""

import subprocess
import sys
import os
import tempfile
from datetime import datetime  # Добавляем импорт для работы с датами

def install_and_import_packages():
    """
    УСТАНОВКА И ИМПОРТ НЕОБХОДИМЫХ ПАКЕТОВ
    
    Особенность Spark-окружений:
    - Часто отсутствуют стандартные Python-пакеты
    - Нужно устанавливать пакеты во время выполнения
    - Используем временные директории для изоляции
    """
    # Создаем временную директорию для установки пакетов
    temp_dir = tempfile.mkdtemp()
    print(f"Временная директория для пакетов: {temp_dir}")
    
    # Устанавливаем переменные окружения для pip
    env = os.environ.copy()
    env['PYTHONUSERBASE'] = temp_dir  # Указываем куда устанавливать
    
    # Словарь необходимых пакетов
    required_packages = {
        'numpy': 'numpy==1.21.0',      # Для численных операций
        'matplotlib': 'matplotlib'      # Для визуализации
    }
    
    # Проверяем и устанавливаем пакеты
    for package_name, pip_name in required_packages.items():
        try:
            __import__(package_name)  # Пробуем импортировать
            print(f"✓ {package_name} уже установлен")
        except ImportError:
            print(f"Установка {package_name}...")
            try:
                # Устанавливаем через subprocess
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "--user", "--no-cache-dir", "--no-warn-script-location",
                    pip_name
                ], env=env)
                print(f"✓ {package_name} успешно установлен")
            except subprocess.CalledProcessError as e:
                print(f"✗ Ошибка установки {package_name}: {e}")
    
    # Добавляем пути к установленным пакетам в sys.path
    # Это нужно чтобы Python нашел только что установленные пакеты
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
            sys.path.insert(0, path)  # Добавляем в начало пути поиска
            print(f"Добавлен путь: {path}")

# Установка и настройка путей перед импортом PySpark
# Это критически важно - сначала установить пакеты, потом импортировать
install_and_import_packages()

# ИМПОРТ БИБЛИОТЕК PySpark
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
    sys.exit(1)  # Завершаем скрипт если PySpark не доступен

# ИМПОРТ СТАНДАРТНЫХ PYTHON-БИБЛИОТЕК С ОБРАБОТКОЙ ОШИБОК
try:
    import numpy as np
    print("✓ NumPy успешно импортирован")
except ImportError:
    print("✗ NumPy не доступен, создаем заглушку")
    # Создаем mock-объект если NumPy не установлен
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
    plt = None  # Устанавливаем None если библиотека не доступна

def prepare_features_adapted(df):
    """
    ПОДГОТОВКА ПРИЗНАКОВ ДЛЯ ML - АДАПТИРОВАННАЯ ВЕРСИЯ
    
    В Spark ML все признаки должны быть представлены в виде одного вектора
    Эта функция создает pipeline для преобразования данных
    
    Параметры:
    df - DataFrame с исходными данными
    
    Возвращает:
    pipeline - подготовленный пайплайн преобразований
    feature_cols - список использованных признаков
    """
    print("Подготовка признаков...")

    # БАЗОВЫЕ ПРИЗНАКИ - используем только существующие
    base_features = [
        "avg_tx_amount",           # Средний чек
        "total_tx_count",          # Общее количество транзакций
        "days_since_last_tx",      # Дней с последней транзакции
        "customer_lifetime_days",  # Время жизни клиента
        "tx_frequency",            # Частота транзакций
        "std_tx_amount"            # Стандартное отклонение сумм
    ]
    
    # Проверяем какие признаки реально существуют в данных
    existing_features = [col for col in base_features if col in df.columns]
    print(f"Используемые признаки: {existing_features}")
    
    stages = []  # Список этапов преобразований
    
    # ОБРАБОТКА КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
    if "region" in df.columns:
        # StringIndexer преобразует строки в числовые индексы
        region_indexer = StringIndexer(inputCol="region", outputCol="region_index")
        stages.append(region_indexer)
        existing_features.append("region_index")  # Добавляем к признакам

    # ВЕКТОРИЗАЦИЯ ПРИЗНАКОВ
    # VectorAssembler объединяет все признаки в один вектор
    assembler = VectorAssembler(
        inputCols=existing_features,  # Какие колонки объединять
        outputCol="features",         # Имя результирующей колонки
        handleInvalid="skip"          # Что делать с некорректными значениями
    )
    stages.append(assembler)
    
    # Создаем пайплайн из всех этапов
    feature_pipeline = Pipeline(stages=stages)
    return feature_pipeline, existing_features

def main():
    """ОСНОВНАЯ ФУНКЦИЯ ML-ПАЙПЛАЙНА"""
    # СОЗДАНИЕ SPARK СЕССИИ С ОПТИМИЗАЦИЯМИ
    spark = SparkSession.builder \
        .appName("Churn_ML_Adapted") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")  # Только важные сообщения

    print("=== Запуск АДАПТИРОВАННОГО ML-пайплайна ===")

    try:
        # ЭТАП 1: ЗАГРУЗКА ДАННЫХ
        print("Загрузка данных для ML...")
        df = spark.read.parquet("s3a://bucket-ml/processed/churn_features/")
        
        # БАЗОВАЯ ФИЛЬТРАЦИЯ - удаляем записи с пропущенными значениями
        df_ml = df.filter(
            col("avg_tx_amount").isNotNull() &
            col("total_tx_count").isNotNull() &
            col("days_since_last_tx").isNotNull() &
            col("customer_lifetime_days").isNotNull()
        )
        
        print(f"Загружено {df_ml.count()} записей для обучения")
        
        # АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
        print(f"Распределение целевой переменной:")
        df_ml.groupBy("is_churned").count().show()
        
        # ПРОВЕРКА НАЛИЧИЯ ДАННЫХ
        if df_ml.count() == 0:
            print("❌ Нет данных для обучения")
            spark.stop()
            return

        # ЭТАП 2: РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
        df_train, df_test = df_ml.randomSplit([0.8, 0.2], seed=42)
        print(f"Размер train: {df_train.count()}, test: {df_test.count()}")

        # ЭТАП 3: ПОДГОТОВКА ПРИЗНАКОВ
        feature_pipeline, feature_cols = prepare_features_adapted(df_train)
        
        # ОБУЧАЕМ ПАЙПЛАЙН ПРЕОБРАЗОВАНИЙ на тренировочных данных
        feature_pipeline_model = feature_pipeline.fit(df_train)
        
        # ПРИМЕНЯЕМ ПРЕОБРАЗОВАНИЯ к train и test данным
        df_train_processed = feature_pipeline_model.transform(df_train)
        df_test_processed = feature_pipeline_model.transform(df_test)

        # ЭТАП 4: ОБУЧЕНИЕ МОДЕЛИ
        print("Обучение логистической регрессии...")
        
        # СОЗДАЕМ МОДЕЛЬ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ
        lr = LogisticRegression(
            featuresCol="features",    # Колонка с признаками
            labelCol="is_churned",     # Целевая переменная
            maxIter=50,                # Максимальное количество итераций
            regParam=0.01              # Параметр регуляризации
        )
        
        # ОБУЧАЕМ МОДЕЛЬ на подготовленных данных
        lr_model = lr.fit(df_train_processed)
        
        # ДЕЛАЕМ ПРЕДСКАЗАНИЯ на тестовых данных
        predictions = lr_model.transform(df_test_processed)
        
        # ЭТАП 5: ОЦЕНКА КАЧЕСТВА МОДЕЛИ
        # BinaryClassificationEvaluator для AUC (Area Under Curve)
        evaluator_auc = BinaryClassificationEvaluator(labelCol="is_churned")
        auc = evaluator_auc.evaluate(predictions)
        
        # MulticlassClassificationEvaluator для F1-меры
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="is_churned",
            predictionCol="prediction", 
            metricName="f1"  # F1-score - гармоническое среднее precision и recall
        )
        f1 = evaluator_f1.evaluate(predictions)
        
        print(f"✅ Модель обучена. AUC: {auc:.4f}, F1: {f1:.4f}")
        
        # ЭТАП 6: СОХРАНЕНИЕ МОДЕЛИ И МЕТРИК
        # Сохраняем обученную модель
        lr_model.write().overwrite().save("s3a://bucket-ml/models/churn_model_adapted/")
        
        # ИСПРАВЛЕННЫЙ БЛОК: Сохранение метрик
        # Используем Python datetime вместо Spark-функций
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Подготавливаем данные метрик
        metrics_data = [{
            'model': 'logistic_regression_adapted',
            'auc': float(auc),    # Преобразуем в стандартный float
            'f1': float(f1),
            'timestamp': current_time  # Строковое представление времени
        }]
        
        # СОЗДАЕМ СХЕМУ ДЛЯ DATAFRAME С МЕТРИКАМИ
        from pyspark.sql.types import StructType, StructField, StringType, FloatType
        
        metrics_schema = StructType([
            StructField("model", StringType(), True),
            StructField("auc", FloatType(), True),
            StructField("f1", FloatType(), True),
            StructField("timestamp", StringType(), True)
        ])
        
        # СОЗДАЕМ DATAFRAME И СОХРАНЯЕМ МЕТРИКИ
        metrics_df = spark.createDataFrame(metrics_data, schema=metrics_schema)
        metrics_df.write.mode("append").json("s3a://bucket-ml/models/model_metrics/")
        
        print("✅ ML-пайплайн успешно завершён!")

    except Exception as e:
        # ОБРАБОТКА ОШИБОК С ПОДРОБНЫМ ВЫВОДОМ
        print(f"❌ Ошибка в ML-пайплайне: {e}")
        import traceback
        traceback.print_exc()  # Печатаем стек вызовов
        raise  # Повторно поднимаем исключение

    finally:
        # ВСЕГДА ЗАКРЫВАЕМ SPARK СЕССИЮ
        spark.stop()

# ТОЧКА ВХОДА ПРИ ПРЯМОМ ЗАПУСКЕ СКРИПТА
if __name__ == "__main__":
    main()