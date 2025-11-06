#!/usr/bin/env python3
"""
Скрипт для обучения ML-модели предсказания оттока - ИСПРАВЛЕННАЯ ВЕРСИЯ С СОХРАНЕНИЕМ
ПОЛНОСТЬЮ ПРОКОММЕНТИРОВАННАЯ ВЕРСИЯ ДЛЯ ОБУЧЕНИЯ СТУДЕНТОВ
"""

# =============================================================================
# ИМПОРТ БИБЛИОТЕК И НАСТРОЙКА ОКРУЖЕНИЯ
# =============================================================================

import subprocess
import sys
import os
import tempfile
import base64
from datetime import datetime

def install_and_import_packages():
    """
    ФУНКЦИЯ ДЛЯ АВТОМАТИЧЕСКОЙ УСТАНОВКИ И ИМПОРТА НЕОБХОДИМЫХ ПАКЕТОВ
    Это важно для студентов, чтобы они могли запустить код без предварительной настройки окружения
    """
    # Создаем временную директорию для установки пакетов
    temp_dir = tempfile.mkdtemp()
    print(f"Временная директория для пакетов: {temp_dir}")
    
    # Настраиваем переменные окружения для установки пакетов в временную директорию
    env = os.environ.copy()
    env['PYTHONUSERBASE'] = temp_dir
    
    # Словарь необходимых пакетов: имя_пакета: версия_для_pip
    required_packages = {
        'numpy': 'numpy==1.21.0',      # Для численных вычислений
        'matplotlib': 'matplotlib'      # Для визуализации
    }
    
    # Попытка установки каждого пакета
    for package_name, pip_name in required_packages.items():
        try:
            __import__(package_name)  # Пробуем импортировать
            print(f"✓ {package_name} уже установлен")
        except ImportError:
            print(f"Установка {package_name}...")
            try:
                # Устанавливаем через pip в временную директорию
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "--user", "--no-cache-dir", "--no-warn-script-location",
                    pip_name
                ], env=env)
                print(f"✓ {package_name} успешно установлен")
            except subprocess.CalledProcessError as e:
                print(f"✗ Ошибка установки {package_name}: {e}")
    
    # Добавляем пути к временной директории в sys.path для импорта
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

# Вызываем функцию установки пакетов
install_and_import_packages()

# =============================================================================
# ИМПОРТ ОСНОВНЫХ БИБЛИОТЕК ДЛЯ ML И АНАЛИЗА ДАННЫХ
# =============================================================================

try:
    # PySpark - основной фреймворк для распределенной обработки данных
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F  # Функции для работы с DataFrame
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, FloatType, IntegerType
    from pyspark.ml.feature import VectorAssembler, StringIndexer  # Преобразование признаков
    from pyspark.ml.classification import LogisticRegression  # Модель классификации
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    from pyspark.ml import Pipeline  # Для создания ML-пайплайнов
    
except ImportError as e:
    print(f"Ошибка импорта PySpark: {e}")
    sys.exit(1)

try:
    # NumPy - фундаментальная библиотека для научных вычислений
    import numpy as np
    print("✓ NumPy успешно импортирован")
except ImportError:
    print("✗ NumPy не доступен, создаем заглушку")
    # Создаем заглушку если NumPy не установлен
    class MockNumpy:
        def randint(self, max_val):
            import random
            return random.randint(0, max_val-1)
    np = MockNumpy()

try:
    # Matplotlib - библиотека для визуализации данных
    import matplotlib.pyplot as plt
    print("✓ Matplotlib успешно импортирован")
except ImportError:
    print("✗ Matplotlib не доступен, визуализация отключена")
    plt = None

# =============================================================================
# ФУНКЦИЯ ИНТЕРПРЕТАЦИИ МОДЕЛИ (ОБЪЯСНЕНИЕ РЕЗУЛЬТАТОВ)
# =============================================================================

def interpret_model(model, feature_cols):
    """
    Интерпретация модели логистической регрессии
    Показывает важность каждого признака для прогнозирования
    
    Args:
        model: обученная модель логистической регрессии
        feature_cols: список имен признаков
    
    Returns:
        feature_importance: отсортированный список кортежей (признак, важность)
    """
    print("5. Интерпретация модели...")

    # Получаем коэффициенты модели (веса признаков)
    coefficients = model.coefficients
    intercept = model.intercept  # Свободный член

    print(f"Intercept (свободный член): {intercept:.4f}")
    print("Важность признаков (коэффициенты логистической регрессии):")
    
    # Создаем пары (признак, коэффициент)
    feature_importance = list(zip(feature_cols, coefficients))

    # Выводим важность каждого признака
    for f, c in feature_importance:
        print(f"  {f}: {float(c):.6f}")

    # Преобразуем и сортируем по абсолютной важности (по убыванию)
    feature_importance = [(str(f), float(c)) for f, c in feature_importance]
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    return feature_importance

# =============================================================================
# ФУНКЦИЯ СОХРАНЕНИЯ РЕЗУЛЬТАТОВ В S3
# =============================================================================

def save_business_artifacts_fixed(spark, df_ml, feature_importance, metrics, business_insights):
    """
    ИСПРАВЛЕННОЕ сохранение бизнес-артефактов в S3
    Сохраняет различные типы результатов для дальнейшего анализа
    
    Args:
        spark: SparkSession
        df_ml: DataFrame с данными
        feature_importance: важность признаков
        metrics: метрики модели
        business_insights: бизнес-инсайты
    """
    print("💼 Сохранение бизнес-артефактов в S3...")
    
    # Создаем путь с текущей датой для организации результатов
    current_date = datetime.now().strftime("%Y-%m-%d")
    base_path = f"s3a://bucket-ml/reports/{current_date}"
    
    try:
        # 1. СОХРАНЕНИЕ ВАЖНОСТИ ПРИЗНАКОВ В CSV
        feature_data = []
        for feat, imp in feature_importance:
            feature_data.append((str(feat), float(imp), float(abs(imp))))
        
        # Создаем схему для DataFrame
        feature_schema = StructType([
            StructField("feature", StringType(), True),
            StructField("coefficient", DoubleType(), True),
            StructField("absolute_importance", DoubleType(), True)
        ])
        
        # Создаем DataFrame и сортируем по важности
        feature_df = spark.createDataFrame(feature_data, schema=feature_schema)
        feature_df = feature_df.orderBy("absolute_importance", ascending=False)
        
        # Сохраняем в CSV
        feature_df.write \
            .mode("overwrite") \
            .option("header", "true") \
            .option("delimiter", ";") \
            .csv(f"{base_path}/feature_importance/")
        
        print("   ✅ Важность признаков сохранена в CSV")
        
        # 2. СОХРАНЕНИЕ БИЗНЕС-ИНСАЙТОВ
        insights_text = f"""Анализ оттока клиентов - Отчет от {current_date}

{business_insights}

---
Сгенерировано автоматически ML-пайплайном
"""
        # Используем DataFrame для сохранения текста
        insights_df = spark.createDataFrame([(insights_text,)], ["content"])
        insights_df.coalesce(1).write \
            .mode("overwrite") \
            .text(f"{base_path}/business_insights/")
        
        print("   ✅ Бизнес-инсайты сохранены")
        
        # 3. СОХРАНЕНИЕ МЕТРИК В ФОРМАТЕ JSON
        churned_count = df_ml.filter(F.col("is_churned") == 1).count()
        total_count = df_ml.count()
        churn_rate = float(churned_count / total_count) if total_count > 0 else 0.0
        
        metrics_summary = f"""{{
    "report_date": "{current_date}",
    "model_metrics": {{
        "auc_score": {float(metrics['auc'])},
        "f1_score": {float(metrics['f1'])}
    }},
    "business_metrics": {{
        "total_customers_analyzed": {total_count},
        "churned_customers": {churned_count},
        "churn_rate": {churn_rate:.4f}
    }}
}}"""
        
        # Сохраняем метрики как текст
        metrics_df = spark.createDataFrame([(metrics_summary,)], ["content"])
        metrics_df.coalesce(1).write \
            .mode("overwrite") \
            .text(f"{base_path}/metrics_summary/")
        
        print("   ✅ Метрики сохранены в JSON")
        
        # 4. СОХРАНЕНИЕ ПРИМЕРА ДАННЫХ ДЛЯ АНАЛИЗА
        sample_data = df_ml.select(
            "customer_id", "is_churned", "avg_tx_amount", 
            "total_tx_count", "days_since_last_tx", "region"
        ).limit(1000)
        
        sample_data.write \
            .mode("overwrite") \
            .option("header", "true") \
            .option("delimiter", ";") \
            .csv(f"{base_path}/sample_data/")
        
        print("   ✅ Пример данных сохранен")
        
        return base_path
        
    except Exception as e:
        print(f"   ❌ Ошибка при сохранении артефактов: {e}")
        return base_path

# =============================================================================
# ФУНКЦИЯ ГЕНЕРАЦИИ HTML ОТЧЕТА С РУССКИМИ НАЗВАНИЯМИ ПРИЗНАКОВ
# =============================================================================

def generate_html_report(spark, feature_importance, metrics, business_insights, save_path):
    """
    Генерация красивого HTML отчета с визуализациями
    ВАЖНО: Заменяем технические названия признаков на понятные русские названия
    """
    print("📊 Генерация HTML отчета...")
    
    if plt is None:
        print("   ⚠️ Matplotlib не доступен, HTML отчет не будет сгенерирован")
        return
    
    try:
        # СЛОВАРЬ ДЛЯ ПРЕОБРАЗОВАНИЯ ТЕХНИЧЕСКИХ НАЗВАНИЙ В РУССКИЕ
        feature_name_mapping = {
            "tx_frequency": "Частота транзакций",
            "days_since_last_tx": "Дней с последней транзакции", 
            "total_tx_count": "Общее количество транзакций",
            "region_index": "Регион (индекс)",
            "customer_lifetime_days": "Время жизни клиента (дни)",
            "avg_tx_amount": "Средняя сумма транзакции",
            "std_tx_amount": "Стандартное отклонение суммы транзакции"
        }
        
        # Преобразуем названия признаков на русский
        feature_importance_russian = []
        for feature_name, importance_value in feature_importance:
            russian_name = feature_name_mapping.get(feature_name, feature_name)
            feature_importance_russian.append((russian_name, importance_value))
        
        # СОЗДАЕМ ГРАФИК ВАЖНОСТИ ПРИЗНАКОВ
        plt.figure(figsize=(12, 8))
        
        # Берем топ-8 признаков для визуализации
        features = [x[0] for x in feature_importance_russian[:8]]
        importance = [x[1] for x in feature_importance_russian[:8]]
        
        # Разные цвета для положительных и отрицательных коэффициентов
        colors = ['#ff6b6b' if x > 0 else '#4ecdc4' for x in importance]
        
        # Создаем горизонтальные столбцы
        bars = plt.barh(features, importance, color=colors, alpha=0.8)
        plt.xlabel('Важность признака (коэффициент)')
        plt.title('Топ-8 самых важных признаков для прогнозирования оттока')
        plt.grid(axis='x', alpha=0.3)
        
        # Добавляем подписи значений на график
        for bar, value in zip(bars, importance):
            plt.text(bar.get_width() + (0.01 if value > 0 else -0.03), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{value:.4f}', 
                    ha='left' if value > 0 else 'right', 
                    va='center', 
                    fontsize=9)
        
        # Добавляем легенду
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='#ff6b6b', alpha=0.7, label='Увеличивает вероятность оттока')
        blue_patch = mpatches.Patch(color='#4ecdc4', alpha=0.7, label='Уменьшает вероятность оттока')
        plt.legend(handles=[red_patch, blue_patch])
        
        plt.tight_layout()
        
        # Сохраняем график временно и конвертируем в base64 для HTML
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            chart_path = tmp_file.name
        
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Читаем график как base64 для встраивания в HTML
        with open(chart_path, "rb") as img_file:
            chart_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Удаляем временный файл
        os.unlink(chart_path)
        
        # ГЕНЕРАЦИЯ HTML КОДА
        html_content = f'''
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Анализ оттока клиентов - {datetime.now().strftime("%Y-%m-%d")}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f6fa; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
                .card {{ background: white; padding: 25px; margin: 15px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .metric-box {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .insight {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; }}
                .feature-positive {{ color: #e74c3c; font-weight: bold; }}
                .feature-negative {{ color: #27ae60; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Анализ оттока клиентов</h1>
                    <p>Автоматический отчет сгенерирован: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
                </div>
                
                <div class="card">
                    <h2>Метрики качества модели</h2>
                    <div class="metrics">
                        <div class="metric-box">
                            <div class="metric-value">{metrics['auc']:.3f}</div>
                            <div>AUC Score</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{metrics['f1']:.3f}</div>
                            <div>F1 Score</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Важность признаков</h2>
                    <img src="data:image/png;base64,{chart_base64}" alt="Feature Importance" style="max-width: 100%; border: 1px solid #ddd; border-radius: 5px;">
                    
                    <h3>Детализация топ-признаков:</h3>
                    <ul>
        {"".join([f'<li><span class="{"feature-positive" if imp > 0 else "feature-negative"}">{feat}</span>: {imp:.4f}</li>' 
                 for feat, imp in feature_importance_russian[:5]])}
                    </ul>
                </div>
                
                <div class="card">
                    <h2>Ключевые инсайты и рекомендации</h2>
                    <div class="insight">
                        {business_insights.replace(chr(10), '<br>').replace('===', '<h3>').replace('===', '</h3>')}
                    </div>
                </div>
                
                <div class="card">
                    <h2>📋 Как использовать этот отчет</h2>
                    <p><strong>Для бизнес-пользователей:</strong> Откройте CSV файлы в Excel для детального анализа</p>
                    <p><strong>Для аналитиков:</strong> Используйте Parquet файлы для глубокого анализа в Python/Jupyter</p>
                    <p><strong>Для дашбордов:</strong> Данные готовы для загрузки в Tableau/Power BI</p>
                </div>
            </div>
        </body>
        </html>
        '''
        
        # Сохраняем HTML в S3
        html_rdd = spark.sparkContext.parallelize([html_content])
        html_rdd.coalesce(1).saveAsTextFile(save_path)
        
        print("   ✅ HTML отчет сохранен")
        
    except Exception as e:
        print(f"   ⚠️ Ошибка при генерации HTML отчета: {e}")

# =============================================================================
# ФУНКЦИЯ ГЕНЕРАЦИИ БИЗНЕС-ИНСАЙТОВ
# =============================================================================

def generate_business_insights(feature_importance, metrics):
    """
    Генерация бизнес-инсайтов на основе модели
    Преобразует технические результаты в понятные бизнес-рекомендации
    """
    insights = []
    insights.append("=== БИЗНЕС-ИНСАЙТЫ И РЕКОМЕНДАЦИИ ===")
    insights.append(f"Качество модели: AUC = {metrics['auc']:.3f}, F1 = {metrics['f1']:.3f}")
    insights.append("")
    insights.append("=== КЛЮЧЕВЫЕ ФАКТОРЫ ОТТОКА ===")
    
    # СЛОВАРЬ ДЛЯ ПРЕОБРАЗОВАНИЯ НАЗВАНИЙ ПРИЗНАКОВ
    feature_name_mapping = {
        "tx_frequency": "Частота транзакций",
        "days_since_last_tx": "Дней с последней транзакции", 
        "total_tx_count": "Общее количество транзакций",
        "region_index": "Регион",
        "customer_lifetime_days": "Время жизни клиента",
        "avg_tx_amount": "Средняя сумма транзакции",
        "std_tx_amount": "Стандартное отклонение суммы транзакции"
    }
    
    # Анализируем топ-3 самых важных признаков
    top_features = feature_importance[:3]
    
    for feature, coef in top_features:
        # Используем русское название если есть в словаре
        feature_display = feature_name_mapping.get(feature, feature)
        
        if coef > 0:
            insights.append(f"{feature_display}: УВЕЛИЧЕНИЕ этого показателя УВЕЛИЧИВАЕТ вероятность оттока")
        else:
            insights.append(f"{feature_display}: УВЕЛИЧЕНИЕ этого показателя УМЕНЬШАЕТ вероятность оттока")
    
    insights.append("")
    insights.append("=== РЕКОМЕНДАЦИИ ДЛЯ БИЗНЕСА ===")
    
    # Генерируем рекомендации на основе важности признаков
    for feature, coef in top_features:
        feature_lower = str(feature).lower()
        
        if "days_since_last_tx" in feature_lower and coef > 0:
            insights.append("Разработать программу реактивации для неактивных клиентов")
        elif "tx_frequency" in feature_lower and coef < 0:
            insights.append("Стимулировать регулярность покупок через программы лояльности")
        elif "avg_tx_amount" in feature_lower and coef < 0:
            insights.append("Разработать стратегию увеличения среднего чека")
        elif "customer_lifetime_days" in feature_lower and coef < 0:
            insights.append("Ценить долгосрочных клиентов - программа признания за лояльность")
        elif "total_tx_count" in feature_lower and coef < 0:
            insights.append("Увеличивать количество транзакций через персональные предложения")
    
    insights.append("")
    insights.append("=== ПРИОРИТЕТНЫЕ ДЕЙСТВИЯ ===")
    insights.append("1. Запустить программу удержания для клиентов с высоким риском оттока")
    insights.append("2. Сегментировать клиентов по ключевым факторам риска") 
    insights.append("3. Мониторить ключевые метрики в реальном времени")
    
    return "\n".join(insights)

# =============================================================================
# ФУНКЦИЯ ПОДГОТОВКИ ПРИЗНАКОВ ДЛЯ ML
# =============================================================================

def prepare_features_adapted(df):
    """
    Подготовка признаков для ML-модели
    Включает кодирование категориальных переменных и создание вектора признаков
    
    Args:
        df: исходный DataFrame
    
    Returns:
        feature_pipeline: пайплайн для преобразования признаков
        existing_features: список используемых признаков
    """
    # Базовые числовые признаки
    base_features = [
        "avg_tx_amount",           # Средняя сумма транзакции
        "total_tx_count",          # Общее количество транзакций
        "days_since_last_tx",      # Дней с последней транзакции
        "customer_lifetime_days",  # Время жизни клиента в днях
        "tx_frequency",            # Частота транзакций
        "std_tx_amount"            # Стандартное отклонение суммы транзакции
    ]
    
    # Оставляем только существующие в DataFrame признаки
    existing_features = [c for c in base_features if c in df.columns]
    print(f"Используемые признаки: {existing_features}")
    
    stages = []  # Этапы пайплайна
    
    # Если есть категориальный признак "region", кодируем его
    if "region" in df.columns:
        region_indexer = StringIndexer(inputCol="region", outputCol="region_index")
        stages.append(region_indexer)
        existing_features.append("region_index")
    
    # Создаем вектор признаков для ML-модели
    assembler = VectorAssembler(
        inputCols=existing_features,
        outputCol="features",      # Выходной столбец с вектором признаков
        handleInvalid="skip"       # Пропускать строки с невалидными значениями
    )
    stages.append(assembler)
    
    # Создаем пайплайн преобразования признаков
    feature_pipeline = Pipeline(stages=stages)
    return feature_pipeline, existing_features

# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ ML-ПАЙПЛАЙНА
# =============================================================================

def main():
    """
    ОСНОВНАЯ ФУНКЦИЯ ML-ПАЙПЛАЙНА С ИСПРАВЛЕННЫМ СОХРАНЕНИЕМ
    Полный цикл ML: загрузка данных, подготовка, обучение, оценка, сохранение результатов
    """
    
    # Инициализация SparkSession - точка входа для работы со Spark
    spark = SparkSession.builder \
        .appName("Churn_ML_Fixed") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")  # Уменьшаем уровень логгирования
    print("=== Запуск ML-пайплайна с исправленным сохранением ===")
    
    try:
        # ЭТАП 1: ЗАГРУЗКА ДАННЫХ
        print("1. Загрузка данных для ML...")
        df = spark.read.parquet("s3a://bucket-ml/processed/churn_features/")
        
        # Фильтрация данных - убираем строки с пропущенными значениями
        df_ml = df.filter(
            F.col("avg_tx_amount").isNotNull() &
            F.col("total_tx_count").isNotNull() &
            F.col("days_since_last_tx").isNotNull() &
            F.col("customer_lifetime_days").isNotNull()
        )
        
        print(f"Загружено {df_ml.count()} записей для обучения")
        
        # ЭТАП 2: РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
        df_train, df_test = df_ml.randomSplit([0.8, 0.2], seed=42)
        print(f"Размер train: {df_train.count()}, test: {df_test.count()}")
        
        # ЭТАП 3: ПОДГОТОВКА ПРИЗНАКОВ И ОБУЧЕНИЕ МОДЕЛИ
        feature_pipeline, feature_cols = prepare_features_adapted(df_train)
        feature_pipeline_model = feature_pipeline.fit(df_train)
        
        # Преобразуем данные с помощью обученного пайплайна
        df_train_processed = feature_pipeline_model.transform(df_train)
        df_test_processed = feature_pipeline_model.transform(df_test)
        
        print("4. Обучение логистической регрессии...")
        # Создаем модель логистической регрессии
        lr = LogisticRegression(
            featuresCol="features",     # Столбец с признаками
            labelCol="is_churned",      # Целевая переменная
            maxIter=50,                 # Максимальное количество итераций
            regParam=0.01               # Параметр регуляризации
        )
        
        # Обучаем модель на тренировочных данных
        lr_model = lr.fit(df_train_processed)
        
        # Делаем предсказания на тестовых данных
        predictions = lr_model.transform(df_test_processed)
        
        # ЭТАП 4: ОЦЕНКА КАЧЕСТВА МОДЕЛИ
        # AUC (Area Under Curve) - площадь под ROC кривой
        evaluator_auc = BinaryClassificationEvaluator(labelCol="is_churned")
        auc = evaluator_auc.evaluate(predictions)
        
        # F1-score - гармоническое среднее точности и полноты
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="is_churned",
            predictionCol="prediction", 
            metricName="f1"
        )
        f1 = evaluator_f1.evaluate(predictions)
        
        print(f"✅ Модель обучена. AUC: {auc:.4f}, F1: {f1:.4f}")
        
        # ЭТАП 5: ИНТЕРПРЕТАЦИЯ МОДЕЛИ
        feature_importance = interpret_model(lr_model, feature_cols)
        
        # ЭТАП 6: ГЕНЕРАЦИЯ И СОХРАНЕНИЕ БИЗНЕС-АРТЕФАКТОВ
        print("6. Генерация и сохранение бизнес-артефактов...")
        
        metrics_dict = {"auc": auc, "f1": f1}
        business_insights = generate_business_insights(feature_importance, metrics_dict)
        
        # Сохраняем артефакты в S3
        base_path = save_business_artifacts_fixed(spark, df_ml, feature_importance, metrics_dict, business_insights)
        
        # Генерируем и сохраняем HTML отчет
        generate_html_report(spark, feature_importance, metrics_dict, business_insights, f"{base_path}/html_report/")
        
        # ЭТАП 7: СОХРАНЕНИЕ МОДЕЛИ И МЕТРИК
        print("7. Сохранение модели и метрик...")
        
        # Сохраняем обученную модель
        lr_model.write().overwrite().save("s3a://bucket-ml/models/churn_model_fixed/")
        
        # Сохраняем метрики модели
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics_data = [{
            'model': 'logistic_regression_fixed',
            'auc': float(auc),
            'f1': float(f1), 
            'timestamp': current_time,
            'features_count': len(feature_cols)
        }]
        
        # Схема для метрик
        metrics_schema = StructType([
            StructField("model", StringType(), True),
            StructField("auc", FloatType(), True),
            StructField("f1", FloatType(), True),
            StructField("timestamp", StringType(), True),
            StructField("features_count", IntegerType(), True)
        ])
        
        # Создаем DataFrame с метриками и сохраняем
        metrics_df = spark.createDataFrame(metrics_data, schema=metrics_schema)
        metrics_df.write.mode("append").json("s3a://bucket-ml/models/model_metrics/")
        
        print("ML-пайплайн успешно завершён!")
        print(f"Все артефакты сохранены в: {base_path}")
        print(f"Топ-3 важных признака:")
        for i, (feat, imp) in enumerate(feature_importance[:3], 1):
            print(f"   {i}. {feat}: {imp:.6f}")
        
    except Exception as e:
        print(f"❌ Ошибка в ML-пайплайне: {e}")
        import traceback
        traceback.print_exc()  # Подробный вывод ошибки для отладки
    finally:
        spark.stop()  # Всегда останавливаем Spark сессию

# Точка входа в программу
if __name__ == "__main__":
    main()