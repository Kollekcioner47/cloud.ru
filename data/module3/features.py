"""
Модуль для генерации признаков и предобработки данных
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def create_composite_risk_feature(df):
    """
    Создает целевую переменную 'композитный риск' на основе поведения клиентов
    
    Args:
        df (pd.DataFrame): Исходный DataFrame с данными клиентов
    
    Returns:
        pd.DataFrame: DataFrame с добавленными признаками риска
    """
    print("Создание метрик риска оттока...")
    
    # Создаем отдельные метрики риска
    df['days_risk'] = (df['days_since_last_transaction'] > 
                       df['days_since_last_transaction'].median()).astype(int)
    
    df['spending_risk'] = (df['total_spent'] < 
                           df['total_spent'].median()).astype(int)
    
    df['activity_risk'] = (df['transactions_last_90d'] < 
                           df['transactions_last_90d'].median()).astype(int)
    
    # Композитный риск (хотя бы по двум метрикам из трех)
    df['composite_risk'] = ((df['days_risk'] + 
                            df['spending_risk'] + 
                            df['activity_risk']) >= 2).astype(int)
    
    print(f"Распределение композитного риска: {df['composite_risk'].value_counts().to_dict()}")
    
    return df


def prepare_ml_features(df, target_column='composite_risk', features_list=None):
    """
    Подготавливает признаки и целевую переменную для ML модели
    
    Args:
        df (pd.DataFrame): Исходный DataFrame
        target_column (str): Название целевой колонки
        features_list (list): Список признаков для использования
    
    Returns:
        tuple: (X, y) - признаки и целевая переменная
    """
    if features_list is None:
        features_list = ['total_spent', 'days_since_last_transaction', 
                        'transactions_last_90d', 'total_transactions',
                        'avg_transaction_amount']
    
    # Выбираем только существующие колонки
    available_features = [f for f in features_list if f in df.columns]
    
    X = df[available_features].copy()
    y = df[target_column]
    
    print(f"Подготовлено {X.shape[1]} признаков и {len(y)} целевых значений")
    print(f"Признаки: {available_features}")
    
    return X, y


def balance_data_with_smote(X, y, random_state=42):
    """
    Балансирует данные с помощью SMOTE
    
    Args:
        X (pd.DataFrame): Признаки
        y (pd.Series): Целевая переменная
        random_state (int): Random state для воспроизводимости
    
    Returns:
        tuple: (X_balanced, y_balanced) - сбалансированные данные
    """
    try:
        # Проверяем, что есть минимум 2 класса
        if len(y.unique()) < 2:
            print("В данных только один класс. SMOTE не может быть применен.")
            return X, y
        
        smote = SMOTE(random_state=random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"⚖️  Данные сбалансированы: {pd.Series(y_balanced).value_counts().to_dict()}")
        
        return X_balanced, y_balanced
    
    except Exception as e:
        print(f"Ошибка при балансировке данных: {e}")
        return X, y


def create_customer_segments(df, n_clusters=3, features_list=None):
    """
    Создает сегменты клиентов с помощью кластеризации K-means
    
    Args:
        df (pd.DataFrame): Исходный DataFrame
        n_clusters (int): Количество кластеров
        features_list (list): Список признаков для кластеризации
    
    Returns:
        pd.DataFrame: DataFrame с добавленной колонкой кластеров
    """
    if features_list is None:
        features_list = ['total_transactions', 'total_spent', 
                        'days_since_last_transaction', 'transactions_last_90d']
    
    print(f"Создание {n_clusters} сегментов клиентов...")
    
    # Выбираем только существующие колонки
    available_features = [f for f in features_list if f in df.columns]
    X_cluster = df[available_features].fillna(0)
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Кластеризация
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df['cluster'] = clusters
    df['cluster'] = df['cluster'].astype(str)  # Для удобства визуализации
    
    print(f"✅ Создано {n_clusters} кластеров: {pd.Series(clusters).value_counts().to_dict()}")
    
    return df, kmeans, scaler


def detect_anomalies(df, contamination=0.1, features_list=None):
    """
    Обнаружение аномалий с помощью Isolation Forest
    
    Args:
        df (pd.DataFrame): Исходный DataFrame
        contamination (float): Доля ожидаемых аномалий
        features_list (list): Список признаков для анализа
    
    Returns:
        pd.DataFrame: DataFrame с добавленными колонками аномалий
    """
    if features_list is None:
        features_list = ['total_transactions', 'total_spent', 
                        'days_since_last_transaction', 'transactions_last_90d']
    
    print("Поиск аномальных клиентов...")
    
    # Выбираем только существующие колонки
    available_features = [f for f in features_list if f in df.columns]
    X_anomaly = df[available_features].fillna(0)
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_anomaly)
    
    # Обнаружение аномалий
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_scores = iso_forest.fit_predict(X_scaled)
    
    df['anomaly_score'] = anomaly_scores
    df['is_anomaly'] = (anomaly_scores == -1).astype(int)
    
    anomaly_count = df['is_anomaly'].sum()
    anomaly_percent = (anomaly_count / len(df)) * 100
    
    print(f"✅ Выявлено аномальных клиентов: {anomaly_count} ({anomaly_percent:.1f}%)")
    
    return df, iso_forest, scaler


def analyze_clusters(df):
    """
    Анализирует и возвращает статистику по кластерам
    
    Args:
        df (pd.DataFrame): DataFrame с колонкой 'cluster'
    
    Returns:
        pd.DataFrame: Статистика по кластерам
    """
    if 'cluster' not in df.columns:
        print("❌ Колонка 'cluster' не найдена")
        return None
    
    cluster_analysis = df.groupby('cluster').agg({
        'total_transactions': ['mean', 'std'],
        'total_spent': ['mean', 'std'],
        'days_since_last_transaction': ['mean', 'std'],
        'transactions_last_90d': ['mean', 'std'],
        'customer_id': 'count'
    }).round(2)
    
    # Упрощаем названия колонок
    cluster_analysis.columns = ['_'.join(col).strip() for col in cluster_analysis.columns.values]
    cluster_analysis = cluster_analysis.rename(columns={'customer_id_count': 'customer_count'})
    
    print("Анализ кластеров завершен")
    return cluster_analysis


def analyze_anomalies(df):
    """
    Анализирует и возвращает статистику по аномалиям
    
    Args:
        df (pd.DataFrame): DataFrame с колонкой 'is_anomaly'
    
    Returns:
        pd.DataFrame: Статистика по аномалиям
    """
    if 'is_anomaly' not in df.columns:
        print("❌ Колонка 'is_anomaly' не найдена")
        return None
    
    anomaly_analysis = df.groupby('is_anomaly').agg({
        'total_transactions': 'mean',
        'total_spent': 'mean',
        'days_since_last_transaction': 'mean',
        'customer_id': 'count'
    }).rename(columns={'customer_id': 'count'})
    
    print("Анализ аномалий завершен")
    return anomaly_analysis


def get_feature_importance_report(model, feature_names):
    """
    Создает отчет о важности признаков
    
    Args:
        model: Обученная ML модель
        feature_names (list): Список названий признаков
    
    Returns:
        pd.DataFrame: DataFrame с важностью признаков
    """
    try:
        # Для моделей с feature_importances_ (RandomForest, XGBoost и т.д.)
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        # Для линейных моделей с coefficients_
        elif hasattr(model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': model.coef_[0],
                'abs_importance': np.abs(model.coef_[0])
            }).sort_values('abs_importance', ascending=False)
            
        else:
            print("Модель не поддерживает стандартные методы анализа важности признаков")
            return None
        
        print("Отчет о важности признаков создан")
        return importance_df
        
    except Exception as e:
        print(f"❌ Ошибка при создании отчета о важности признаков: {e}")
        return None


# Пример использования модуля
if __name__ == "__main__":
    # Создаем тестовые данные для демонстрации
    np.random.seed(42)
    test_data = pd.DataFrame({
        'customer_id': range(1000),
        'total_transactions': np.random.randint(1, 50, 1000),
        'total_spent': np.random.uniform(100, 50000, 1000),
        'days_since_last_transaction': np.random.randint(1, 1000, 1000),
        'transactions_last_90d': np.random.randint(0, 10, 1000),
        'avg_transaction_amount': np.random.uniform(10, 5000, 1000)
    })
    
    print("Тестирование модуля features.py...")
    
    # Тестируем создание композитного риска
    df_with_risk = create_composite_risk_feature(test_data)
    print(f"✅ Добавлены колонки: {[col for col in df_with_risk.columns if 'risk' in col]}")
    
    # Тестируем подготовку признаков
    X, y = prepare_ml_features(df_with_risk)
    print(f"✅ Размерность X: {X.shape}, y: {y.shape}")
    
    # Тестируем балансировку
    if len(y.unique()) > 1:
        X_balanced, y_balanced = balance_data_with_smote(X, y)
        print(f"✅ Сбалансированные данные: X{X_balanced.shape}, y{y_balanced.shape}")
    
    # Тестируем кластеризацию
    df_clustered, kmeans, scaler = create_customer_segments(df_with_risk, n_clusters=3)
    cluster_stats = analyze_clusters(df_clustered)
    print("Статистика кластеров:")
    print(cluster_stats)
    
    # Тестируем обнаружение аномалий
    df_anomalies, iso_forest, anomaly_scaler = detect_anomalies(df_with_risk)
    anomaly_stats = analyze_anomalies(df_anomalies)
    print("Статистика аномалий:")
    print(anomaly_stats)
    
    print("Все функции features.py работают корректно!")