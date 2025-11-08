"""
Модуль для загрузки данных из Data Platform через Trino
"""

import trino
import pandas as pd
import os


def get_trino_connection(ca_cert_path=None):
    """
    Создает и возвращает соединение с Trino
    
    Args:
        ca_cert_path (str): Путь к корневому сертификату
    
    Returns:
        trino.dbapi.Connection: Объект соединения с Trino
    """
    try:
        conn = trino.dbapi.connect(
            host="external-trino-1114bee6-c997-443d-80ce-08436240f340.cluster-6048b1a0-4aa2-42aa-a1ea-99c0f7bfd669.dataplatform.cloud.ru",
            port=443,
            user="engineer",
            auth=trino.auth.BasicAuthentication("engineer", "8923b6ac941329de1b9e49d71cfec78d"),
            catalog="trino_catalog_ml",
            schema="analytics",
            verify=ca_cert_path
        )
        print("✅ Соединение с Trino установлено")
        return conn
    except Exception as e:
        print(f"❌ Ошибка подключения к Trino: {e}")
        raise


def load_customer_data(connection, limit=1000):
    """
    Загружает данные о клиентах из витрины customer_360
    
    Args:
        connection: Соединение с Trino
        limit (int): Ограничение количества записей
    
    Returns:
        pd.DataFrame: DataFrame с данными клиентов
    """
    try:
        cursor = connection.cursor()
        query = f"SELECT * FROM customer_360 LIMIT {limit}"
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Получаем названия колонок
        column_names = [desc[0] for desc in cursor.description]
        
        # Создаем DataFrame
        customer_df = pd.DataFrame(rows, columns=column_names)
        print(f"✅ Загружено {len(customer_df)} записей из витрины customer_360")
        
        return customer_df
    except Exception as e:
        print(f"❌ Ошибка загрузки данных клиентов: {e}")
        raise


def load_churn_prediction_data(connection, limit=5000):
    """
    Загружает данные для прогнозирования оттока
    
    Args:
        connection: Соединение с Trino  
        limit (int): Ограничение количества записей
    
    Returns:
        pd.DataFrame: DataFrame с данными для ML
    """
    try:
        cursor = connection.cursor()
        query = f"""
        SELECT * FROM churn_prediction_dataset 
        WHERE churn_label IS NOT NULL 
        LIMIT {limit}
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        column_names = [desc[0] for desc in cursor.description]
        ml_df = pd.DataFrame(rows, columns=column_names)
        
        print(f"✅ Загружено {len(ml_df)} записей для ML-модели")
        print(f"📊 Размерность данных: {ml_df.shape}")
        print(f"⚖️ Баланс классов оттока: {ml_df['churn_label'].value_counts().to_dict()}")
        
        return ml_df
    except Exception as e:
        print(f"❌ Ошибка загрузки данных для ML: {e}")
        raise


def load_transaction_data(connection, customer_limit=100):
    """
    Загружает агрегированную статистику по транзакциям
    
    Args:
        connection: Соединение с Trino
        customer_limit (int): Ограничение количества клиентов
    
    Returns:
        pd.DataFrame: DataFrame со статистикой транзакций
    """
    try:
        cursor = connection.cursor()
        query = f"""
        SELECT 
            customer_id, 
            COUNT(*) as tx_count, 
            AVG(tx_amount) as avg_amount,
            SUM(tx_amount) as total_amount,
            MIN(tx_date) as first_transaction,
            MAX(tx_date) as last_transaction
        FROM transactions 
        GROUP BY customer_id
        LIMIT {customer_limit}
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        column_names = [desc[0] for desc in cursor.description]
        tx_df = pd.DataFrame(rows, columns=column_names)
        
        print(f"✅ Загружена статистика по {len(tx_df)} клиентам")
        return tx_df
    except Exception as e:
        print(f"❌ Ошибка загрузки данных транзакций: {e}")
        raise


def get_available_tables(connection):
    """
    Возвращает список доступных таблиц в схеме
    
    Args:
        connection: Соединение с Trino
    
    Returns:
        list: Список названий таблиц
    """
    try:
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES FROM trino_catalog_ml.analytics")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        print("📋 Доступные таблицы в схеме analytics:")
        for table in table_names:
            print(f"   - {table}")
            
        return table_names
    except Exception as e:
        print(f"❌ Ошибка получения списка таблиц: {e}")
        return []


# Пример использования модуля
if __name__ == "__main__":
    # Тестирование функций
    ca_cert_path = "/home/jovyan/dp-cert.crt"
    
    try:
        # Подключение к Trino
        conn = get_trino_connection(ca_cert_path)
        
        # Получение списка таблиц
        tables = get_available_tables(conn)
        
        # Загрузка данных клиентов
        customer_df = load_customer_data(conn, limit=100)
        print(f"Структура данных клиентов: {customer_df.shape}")
        
        # Загрузка ML данных
        ml_df = load_churn_prediction_data(conn, limit=100)
        print(f"Структура ML данных: {ml_df.shape}")
        
        # Закрытие соединения
        conn.close()
        print("🔌 Соединение закрыто")
        
    except Exception as e:
        print(f"💥 Ошибка при тестировании модуля: {e}")