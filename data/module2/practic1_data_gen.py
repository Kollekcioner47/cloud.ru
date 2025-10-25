from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, count, max as spark_max, avg, sum, when, desc
import random
import builtins
from datetime import datetime, timedelta

spark = SparkSession.builder.appName("Data_Generation").getOrCreate()

# Настройки генерации
NUM_CUSTOMERS = 5000
NUM_TRANSACTIONS = 50000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 1, 20)

# Список регионов для реалистичности
REGIONS = ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань", 
           "Нижний Новгород", "Челябинск", "Самара", "Омск", "Ростов-на-Дону"]

# 1. Генерация данных клиентов
customers_data = []

for i in range(1, NUM_CUSTOMERS + 1):
    customer_id = f"CUST_{i:06d}"
    registration_date = START_DATE + timedelta(
        days=random.randint(0, (END_DATE - START_DATE).days - 365)  # регистрация в течение года
    )
    region = random.choice(REGIONS)
    
    customers_data.append({
        "customer_id": customer_id,
        "registration_date": registration_date,
        "region": region
    })

# Создаем DataFrame клиентов
customers_schema = StructType([
    StructField("customer_id", StringType(), True),
    StructField("registration_date", DateType(), True),
    StructField("region", StringType(), True)
])

df_customers = spark.createDataFrame(customers_data, schema=customers_schema)

print("Генерация клиентов завершена:")
df_customers.show(10)
print(f"Всего клиентов: {df_customers.count()}")

# 2. Генерация транзакций
transactions_data = []

# Создаем паттерны транзакций для разных типов клиентов
customer_activity = {}
total_desired_transactions = 0

# Первый проход: вычисляем общее желаемое количество транзакций
for cust in customers_data:
    if random.random() < 0.7:  # 70% активных клиентов
        tx_count = random.randint(5, 30)
        avg_amount = random.uniform(500, 5000)
        last_activity = random.random() < 0.8  # 80% совершали транзакции в последние 30 дней
    else:
        tx_count = random.randint(1, 8)
        avg_amount = random.uniform(300, 2000)
        last_activity = random.random() < 0.2  # 20% совершали транзакции в последние 30 дней
    
    customer_activity[cust["customer_id"]] = {
        "desired_tx_count": tx_count,
        "avg_amount": avg_amount,
        "last_activity": last_activity
    }
    total_desired_transactions += tx_count

# Корректируем количество транзакций для достижения NUM_TRANSACTIONS
scaling_factor = NUM_TRANSACTIONS / total_desired_transactions

# Второй проход: устанавливаем финальное количество транзакций с коррекцией
final_transaction_count = 0
for cust_id, activity in customer_activity.items():
    desired_count = activity["desired_tx_count"]
    # Применяем scaling factor и округляем - используем встроенную функцию round и max
    final_count = int(builtins.max(1, builtins.round(desired_count * scaling_factor)))
    customer_activity[cust_id]["final_tx_count"] = final_count
    final_transaction_count += final_count

# Корректируем возможную погрешность округления
difference = final_transaction_count - NUM_TRANSACTIONS
if difference != 0:
    # Находим клиентов, у которых можно изменить количество транзакций
    adjustable_customers = [cust_id for cust_id, activity in customer_activity.items() 
                           if activity["final_tx_count"] > 1 or difference < 0]
    
    if adjustable_customers:
        adjustment_per_customer = difference // len(adjustable_customers)
        remainder = difference % len(adjustable_customers)
        
        for i, cust_id in enumerate(adjustable_customers):
            adjustment = adjustment_per_customer + (1 if i < remainder else 0)
            new_count = customer_activity[cust_id]["final_tx_count"] - adjustment
            customer_activity[cust_id]["final_tx_count"] = int(builtins.max(1, new_count))

# Генерация транзакций
transaction_id = 1
for customer_id, activity in customer_activity.items():
    tx_count = activity["final_tx_count"]
    avg_amount = activity["avg_amount"]
    
    # Определяем временной диапазон для транзакций
    if activity["last_activity"]:
        end_range = END_DATE
    else:
        end_range = END_DATE - timedelta(days=random.randint(31, 180))
    
    for _ in range(tx_count):
        # Генерируем дату транзакции
        days_offset = random.randint(0, (end_range - START_DATE).days)
        tx_date = START_DATE + timedelta(days=days_offset)
        
        # Генерируем сумму (нормальное распределение вокруг avg_amount)
        tx_amount = builtins.max(50.0, random.normalvariate(avg_amount, avg_amount * 0.3))
        
        transactions_data.append({
            "transaction_id": f"TX_{transaction_id:08d}",
            "customer_id": customer_id,
            "tx_amount": float(round(tx_amount, 2)),  # Явное преобразование к float
            "tx_date": tx_date
        })
        
        transaction_id += 1

# Создаем DataFrame транзакций
transactions_schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("tx_amount", DoubleType(), True),
    StructField("tx_date", DateType(), True)
])

df_transactions = spark.createDataFrame(transactions_data, schema=transactions_schema)

print("\nГенерация транзакций завершена:")
df_transactions.show(10)
print(f"Всего транзакций: {df_transactions.count()}")

# 3. Сохранение в S3-совместимое хранилище
print("\nСохранение данных...")

# Сохраняем транзакции в Parquet (оптимальный формат для аналитики)
df_transactions.write \
    .mode("overwrite") \
    .parquet("s3a://bucket-ml/raw/transactions/")

# Сохраняем клиентов в JSON
df_customers.write \
    .mode("overwrite") \
    .json("s3a://bucket-ml/raw/customers/")

print("Данные успешно сохранены:")
print("- Транзакции: s3a://bucket-ml/raw/transactions/")
print("- Клиенты: s3a://bucket-ml/raw/customers/")

# 4. Проверяем статистику сгенерированных данных
print("\n=== СТАТИСТИКА СГЕНЕРИРОВАННЫХ ДАННЫХ ===")

# Статистика по транзакциям
df_transactions.describe("tx_amount").show()

# Анализ активности клиентов
activity_stats = df_transactions.groupBy("customer_id").agg(
    count("*").alias("tx_count"),
    spark_max("tx_date").alias("last_tx_date"),
    avg("tx_amount").alias("avg_tx_amount")
)

# Определяем отток (последняя транзакция более 30 дней назад)
cutoff_date = END_DATE - timedelta(days=30)
churn_stats = activity_stats.withColumn(
    "is_churned", 
    when(col("last_tx_date") < cutoff_date, 1).otherwise(0)
)

churn_summary = churn_stats.groupBy("is_churned").agg(
    count("*").alias("customer_count"),
    avg("tx_count").alias("avg_transactions"),
    avg("avg_tx_amount").alias("avg_amount")
)

print("\nСтатистика оттока клиентов:")
churn_summary.show()

# Региональная статистика
regional_stats = df_customers.join(activity_stats, "customer_id") \
    .groupBy("region") \
    .agg(
        count("*").alias("customer_count"),
        avg("tx_count").alias("avg_transactions_per_customer"),
        sum("tx_count").alias("total_transactions"),
        avg("avg_tx_amount").alias("avg_tx_amount")
    ).orderBy(desc("customer_count"))

print("\nСтатистика по регионам:")
regional_stats.show()

spark.stop()