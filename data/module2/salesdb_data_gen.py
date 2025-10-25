from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql import Row
import random

# Создаем Spark сессию с поддержкой Hive
spark = SparkSession.builder \
    .appName("generate_sales_data") \
    .config("spark.sql.warehouse.dir", "s3a://bucket-ml/warehouse/") \
    .enableHiveSupport() \
    .getOrCreate()

# Параметры генерации
TOTAL_ROWS = 1000
NULL_PERCENTAGE = 0.1  # 10% NULL в колонке amount
NEGATIVE_PERCENTAGE = 0.05  # 5% отрицательных значений

# Схема таблицы
schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("product", StringType(), True),
    StructField("amount", DoubleType(), True)
])

# Генерируем данные
data = []
products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']

for i in range(1, TOTAL_ROWS + 1):
    product = products[i % len(products)]
    
    # Генерация значений amount с заданным распределением
    rand_val = random.random()
    if rand_val < NULL_PERCENTAGE:
        amount = None  # NULL значение
    elif rand_val < NULL_PERCENTAGE + NEGATIVE_PERCENTAGE:
        amount = round(random.uniform(-100, -1), 2)  # Отрицательные значения
    else:
        amount = round(random.uniform(0, 1000), 2)  # Положительные значения
    
    data.append(Row(id=i, product=product, amount=amount))

# Создаем DataFrame
df = spark.createDataFrame(data, schema)

# Создаем базу данных если не существует
spark.sql("CREATE DATABASE IF NOT EXISTS sales_db")

# Сохраняем как Hive таблицу в S3 (формат, который понимает Trino)
df.write \
    .mode("overwrite") \
    .option("path", "s3a://bucket-ml/tables/sales") \
    .saveAsTable("sales_db.sales")

# Проверяем что создалось
print("Таблица создана успешно!")
print("Схема таблицы:")
spark.sql("DESCRIBE FORMATTED sales_db.sales").show(truncate=False)

print("Первые 10 строк:")
spark.sql("SELECT * FROM sales_db.sales LIMIT 10").show()

print("Статистика по NULL и отрицательным значениям:")
spark.sql("""
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) as null_amounts,
        SUM(CASE WHEN amount < 0 THEN 1 ELSE 0 END) as negative_amounts,
        ROUND(SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as null_percentage,
        ROUND(SUM(CASE WHEN amount < 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as negative_percentage
    FROM sales_db.sales
""").show()

spark.stop()