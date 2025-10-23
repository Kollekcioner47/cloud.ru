from pyspark.sql import SparkSession
from pyspark.sql.functions import count, sum as Fsum, avg, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

def main():
    try:
        # Создаем SparkSession
        spark = SparkSession.builder \
            .appName("sales_analysis") \
            .getOrCreate()

        bucket_name = 'bucket-ml'
        base_path = f"s3a://{bucket_name}/spark_source"

        # Схемы данных
        orders_schema = StructType([
            StructField("order_id", StringType(), True),
            StructField("client_id", StringType(), True),
            StructField("order_amount", DoubleType(), True),
            StructField("order_date", StringType(), True),
            StructField("city", StringType(), True),
            StructField("loyalty_level", StringType(), True)
        ])

        clients_schema = StructType([
            StructField("client_id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("city", StringType(), True),
            StructField("loyalty_level", StringType(), True)
        ])

        # Читаем данные
        print("=== Чтение данных ===")
        df_clients = spark.read.option("header", "true").schema(clients_schema).csv(f"{base_path}/clients.csv")
        df_orders = spark.read.option("header", "true").schema(orders_schema).csv(f"{base_path}/orders.csv")

        # Проверяем данные
        print("=== Клиенты ===")
        df_clients.printSchema()
        df_clients.show(3, truncate=False)
        
        print("=== Заказы ===")
        df_orders.printSchema()
        df_orders.show(3, truncate=False)

        # Переименовываем столбцы перед JOIN чтобы избежать конфликтов
        df_clients_renamed = df_clients \
            .withColumnRenamed("city", "client_city") \
            .withColumnRenamed("loyalty_level", "client_loyalty")

        # JOIN с переименованными столбцами
        df_joined = df_orders.join(df_clients_renamed, "client_id", "inner")
        
        print("=== Объединенные данные ===")
        df_joined.printSchema()
        df_joined.show(5, truncate=False)

        # Кэшируем
        df_joined.cache()

        # Анализ по городам (используем город из заказов)
        city_analysis = df_joined.groupBy("city").agg(
            count("order_id").alias("total_orders"),
            Fsum(col("order_amount")).alias("total_revenue"),
            avg(col("order_amount")).alias("avg_order_value")
        ).orderBy(col("total_revenue").desc())

        print("=== Анализ по городам (из заказов) ===")
        city_analysis.show(20, truncate=False)

        # Анализ по уровням лояльности (используем лояльность из клиентов)
        loyalty_analysis = df_joined.groupBy("client_loyalty").agg(
            count("order_id").alias("total_orders"),
            Fsum(col("order_amount")).alias("total_revenue"),
            avg(col("order_amount")).alias("avg_order_value")
        ).orderBy(col("total_revenue").desc())

        print("=== Анализ по уровням лояльности (из клиентов) ===")
        loyalty_analysis.show(20, truncate=False)

        # Сохраняем результаты
        output_path = f"s3a://{bucket_name}/spark_output_parquet"
        
        print("=== Сохранение результатов ===")
        city_analysis.write.mode("overwrite").parquet(f"{output_path}/city_analysis")
        loyalty_analysis.write.mode("overwrite").parquet(f"{output_path}/loyalty_analysis")

        print("=== Успешно завершено! ===")

    except Exception as e:
        print(f"=== ОШИБКА: {str(e)} ===")
        import traceback
        traceback.print_exc()
        
    finally:
        # Очистка ресурсов
        try:
            df_joined.unpersist()
        except:
            pass
        spark.stop()
        print("Spark сессия остановлена")

if __name__ == "__main__":
    main()