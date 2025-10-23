from pyspark.sql import SparkSession
from pyspark.sql.functions import count, sum, avg, col

def main():
    # Создаем Spark сессию
    spark = SparkSession.builder \
        .appName("sales_analysis") \
        .getOrCreate()
    
    # Читаем данные
    bucket_name = 'bucket-ml'
    
    df_clients = spark.read \
        .option("header", "true") \
        .csv(f"s3a://{bucket_name}/spark_source/clients.csv")

    df_orders = spark.read \
        .option("header", "true") \
        .csv(f"s3a://{bucket_name}/spark_source/orders.csv")

    # Показываем структуру данных
    print("=== Клиенты ===")
    df_clients.show(5)
    
    print("=== Заказы ===")
    df_orders.show(5)

    # Объединяем таблицы
    df_joined = df_orders.join(df_clients, "client_id", "inner")
    
    # Анализ по городам
    city_analysis = df_joined.groupBy("city") \
        .agg(
            count("order_id").alias("total_orders"),
            sum("order_amount").alias("total_revenue"),
            avg("order_amount").alias("avg_order_value")
        ) \
        .orderBy(col("total_revenue").desc())
    
    print("=== Анализ по городам ===")
    city_analysis.show()

    # Анализ по уровням лояльности
    loyalty_analysis = df_joined.groupBy("loyalty_level") \
        .agg(
            count("order_id").alias("total_orders"),
            sum("order_amount").alias("total_revenue"),
            avg("order_amount").alias("avg_order_value")
        ) \
        .orderBy(col("total_revenue").desc())
    
    print("=== Анализ по уровням лояльности ===")
    loyalty_analysis.show()

    # Сохраняем результаты
    output_path = f"s3a://{bucket_name}/spark_output"
    
    city_analysis.write \
        .option("header", "true") \
        .csv(f"{output_path}/city_analysis")
        
    loyalty_analysis.write \
        .option("header", "true") \
        .csv(f"{output_path}/loyalty_analysis")

    print("=== Анализ завершен ===")
    spark.stop()

if __name__ == "__main__":
    main()