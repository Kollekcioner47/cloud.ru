"""
Основной скрипт обучения модели прогнозирования оттока клиентов
с комплексным трекингом экспериментов через MLflow
"""

import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.sklearn
import joblib
import os
import sys
import warnings
import json
import shutil
import hashlib
import platform
import subprocess
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                            classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# Добавляем путь для импорта наших модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import get_trino_connection, load_churn_prediction_data
from src.features import (create_composite_risk_feature, prepare_ml_features,
                         balance_data_with_smote, get_feature_importance_report)

warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Класс для обучения и оценки ML моделей с комплексным трекингом экспериментов
    """

    def __init__(self, config_path="configs/config.yaml"):
        """
        Инициализация тренера с конфигурацией

        Args:
            config_path (str): Путь к файлу конфигурации
        """
        self.config = self.load_config(config_path)
        self.model = None
        self.metrics = {}
        self.feature_names = []
        self.current_run_id = None
        self.experiment_name = self.config['mlflow']['experiment_name']

    def load_config(self, config_path):
        """
        Загрузка конфигурации из YAML файла

        Args:
            config_path (str): Путь к конфигурационному файлу

        Returns:
            dict: Конфигурация проекта
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Конфигурация загружена из {config_path}")
            print(f"📄 Имя эксперимента в конфиге: {config['mlflow']['experiment_name']}")
            return config
        except Exception as e:
            print(f"❌ Ошибка загрузки конфигурации: {e}")
            raise

    def setup_mlflow(self):
        """Настройка MLflow эксперимента"""
        if not self.config['mlflow'].get('enabled', False):
            return

        try:
            # Устанавливаем tracking URI для подключения к MLflow серверу
            tracking_uri = "http://127.0.0.1:48399"
            mlflow.set_tracking_uri(tracking_uri)
            print(f"🔗 Подключение к MLflow: {tracking_uri}")
            
            # Получаем список всех экспериментов для отладки
            try:
                experiments = mlflow.search_experiments()
                print("📋 Доступные эксперименты:")
                for exp in experiments:
                    print(f"   - {exp.name} (ID: {exp.experiment_id})")
            except Exception as e:
                print(f"⚠️ Не удалось получить список экспериментов: {e}")
                print("🔄 Пытаемся продолжить...")
            
            # Явно создаем или получаем эксперимент
            experiment_name = self.experiment_name
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                    print(f"✅ Создан новый эксперимент: {experiment_name} (ID: {experiment_id})")
                else:
                    experiment_id = experiment.experiment_id
                    print(f"✅ Используем существующий эксперимент: {experiment_name} (ID: {experiment_id})")
            except Exception as e:
                print(f"⚠️ Ошибка при работе с экспериментом: {e}")
                # Пробуем создать заново
                try:
                    experiment_id = mlflow.create_experiment(experiment_name)
                    print(f"✅ Создан эксперимент: {experiment_name} (ID: {experiment_id})")
                except Exception as create_error:
                    print(f"❌ Не удалось создать эксперимент: {create_error}")
                    return

            # Устанавливаем эксперимент
            mlflow.set_experiment(experiment_name)
            
            # Выводим финальную информацию о настройке
            current_tracking_uri = mlflow.get_tracking_uri()
            print(f"🎯 MLflow настроен:")
            print(f"   - Tracking URI: {current_tracking_uri}")
            print(f"   - Эксперимент: {experiment_name}")
            
        except Exception as e:
            print(f"⚠️ Ошибка настройки MLflow: {e}")
            print("💡 Проверьте, что MLflow сервер запущен на порту 48399")

    def load_data(self):
        """
        Загрузка и подготовка данных

        Returns:
            tuple: (X, y) - признаки и целевая переменная
        """
        print("Загрузка данных...")
        try:
            # Подключаемся к Trino и загружаем данные
            conn = get_trino_connection(self.config['data']['ca_cert_path'])
            df = load_churn_prediction_data(
                conn,
                limit=self.config['data'].get('limit', 5000)
            )
            conn.close()

            # Создаем целевую переменную и признаки
            df = create_composite_risk_feature(df)
            X, y = prepare_ml_features(
                df,
                target_column=self.config['features']['target_column'],
                features_list=self.config['features'].get('feature_list')
            )

            self.feature_names = X.columns.tolist()
            print(f"✅ Данные загружены: {X.shape[0]} samples, {X.shape[1]} features")
            
            return X, y

        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            raise

    def prepare_data(self, X, y):
        """
        Подготовка данных для обучения

        Args:
            X (pd.DataFrame): Признаки
            y (pd.Series): Целевая переменная

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("Подготовка данных...")

        # Балансировка данных если требуется
        if self.config['training'].get('balance_data', True):
            X, y = balance_data_with_smote(X, y, self.config['training']['random_state'])

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=y
        )

        print(f"✅ Данные подготовлены:")
        print(f" - Train: {X_train.shape[0]} samples")
        print(f" - Test: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test

    def initialize_model(self):
        """
        Инициализация модели на основе конфигурации

        Returns:
            model: Инициализированная ML модель
        """
        model_config = self.config['model']
        model_type = model_config['type']
        print(f"Инициализация модели: {model_type}")

        if model_type == 'LogisticRegression':
            model = LogisticRegression(**model_config.get('params', {}))
        elif model_type == 'RandomForestClassifier':
            model = RandomForestClassifier(**model_config.get('params', {}))
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

        return model

    def train_model(self, X_train, y_train):
        """
        Обучение модели

        Args:
            X_train (pd.DataFrame): Обучающие признаки
            y_train (pd.Series): Обучающая целевая переменная

        Returns:
            model: Обученная модель
        """
        print("Обучение модели...")
        self.model = self.initialize_model()
        self.model.fit(X_train, y_train)
        print("✅ Модель обучена")
        return self.model

    def evaluate_model(self, X_test, y_test):
        """
        Оценка качества модели

        Args:
            X_test (pd.DataFrame): Тестовые признаки
            y_test (pd.Series): Тестовая целевая переменная
        """
        print("Оценка модели...")
        if self.model is None:
            raise ValueError("Модель не обучена!")

        # Предсказания
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Расчет метрик
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        # Вывод результатов
        print("Результаты оценки:")
        for metric, value in self.metrics.items():
            print(f" - {metric}: {value:.4f}")

        # Детальный отчет
        print("\nДетальный отчет:")
        print(classification_report(y_test, y_pred))

        return self.metrics

    def log_to_mlflow(self, X_train, X_test, y_train, y_test):
        """
        Комплексное логирование эксперимента в MLflow
        """
        if not self.config['mlflow'].get('enabled', False):
            return

        try:
            with mlflow.start_run(run_name=self._generate_run_name()) as run:
                self.current_run_id = run.info.run_id
                
                print(f"🚀 Начался MLflow запуск: {self.current_run_id}")
                print(f"📝 Experiment ID: {run.info.experiment_id}")
                
                # Логируем параметры модели
                mlflow.log_params(self.config['model']['params'])
                mlflow.log_param('model_type', self.config['model']['type'])
                
                # Логируем параметры данных
                mlflow.log_param('data_limit', self.config['data']['limit'])
                mlflow.log_param('test_size', self.config['training']['test_size'])
                mlflow.log_param('random_state', self.config['training']['random_state'])
                
                # Логируем информацию о данных
                mlflow.log_param('n_features', len(self.feature_names))
                mlflow.log_param('n_train_samples', len(X_train))
                mlflow.log_param('n_test_samples', len(X_test))
                
                # Логируем метрики
                mlflow.log_metrics(self.metrics)
                
                # Логируем модель с input_example чтобы избежать warning
                sample_input = X_test.iloc[:1]
                mlflow.sklearn.log_model(
                    self.model, 
                    "model",
                    registered_model_name="CustomerChurnModel",
                    input_example=sample_input
                )
                
                # Логируем feature importance
                importance_df = get_feature_importance_report(self.model, self.feature_names)
                if importance_df is not None:
                    # Создаем временный файл для feature importance
                    importance_path = "feature_importance.csv"
                    importance_df.to_csv(importance_path, index=False)
                    mlflow.log_artifact(importance_path, "feature_importance")
                    os.remove(importance_path)  # Удаляем временный файл
                
                # Логируем графики
                plots_path = self.create_evaluation_plots(X_test, y_test, save_only=True)
                if plots_path and os.path.exists(plots_path):
                    mlflow.log_artifact(plots_path, "evaluation_plots")
                    # Удаляем временный файл после логирования
                    os.remove(plots_path)
                
                # Логируем конфигурацию
                mlflow.log_artifact("configs/config.yaml", "config")
                
                # Логируем дополнительную информацию
                self.log_environment()
                
                print("✅ Эксперимент полностью записан в MLflow")
                print(f"   Run ID: {self.current_run_id}")
                print(f"   Run Name: {run.info.run_name}")
                print(f"   Experiment: {self.experiment_name}")
                print(f"   🔗 Посмотреть в UI: http://127.0.0.1:48399")

        except Exception as e:
            print(f"⚠️ Ошибка логирования в MLflow: {e}")
            print("💡 Проверьте, что MLflow сервер запущен и доступен")

    def _generate_run_name(self):
        """Генерация читаемого имени для запуска"""
        model_type = self.config['model']['type']
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        return f"{model_type}_{timestamp}"

    def log_environment(self):
        """Логирование информации об окружении"""
        try:
            mlflow.log_param("python_version", platform.python_version())
            mlflow.log_param("sklearn_version", sklearn.__version__)
            mlflow.log_param("pandas_version", pd.__version__)
            mlflow.log_param("numpy_version", np.__version__)
            mlflow.log_param("platform", platform.platform())
            
            print("✅ Информация об окружении записана")
        except Exception as e:
            print(f"⚠️ Ошибка логирования окружения: {e}")

    def create_evaluation_plots(self, X_test, y_test, save_only=False):
        """
        Создание визуализаций для оценки модели

        Args:
            X_test (pd.DataFrame): Тестовые данные
            y_test (pd.Series): Тестовые целевые значения
            save_only (bool): Только сохранить, не показывать

        Returns:
            str: Путь к сохраненному файлу с графиками
        """
        if not self.config['output'].get('create_plots', True):
            return None

        try:
            print("Создание визуализаций...")
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Матрица ошибок
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')

            # 2. Важность признаков
            importance_df = get_feature_importance_report(self.model, self.feature_names)
            if importance_df is not None:
                top_features = importance_df.head(10)
                if 'importance' in top_features.columns:
                    top_features.sort_values('importance', ascending=True).plot(
                        kind='barh', x='feature', y='importance', ax=axes[0, 1]
                    )
                axes[0, 1].set_title('Top 10 Feature Importance')

            # 3. ROC кривая
            if y_pred_proba is not None:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                axes[1, 0].plot(fpr, tpr, label=f'ROC curve (AUC = {self.metrics.get("roc_auc", 0):.3f})')
                axes[1, 0].plot([0, 1], [0, 1], 'k--')
                axes[1, 0].set_xlabel('False Positive Rate')
                axes[1, 0].set_ylabel('True Positive Rate')
                axes[1, 0].set_title('ROC Curve')
                axes[1, 0].legend()

            # 4. Сравнение метрик
            metrics_for_plot = {k: v for k, v in self.metrics.items() if k in ['accuracy', 'f1_score', 'roc_auc']}
            axes[1, 1].bar(metrics_for_plot.keys(), metrics_for_plot.values())
            axes[1, 1].set_title('Model Metrics Comparison')
            axes[1, 1].set_ylabel('Score')
            for i, v in enumerate(metrics_for_plot.values()):
                axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

            plt.tight_layout()

            # Сохранение графиков
            plots_path = "evaluation_plots.png"
            plt.savefig(plots_path, dpi=300, bbox_inches='tight')
            
            if not save_only:
                plt.show()
            else:
                plt.close()

            print(f"✅ Визуализации сохранены: {plots_path}")
            return plots_path

        except Exception as e:
            print(f"⚠️ Ошибка создания визуализаций: {e}")
            return None

    def run_training_pipeline(self):
        """
        Запуск полного пайплайна обучения с комплексным трекингом
        """
        print("Запуск пайплайна обучения...")
        print("=" * 50)

        try:
            # 1. Настройка MLflow
            self.setup_mlflow()

            # 2. Загрузка данных
            X, y = self.load_data()

            # 3. Подготовка данных
            X_train, X_test, y_train, y_test = self.prepare_data(X, y)

            # 4. Обучение модели
            self.train_model(X_train, y_train)

            # 5. Оценка модели
            self.evaluate_model(X_test, y_test)

            # 6. Логирование в MLflow (включая создание графиков)
            self.log_to_mlflow(X_train, X_test, y_train, y_test)

            print("=" * 50)
            print("🎉 Пайплайн обучения успешно завершен!")
            if self.current_run_id:
                print(f"📊 Run ID: {self.current_run_id}")
                print(f"🔍 Посмотреть результаты: http://127.0.0.1:48399")
            
            return self.model, self.metrics

        except Exception as e:
            print(f"❌ Ошибка в пайплайне обучения: {e}")
            raise

def main():
    """
    Основная функция для запуска обучения
    """
    try:
        # Инициализация тренера
        trainer = ModelTrainer("configs/config.yaml")

        # Запуск пайплайна
        model, metrics = trainer.run_training_pipeline()

        # Вывод итоговых результатов
        print("\n" + "=" * 50)
        print("🏆 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        print("=" * 50)
        for metric, value in metrics.items():
            print(f" {metric.upper()}: {value:.4f}")

        return model, metrics

    except Exception as e:
        print(f"💥 Критическая ошибка: {e}")
        return None, None

if __name__ == "__main__":
    main()
