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

# Добавляем путь для импорта наших модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import get_trino_connection, load_churn_prediction_data
from src.features import (create_composite_risk_feature, prepare_ml_features,
                         balance_data_with_smote, get_feature_importance_report)

# Попытка импорта PyTorch для чекпоинтинга
try:
    import torch
except ImportError:
    torch = None

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
            return config
        except Exception as e:
            print(f"❌ Ошибка загрузки конфигурации: {e}")
            raise

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
            
            # Логируем версию данных
            if self.config['mlflow'].get('enabled', False):
                self.log_data_version(df)
                
            return X, y

        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            raise

    def log_data_version(self, df):
        """
        Логирование версии данных для воспроизводимости

        Args:
            df (pd.DataFrame): DataFrame с данными
        """
        try:
            # Создаем хэш от данных
            data_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
            
            # Логируем в MLflow
            mlflow.log_param("data_source", "trino")
            mlflow.log_param("data_shape", str(df.shape))
            mlflow.log_param("data_hash", data_hash)
            mlflow.log_param("data_timestamp", pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
            mlflow.log_param("data_limit", self.config['data'].get('limit', 5000))
            
            print(f"✅ Версия данных зафиксирована: {data_hash}")
        except Exception as e:
            print(f"⚠️ Ошибка логирования версии данных: {e}")

    def log_environment(self):
        """
        Логирование информации об окружении
        """
        try:
            mlflow.log_param("python_version", platform.python_version())
            mlflow.log_param("sklearn_version", sklearn.__version__)
            mlflow.log_param("pandas_version", pd.__version__)
            mlflow.log_param("numpy_version", np.__version__)
            mlflow.log_param("platform", platform.platform())
            
            # Сохраняем полные requirements
            with open("requirements.txt", "w") as f:
                # Здесь может быть команда pip freeze
                pass
            mlflow.log_artifact("requirements.txt")
            
            print("✅ Информация об окружении записана")
        except Exception as e:
            print(f"⚠️ Ошибка логирования окружения: {e}")

    def log_git_info(self):
        """
        Логирование информации о версии кода
        """
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
            git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('utf-8').strip()
            
            mlflow.log_param("git_commit", git_commit)
            mlflow.log_param("git_branch", git_branch)
            mlflow.log_param("code_version", git_commit[:8])
            
            print(f"✅ Информация о Git записана: {git_branch}@{git_commit[:8]}")
        except Exception as e:
            print(f"⚠️ Не удалось получить информацию из Git: {e}")

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

        # Кросс-валидация
        if self.config['training'].get('cross_validation', False):
            cv_scores = cross_val_score(
                self.model, X_test, y_test,
                cv=self.config['training'].get('cv_folds', 5),
                scoring='f1'
            )
            cv_metric = cv_scores.mean()
            cv_std = cv_scores.std() * 2
            self.metrics['cv_f1_mean'] = cv_metric
            self.metrics['cv_f1_std'] = cv_std
            print(f"Кросс-валидация F1: {cv_metric:.4f} (+/- {cv_std:.4f})")

        return self.metrics

    def save_checkpoint(self, model, metrics, epoch, checkpoint_dir="checkpoints"):
        """
        Сохранение чекпоинта обучения с поддержкой разных типов моделей

        Args:
            model: Модель для сохранения
            metrics (dict): Метрики на текущей эпохе
            epoch (int): Номер эпохи
            checkpoint_dir (str): Директория для чекпоинтов
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': pd.Timestamp.now()
        }

        # Определяем тип модели и соответствующий способ сохранения
        if hasattr(model, 'state_dict'):
            # Для PyTorch моделей
            checkpoint['model_state'] = model.state_dict()
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            # Используем torch.save для PyTorch моделей
            if torch is not None:
                torch.save(checkpoint, checkpoint_path)
            else:
                print("⚠️ PyTorch не установлен, невозможно сохранить чекпоинт PyTorch модели")
                return
        else:
            # Для sklearn-подобных моделей
            checkpoint['model'] = model
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pkl")
            joblib.dump(checkpoint, checkpoint_path)

        # Логируем в MLflow
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
        print(f"✅ Чекпоинт сохранен: {checkpoint_path}")

    def save_comprehensive_artifacts(self, X_test, y_test):
        """
        Комплексное сохранение всех артефактов эксперимента

        Args:
            X_test (pd.DataFrame): Тестовые данные
            y_test (pd.Series): Тестовые целевые значения
        """
        # Создаем директории
        os.makedirs(self.config['output']['artifacts_dir'], exist_ok=True)

        # 1. Сохраняем метрики
        metrics_path = os.path.join(self.config['output']['artifacts_dir'], 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # 2. Сохраняем feature importance
        importance_df = get_feature_importance_report(self.model, self.feature_names)
        if importance_df is not None:
            importance_path = os.path.join(self.config['output']['artifacts_dir'], 'feature_importance.csv')
            importance_df.to_csv(importance_path, index=False)

        # 3. Сохраняем примеры предсказаний
        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': self.model.predict(X_test),
            'probability': self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        })
        predictions_path = os.path.join(self.config['output']['artifacts_dir'], 'predictions_sample.csv')
        predictions_df.head(100).to_csv(predictions_path, index=False)

        # 4. Сохраняем конфигурацию
        shutil.copy2('configs/config.yaml', 
                    os.path.join(self.config['output']['artifacts_dir'], 'training_config.yaml'))

        print(f"✅ Все артефакты сохранены в: {self.config['output']['artifacts_dir']}")

    def log_to_mlflow(self):
        """
        Комплексное логирование эксперимента в MLflow
        """
        if not self.config['mlflow'].get('enabled', False):
            return

        try:
            # Настройка MLflow
            mlflow.set_experiment(self.config['mlflow']['experiment_name'])
            
            with mlflow.start_run() as run:
                self.current_run_id = run.info.run_id
                
                # Логируем параметры модели
                mlflow.log_params(self.config['model']['params'])
                mlflow.log_param('model_type', self.config['model']['type'])
                
                # Логируем параметры данных
                mlflow.log_param('data_limit', self.config['data']['limit'])
                mlflow.log_param('test_size', self.config['training']['test_size'])
                mlflow.log_param('random_state', self.config['training']['random_state'])
                mlflow.log_param('balance_data', self.config['training'].get('balance_data', True))
                
                # Логируем метрики
                mlflow.log_metrics(self.metrics)
                
                # Логируем модель
                mlflow.sklearn.log_model(self.model, "model")
                
                # Логируем артефакты
                mlflow.log_artifact("configs/config.yaml")
                mlflow.log_artifact(os.path.join(self.config['output']['artifacts_dir'], 'metrics.json'))
                
                importance_path = os.path.join(self.config['output']['artifacts_dir'], 'feature_importance.csv')
                if os.path.exists(importance_path):
                    mlflow.log_artifact(importance_path)
                
                plots_path = os.path.join(self.config['output']['artifacts_dir'], 'evaluation_plots.png')
                if os.path.exists(plots_path):
                    mlflow.log_artifact(plots_path)
                
                # Логируем дополнительную информацию
                self.log_environment()
                self.log_git_info()
                
                # Документируем гипотезу и результаты (пример)
                mlflow.log_param("hypothesis", "Базовая модель для прогнозирования оттока клиентов")
                mlflow.log_param("business_context", "Модель для выявления клиентов с риском оттока")
                
                conclusion = f"Модель {self.config['model']['type']} достигла ROC-AUC: {self.metrics.get('roc_auc', 0):.4f}"
                mlflow.log_param("conclusion", conclusion)

                print("✅ Эксперимент полностью записан в MLflow")
                print(f"   Run ID: {self.current_run_id}")

        except Exception as e:
            print(f"⚠️ Ошибка логирования в MLflow: {e}")

    def register_best_model(self, run_id=None):
        """
        Регистрация лучшей модели в MLflow Model Registry

        Args:
            run_id (str): ID запуска MLflow (если None, используется текущий)
        """
        if not self.config['mlflow'].get('enabled', False):
            return

        try:
            if run_id is None:
                run_id = self.current_run_id
            if run_id is None:
                print("⚠️ Нет run_id для регистрации модели")
                return

            # Регистрируем модель
            model_uri = f"runs:/{run_id}/model"
            registered_model = mlflow.register_model(model_uri, "CustomerChurnModel")

            # Добавляем описание
            client = mlflow.tracking.MlflowClient()
            client.update_registered_model(
                name=registered_model.name,
                description="Модель для прогнозирования оттока клиентов"
            )

            # Добавляем метки
            client.set_registered_model_tag(
                name=registered_model.name,
                key="problem_type",
                value="classification"
            )
            
            client.set_registered_model_tag(
                name=registered_model.name,
                key="metric_roc_auc",
                value=str(self.metrics.get('roc_auc', 0))
            )

            print(f"✅ Модель зарегистрирована: {registered_model.name} версия {registered_model.version}")
            
            return registered_model

        except Exception as e:
            print(f"❌ Ошибка регистрации модели: {e}")

    def create_evaluation_plots(self, X_test, y_test):
        """
        Создание визуализаций для оценки модели

        Args:
            X_test (pd.DataFrame): Тестовые данные
            y_test (pd.Series): Тестовые целевые значения
        """
        if not self.config['output'].get('create_plots', True):
            return

        try:
            print("Создание визуализаций...")
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Матрица ошибок
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            axes[0, 0].set_title('Матрица ошибок')
            axes[0, 0].set_xlabel('Предсказанные')
            axes[0, 0].set_ylabel('Фактические')

            # 2. Важность признаков
            importance_df = get_feature_importance_report(self.model, self.feature_names)
            if importance_df is not None:
                if 'importance' in importance_df.columns:
                    importance_df.sort_values('importance', ascending=True).plot(
                        kind='barh', x='feature', y='importance', ax=axes[0, 1]
                    )
                else:
                    importance_df.sort_values('abs_importance', ascending=True).plot(
                        kind='barh', x='feature', y='abs_importance', ax=axes[0, 1]
                    )
                axes[0, 1].set_title('Важность признаков')

            # 3. ROC кривая (если есть вероятности)
            if y_pred_proba is not None:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                axes[1, 0].plot(fpr, tpr, label=f'ROC curve (AUC = {self.metrics.get("roc_auc", 0):.3f})')
                axes[1, 0].plot([0, 1], [0, 1], 'k--')
                axes[1, 0].set_xlabel('False Positive Rate')
                axes[1, 0].set_ylabel('True Positive Rate')
                axes[1, 0].set_title('ROC кривая')
                axes[1, 0].legend()
            else:
                axes[1, 0].text(0.5, 0.5, 'ROC кривая недоступна\n(модель не возвращает вероятности)', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)

            # 4. Сравнение фактических и предсказанных значений
            axes[1, 1].scatter(range(len(y_test)), y_test, alpha=0.5, label='Фактические')
            axes[1, 1].scatter(range(len(y_pred)), y_pred, alpha=0.5, label='Предсказанные')
            axes[1, 1].set_xlabel('Образцы')
            axes[1, 1].set_ylabel('Класс')
            axes[1, 1].set_title('Сравнение фактических и предсказанных значений')
            axes[1, 1].legend()

            plt.tight_layout()

            # Сохранение графиков
            plots_path = os.path.join(self.config['output']['artifacts_dir'], 'evaluation_plots.png')
            plt.savefig(plots_path, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"✅ Визуализации сохранены: {plots_path}")

        except Exception as e:
            print(f"⚠️ Ошибка создания визуализаций: {e}")

    def save_to_s3(self):
        """
        Сохранение моделей и артефактов в S3
        """
        if not self.config.get('s3', {}).get('enabled', False):
            return

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            bucket = self.config['s3']['bucket']

            # Создаем структуру в домашней директории
            s3_backup_dir = f"s3_backup_{timestamp}"
            local_backup_path = os.path.join(os.getcwd(), s3_backup_dir)
            os.makedirs(local_backup_path, exist_ok=True)

            # Копируем модели
            model_files = [f for f in os.listdir(self.config['output']['model_dir']) 
                         if os.path.isfile(os.path.join(self.config['output']['model_dir'], f))]
            for model_file in model_files:
                shutil.copy2(os.path.join(self.config['output']['model_dir'], model_file), 
                           local_backup_path)

            # Копируем артефакты
            artifact_files = [f for f in os.listdir(self.config['output']['artifacts_dir']) 
                            if os.path.isfile(os.path.join(self.config['output']['artifacts_dir'], f))]
            for artifact_file in artifact_files:
                shutil.copy2(os.path.join(self.config['output']['artifacts_dir'], artifact_file), 
                           local_backup_path)

            # Копируем конфигурацию
            shutil.copy2('configs/config.yaml', local_backup_path)

            # Используем команды shell для копирования в S3
            s3_dest_path = f"/mnt/s3/{bucket}/backups/{s3_backup_dir}"
            os.makedirs(s3_dest_path, exist_ok=True)

            # Копируем с помощью shell команд
            subprocess.run(['cp', '-r', local_backup_path + '/.', s3_dest_path], check=True)

            # Очищаем временные файлы
            shutil.rmtree(local_backup_path)

            print(f"✅ Модели и артефакты сохранены в S3: {s3_dest_path}")

        except Exception as e:
            print(f"⚠️ Ошибка сохранения в S3: {e}")
            print("Но это не критично - все файлы сохранены локально")

    def run_training_pipeline(self):
        """
        Запуск полного пайплайна обучения с комплексным трекингом
        """
        print("Запуск пайплайна обучения...")
        print("=" * 50)

        try:
            # 1. Загрузка данных
            X, y = self.load_data()

            # 2. Подготовка данных
            X_train, X_test, y_train, y_test = self.prepare_data(X, y)

            # 3. Обучение модели
            self.train_model(X_train, y_train)

            # 4. Оценка модели
            self.evaluate_model(X_test, y_test)

            # 5. Сохранение артефактов
            self.save_comprehensive_artifacts(X_test, y_test)

            # 6. Создание визуализаций
            self.create_evaluation_plots(X_test, y_test)

            # 7. Логирование в MLflow
            self.log_to_mlflow()

            # 8. Регистрация модели (опционально)
            if self.config['mlflow'].get('register_model', False):
                self.register_best_model()

            # 9. Сохранение в S3 (опционально)
            self.save_to_s3()

            print("=" * 50)
            print("Пайплайн обучения успешно завершен!")
            if self.current_run_id:
                print(f"Run ID: {self.current_run_id}")
            return self.model, self.metrics

        except Exception as e:
            print(f"Ошибка в пайплайне обучения: {e}")
            raise


def reproduce_experiment(run_id):
    """
    Воспроизведение эксперимента по run_id

    Args:
        run_id (str): ID запуска MLflow

    Returns:
        model: Воспроизведенная модель
    """
    try:
        # Получаем информацию о запуске
        run = mlflow.get_run(run_id)
        print(f"Воспроизведение эксперимента: {run_id}")
        print(f"Параметры: {run.data.params}")
        print(f"Метрики: {run.data.metrics}")

        # Загружаем модель
        model_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
        model = mlflow.sklearn.load_model(f"{model_path}")

        # Загружаем конфигурацию
        try:
            config_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="training_config.yaml")
            print(f"✅ Конфигурация загружена: {config_path}")
        except:
            print("⚠️ Конфигурация не найдена в артефактах")

        print("✅ Эксперимент успешно воспроизведен")
        return model

    except Exception as e:
        print(f"❌ Ошибка воспроизведения эксперимента: {e}")
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
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        print("=" * 50)
        for metric, value in metrics.items():
            print(f" {metric.upper()}: {value:.4f}")

        return model, metrics

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        return None, None


if __name__ == "__main__":
    main()
