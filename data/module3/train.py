"""
Основной скрипт обучения модели прогнозирования оттока клиентов
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

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Класс для обучения и оценки ML моделей
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
        
        print(f"✅ Данные подготовены:")
        print(f"   - Train: {X_train.shape[0]} samples")
        print(f"   - Test: {X_test.shape[0]} samples")
        
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
            print(f"   - {metric}: {value:.4f}")
        
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
            print(f"Кросс-валидация F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.metrics
    
    
    def log_to_mlflow(self):
        """
        Логирование эксперимента в MLflow
        """
        if not self.config['mlflow'].get('enabled', False):
            return
        
        try:
            # Настройка MLflow
            mlflow.set_experiment(self.config['mlflow']['experiment_name'])
            
            with mlflow.start_run():
                # Логируем параметры
                mlflow.log_params(self.config['model']['params'])
                mlflow.log_param('model_type', self.config['model']['type'])
                
                # Логируем метрики
                mlflow.log_metrics(self.metrics)
                
                # Логируем модель
                mlflow.sklearn.log_model(self.model, "model")
                
                # Логируем важность признаков
                if hasattr(self.model, 'feature_importances_') or hasattr(self.model, 'coef_'):
                    importance_df = get_feature_importance_report(self.model, self.feature_names)
                    if importance_df is not None:
                        importance_path = "feature_importance.csv"
                        importance_df.to_csv(importance_path, index=False)
                        mlflow.log_artifact(importance_path)
                
                print("✅ Эксперимент записан в MLflow")
                
        except Exception as e:
            print(f"⚠️ Ошибка логирования в MLflow: {e}")
    
    
    def save_model_and_artifacts(self, X_test, y_test):
        """
        Сохранение модели и артефактов
        
        Args:
            X_test (pd.DataFrame): Тестовые данные для примеров
            y_test (pd.Series): Тестовые целевые значения
        """
        print("Сохранение модели и артефактов...")
        
        # Создаем директории если нужно
        os.makedirs(self.config['output']['model_dir'], exist_ok=True)
        os.makedirs(self.config['output']['artifacts_dir'], exist_ok=True)
        
        # Сохраняем модель
        model_path = os.path.join(
            self.config['output']['model_dir'], 
            f"{self.config['model']['type']}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        )
        joblib.dump(self.model, model_path)
        
        # Сохраняем метрики
        metrics_path = os.path.join(self.config['output']['artifacts_dir'], 'metrics.json')
        with open(metrics_path, 'w') as f:
            import json
            json.dump(self.metrics, f, indent=2)
        
        # Сохраняем примеры предсказаний
        predictions = self.model.predict(X_test)
        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': predictions
        })
        predictions_path = os.path.join(self.config['output']['artifacts_dir'], 'predictions_sample.csv')
        predictions_df.head(100).to_csv(predictions_path, index=False)
        
        # Сохраняем важность признаков
        importance_df = get_feature_importance_report(self.model, self.feature_names)
        if importance_df is not None:
            importance_path = os.path.join(self.config['output']['artifacts_dir'], 'feature_importance.csv')
            importance_df.to_csv(importance_path, index=False)
        
        print(f"✅ Модель сохранена: {model_path}")
        print(f"✅ Артефакты сохранены в: {self.config['output']['artifacts_dir']}")
        
        return model_path
    
    
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
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
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
            
            # 3. Распределение вероятностей
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            axes[1, 0].plot(fpr, tpr, label=f'ROC curve (AUC = {self.metrics["roc_auc"]:.3f})')
            axes[1, 0].plot([0, 1], [0, 1], 'k--')
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('ROC кривая')
            axes[1, 0].legend()
            
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
        if not self.config['s3']['enabled']:
            return
        
        try:
            import shutil
            from pathlib import Path
            
            bucket = self.config['s3']['bucket']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            
            # Создаем структуру в домашней директории
            s3_backup_dir = f"s3_backup_{timestamp}"
            local_backup_path = Path(s3_backup_dir)
            
            (local_backup_path / 'models').mkdir(parents=True, exist_ok=True)
            (local_backup_path / 'artifacts').mkdir(parents=True, exist_ok=True)
            
            # Копируем модели
            model_files = list(Path(self.config['output']['model_dir']).glob('*'))
            for model_file in model_files:
                if model_file.is_file():
                    shutil.copy2(model_file, local_backup_path / 'models')
            
            # Копируем артефакты
            artifact_files = list(Path(self.config['output']['artifacts_dir']).glob('*'))
            for artifact_file in artifact_files:
                if artifact_file.is_file():
                    shutil.copy2(artifact_file, local_backup_path / 'artifacts')
            
            # Копируем конфигурацию
            shutil.copy2('configs/config.yaml', local_backup_path)
            
            # Используем команды shell для копирования в S3 (как в модуле 3_2)
            s3_dest_path = f"/mnt/s3/{bucket}/backups/{s3_backup_dir}"
            
            # Создаем директорию в S3
            os.makedirs(s3_dest_path, exist_ok=True)
            
            # Копируем с помощью shell команд
            import subprocess
            subprocess.run(['cp', '-r', str(local_backup_path) + '/.', s3_dest_path], check=True)
            
            # Очищаем временные файлы
            shutil.rmtree(local_backup_path)
            
            print(f"✅ Модели и артефакты сохранены в S3: {s3_dest_path}")
            
        except Exception as e:
            print(f"⚠️ Ошибка сохранения в S3: {e}")
            print("Но это не критично - все файлы сохранены локально")
    
    
    def run_training_pipeline(self):
        """
        Запуск полного пайплайна обучения
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
            
            # 5. Логирование в MLflow
            self.log_to_mlflow()
            
            # 6. Сохранение модели и артефактов
            model_path = self.save_model_and_artifacts(X_test, y_test)
            
            # 7. Создание визуализаций
            self.create_evaluation_plots(X_test, y_test)
            
            # 8. Сохранение в S3
            self.save_to_s3()
            
            print("=" * 50)
            print("Пайплайн обучения успешно завершен!")
            print(f"Модель сохранена: {model_path}")
            
            return self.model, self.metrics
            
        except Exception as e:
            print(f"Ошибка в пайплайне обучения: {e}")
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
            print(f"   {metric.upper()}: {value:.4f}")
        
        return model, metrics
        
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        return None, None


if __name__ == "__main__":
    main()