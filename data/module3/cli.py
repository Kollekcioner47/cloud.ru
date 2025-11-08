"""
CLI-интерфейс для запуска обучения модели
"""

import click
from train import ModelTrainer

@click.command()
@click.option('--config', default='configs/config.yaml', help='Путь к конфигурационному файлу')
@click.option('--experiment-name', help='Название эксперимента в MLflow')
def train(config, experiment_name):
    """Запускает обучение модели с указанными параметрами"""
    trainer = ModelTrainer(config)
    
    if experiment_name:
        trainer.config['mlflow']['experiment_name'] = experiment_name
    
    trainer.run_training_pipeline()

if __name__ == '__main__':
    train()