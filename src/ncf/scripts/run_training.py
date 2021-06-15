from typing import Any

import click
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger

from src.ncf.data import LibRecommenderDM
from src.ncf.models import GMF, MLP, NeuMF
from src.ncf.recommender import RecommenderSystem


@click.command()
@click.option('--log-dir', type=click.Path(file_okay=False), required=True,
              help="Directory where logs will be stored")
@click.option('--experiment-name', type=click.STRING, default='ncf',
              help="Name of experiment passed to the logger")
@click.option('--seed', type=click.INT, default=2137123,
              help="Random seed for reproducibility")
def run_ncf_training(log_dir: str, experiment_name: str, seed: int):
    config = {
        'model': 'mlp',
        'embedding_dim': 16,
        'manual_feat_dim': len(LibRecommenderDM.FEATURE_NAMES),
        # 'manual_feat_dim': 0,
        'manual_feat_combination_out_dim': 0,
        'num_repos': 1016,
        'num_libs': 401,
        'num_negatives': 4,
        'batch_size': 128,
        'num_workers': 0,
        'learning_rate': 1e-3,
        'max_epochs': 32,
        'k_eval': 5,
        'seed': seed,
    }
    config = _compute_dimensions(config)
    seed_everything(seed)

    book_crossing_dm = LibRecommenderDM(config)
    model = _get_model(config)

    experiment_name = f'{type(model).__name__}__{experiment_name}'
    trainer = Trainer(
        max_epochs=config['max_epochs'],
        logger=_get_loggers(log_dir, experiment_name),
        default_root_dir=log_dir,
        deterministic=True,
    )
    trainer.fit(model, datamodule=book_crossing_dm)


def _compute_dimensions(config: dict[str, Any]) -> dict[str, Any]:
    if config['manual_feat_combination_out_dim']:  # when using combination layer after feat concat
        assert config['manual_feat_dim']
        repo_embedding_dim = config['manual_feat_combination_out_dim']
    elif config['manual_feat_dim']:  # when using only feat concat
        repo_embedding_dim = config['manual_feat_dim'] + config['embedding_dim']
    else:
        repo_embedding_dim = config['embedding_dim']

    lib_embedding_dim = config['embedding_dim']

    concat_dim = repo_embedding_dim + lib_embedding_dim
    config |= {
        'latent_dim_mf': None,
        'layers': [concat_dim, 32, 8],
    }
    return config


def _get_loggers(log_dir: str, experiment_name: str) -> LightningLoggerBase:
    return WandbLogger(experiment_name, log_dir, project='lib_suggest_ncf', log_model=True)


def _get_model(config: dict[str, Any]) -> RecommenderSystem:
    name = config['model']
    if name == 'gmf':
        return GMF(config)
    elif name == 'mlp':
        return MLP(config)
    elif name == 'neumf':
        return NeuMF(config)
    else:
        raise ValueError(f"Invalid model name: {name}")


run_ncf_training()
