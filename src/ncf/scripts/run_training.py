import click
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger

from src.ncf.data import LibRecommenderDM
from src.ncf.models import NeuMF


@click.command()
@click.option('--log-dir', type=click.Path(file_okay=False), required=True,
              help="Directory where logs will be stored")
@click.option('--experiment-name', type=click.STRING, default='ncf',
              help="Name of experiment passed to the logger")
def run_ncf_training(log_dir: str, experiment_name: str):
    config = {
        'num_repos': 1016,
        'num_libs': 401,
        'num_negatives': 4,
        'batch_size': 128,
        'num_workers': 16,
        'learning_rate': 1e-4,
        'latent_dim_mf': 32,
        'latent_dim_mlp': 32,
        'layers': [64, 32, 8],
        'max_epochs': 32,
    }

    seed_everything(2137)

    book_crossing_dm = LibRecommenderDM(config)
    book_crossing_dm.setup()
    model = NeuMF(config)
    # model = GMF(config)
    # model = MLP(config)

    experiment_name = f'{type(model).__name__}__{experiment_name}'
    trainer = Trainer(
        max_epochs=config['max_epochs'],
        logger=_get_loggers(log_dir, experiment_name),
        default_root_dir=log_dir,
        deterministic=True,
    )
    trainer.fit(model, datamodule=book_crossing_dm)


def _get_loggers(log_dir: str, experiment_name: str) -> LightningLoggerBase:
    return WandbLogger(experiment_name, log_dir, project='lib-suggest-ncf', log_model=True)


run_ncf_training()
