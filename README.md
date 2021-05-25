# lib-suggest

Recommender system for suggesting some new libraries for Python projects.   
Authors: Jakub Binkowski, Denis Janiak, Albert Sawczyn

----

## Usage

To run lib-suggest recommendation interface, simply type following command:

## Application
```bash
GITHUB_TOKEN=<your_github_access_token>; streamlit run recommend.py
```

## Dataset loading

Dataset is provided with dvc under `storage/dataset` directory, which contains following files:

```commandline
storage/dataset
├── repo_features.csv
├── test_interactions.csv
└── train_interaction.csv

```

To load dataset you can use the code below:

```python
import pandas as pd

from src.defaults import REPO_DS, TRAIN_DS, TEST_DS

repository_features = pd.read_csv(REPO_DS)
train_interactions = pd.read_csv(TRAIN_DS)
test_interactions = pd.read_csv(TEST_DS)
```
