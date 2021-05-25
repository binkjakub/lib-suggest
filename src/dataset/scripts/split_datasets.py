from src.data.loader import load_all_datasets
from src.dataset.split import split_leave_one_out
from src.defaults import RAW_DATA, REPO_DS, TEST_DS, TRAIN_DS

RANDOM_STATE = 2137

DROP_DUPLICATES = True
DROP_NAN_COLUMNS = True
MIN_LIB_OCCURRENCES = 10
MIN_REQUIREMENTS_REPO = 5

dataset = load_all_datasets(RAW_DATA,
                            drop_duplicates=DROP_DUPLICATES,
                            drop_all_nan_cols=DROP_NAN_COLUMNS,
                            min_requirements_num=MIN_REQUIREMENTS_REPO,
                            min_library_occurrences=MIN_LIB_OCCURRENCES)

print(f"Dataset size: {len(dataset)}")
print(f"Dataset features:\n{dataset.columns.tolist()}")

interactions = dataset[['full_name', 'repo_requirements']]
interactions = interactions.explode('repo_requirements').reset_index(drop=True)
repo_features = dataset[dataset.columns.difference(['repo_requirements'])]

train_interactions, test_interactions = split_leave_one_out(interactions,
                                                            repo_col='full_name',
                                                            num_test_examples=1,
                                                            random_state=RANDOM_STATE)

print(f"Train size: {len(train_interactions)}")
print(f"Test size: {len(test_interactions)}")

repo_features.to_csv(REPO_DS, index=False)
train_interactions.to_csv(TRAIN_DS, index=False)
test_interactions.to_csv(TEST_DS, index=False)
