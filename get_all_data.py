import pandas as pd
import os

results_dir = 'Results Files'
def get_all_result_file_data() -> pd.DataFrame:
    dfs = []
    for model_dir in os.listdir(results_dir):
        model_path = f'{results_dir}/{model_dir}'
        if not os.path.isdir(model_path):
            continue
        for file in os.listdir(model_path):
            result_df = pd.read_csv(f'{model_path}/{file}')

            if 'xsum' in file:
                result_df['dataset'] = 'xsum'
            elif 'writingprompts' in file:
                result_df['dataset'] = 'writingprompts'
            elif 'wmt16' in file:
                result_df['dataset'] = 'wmt16'
            elif 'squad' in file:
                result_df['dataset'] = 'squad'
            else:
                result_df['dataset'] = 'pubmedqa'

            result_df['test'] = 1 if 'test' in file else 0

            dfs.append(result_df)

    return pd.concat(dfs)