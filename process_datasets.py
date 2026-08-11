import argparse
import os

import pandas as pd
    

if __name__ == '__main__':

    dataset_dir = 'Sampled Datasets'

    for files in os.listdir(dataset_dir):
        df = pd.read_parquet(f'{dataset_dir}/{files}')
        
        print(f'------ Running for file {files} -------')

        if 'xsum' in files:
            df = df[['document']]

        elif 'writingprompts' in files:
            df['prompt'] = df['prompt'].str.replace('[ WP ] ', '').str.strip()
            df['context'] = df.apply(
                lambda row:f'Using the following writing prompt, produce a story.\n'
                           f'Writing prompt: {row['prompt']}\nStory: ',
                axis=1
            )  # provide format to hopefully avoid "sure based on the context..." and such phrases appearing in output
            df['document'] = df['story']
            df = df [['document', 'context']]

        elif 'pubmedqa' in files:
            # PubMedQA context column is in JSON format, parquet preloads data in this column as a dict so can query it directly
            df['context'] = df['context'].apply(lambda row: ' '.join(row['contexts']).strip())
            df['context'] = df.apply(
                lambda row: f'Answer the question with support from the context. '
                           f'Question: {row['question'].strip()}\nContext: {row['context']}\nAnswer: ',
                axis=1
            )  # provide format to hopefully avoid "sure based on the context..." and such phrases appearing in output
            df['document'] = df['long_answer']
            df = df[['document', 'context']]

        elif 'wmt16' in files:
            # wmt16 translation column is in JSON format, parquet preloads data in this column as a dict so can query it directly
            df['document'] = df['translation'].apply(lambda row: ' '.join(row['en']))
            df['context'] = df.apply(
                lambda row: f'Translate the following German text into English.\n'
                            f'German: {row['translation']['de'].strip()}\nTranslation: ',
                axis=1
            )  # provide format to hopefully avoid "sure based on the context..." and such phrases appearing in output
            df = df[['document', 'context']]

        elif 'squad' in files:
            # SQuAD answers column is in JSON format, parquet preloads data in this column as a dict so can query it directly
            df['document'] = df['answers'].apply(lambda row: ' '.join(row['text']))
            df['context'] = df.apply(
                lambda row: f'Answer the question with support from the context. '
                           f'Question: {row['question'].strip()}\nContext: {row['context'].strip()}\nAnswer: ',
                axis=1
            )  # provide format to hopefully avoid "sure based on the context..." and such phrases appearing in output
            df = df[['document', 'context']]

        else:
            raise NotImplementedError(
                f'The following file {files} is not recognized and does not have implemented preprocessing steps'
            )

        df['is_machine'] = 0
        print(f'------ Columns: {df.columns} -------')
        df.to_parquet(f'{dataset_dir}/{files}')
