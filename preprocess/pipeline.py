import pickle
import os
from pathlib import Path
from queue import Queue
import random
import csv
from . import generate_cv_template, generate_jd_template
from . import generate_pairs
from tqdm import tqdm
import pandas as pd


class Pipeline:
    def __init__(self,
                  cv_path = '/autodl-fs/data/wang/falcon/csvfile/CV.csv',
                  jd_path = '/autodl-fs/data/wang/falcon/csvfile/JD.csv', 
                  save_template_path = '/autodl-fs/data/song/jd_cv_gpt4_template',
                  save_data_path = '/root/havi/src/preprocess/data/',
                  pairs_path = '/root/autodl-fs/wang/falcon/JD_CV.csv'):
        """
        :param cv_path, jd_path, pairs_path: original cv/jd data directory.
        :param save_template_path: save cv/jd template directory.
        :param save_data_path: save preprocess data directory.
        """
        random.seed(0)
        self.cv_path = cv_path
        self.jd_path = jd_path
        self.save_cv_template_path = os.path.join(save_template_path, 'cv_template_embedding.csv')
        self.save_jd_template_path = os.path.join(save_template_path, 'jd_template_embedding.csv')

        self.save_data_path = save_data_path
        self.pairs_path = pairs_path


    def run(self):
        """
        Runner for pipeline.
        """
        print('1. Generating template...')
        self.generate_template()
        print('2. Generating pairs...')
        generate_pairs.generate_pairs(self.save_data_path, self.pairs_path)
        print('3. Sampling pairs...')
        sample_pairs = generate_pairs.sample_pairs(self.save_data_path)
        print('4. Saving data...')
        self.save_data(sample_pairs)
        print("End of preprocessing")
       
    
    def generate_template(self):
        print('generating cv template...')
        generate_cv_template.generate_cv_template(self.cv_path, self.save_cv_template_path)
        print('generating jd template...')
        generate_jd_template.generate_jd_template(self.jd_path, self.save_jd_template_path)


    def save_data(self, all_pairs):

        with open(os.path.join(self.save_data_path, 'dicts.pkl'), 'rb') as f:
            jd2idx = pickle.load(f)
            idx2jd = pickle.load(f)
            cv2idx = pickle.load(f)
            idx2cv = pickle.load(f)

        # save_jd_template_path = '/autodl-fs/data/song/jd_cv_gpt4_template/jd_template_embedding.csv'
        jd_df = pd.read_csv(self.save_jd_template_path)
        # save_cv_template_path = '/autodl-fs/data/song/jd_cv_gpt4_template/cv_template_embedding.csv'
        cv_df = pd.read_csv(self.save_cv_template_path)

        print('Number of jd: ', len(jd_df))  # 2461
        print('Number of cv: ', len(cv_df))  # 1560

        column_names = ['jd_id', 'cv_id', 'label', 'features']
        with open(os.path.join(self.save_data_path, 'jd_cv_features.csv'), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(column_names)
                for ((jd_id, cv_id), label) in all_pairs:
                    jd = idx2jd[jd_id]
                    # jd = 'a3662754-c4ad-484c-8c11-6689f8995dda'
                    jd_embedding = jd_df.loc[jd_df['uuid'] == jd]['embedding'].values[0]
                    
                    cv = idx2cv[cv_id]
                    # cv = 'e8fe1775-2106-447e-a12a-b87990fa7bc5'
                    cv_embedding = cv_df.loc[cv_df['uuid'] == cv]['embedding'].values[0]

                    new_row = []
                    new_row.append(jd_id)
                    new_row.append(cv_id)
                    new_row.append(label)
                    new_row.append(jd_embedding + cv_embedding)
                    writer.writerow(new_row)
        print("End of Saving data")

