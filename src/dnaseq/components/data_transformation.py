import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import scipy as sp
from dnaseq.entity.config_entity import DataTransformationConfig
from dnaseq import logger

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_data(self):
        def getKmers(sequence, size=6):
            return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

        data = pd.read_table(self.config.data_path)
        data['kmers'] = data.apply(lambda x: getKmers(x['sequence']), axis=1)
        data = data.drop('sequence', axis=1)

        X_data = list(data['kmers'])
        for item in range(len(X_data)):
            X_data[item] = ' '.join(X_data[item])

        y_data = data.iloc[:, 0].values

        ngram_range = self.config.ngram, self.config.ngram
        cv = CountVectorizer(ngram_range=ngram_range)
        X_data_cv = cv.fit_transform(X_data)
        
        logger.info("Transformed train data with CoountVectorizer")
        logger.info(X_data_cv.shape)
        logger.info(type(X_data_cv))

        model = SentenceTransformer('all-MiniLM-L6-v2')
        X_data_emb = model.encode(X_data)
        
        logger.info("Transformed train data with SentenceTransformer")
        logger.info(X_data_emb.shape)
        logger.info(type(X_data_emb))

        return X_data_cv, X_data_emb, y_data

    def train_test_spliting(self, x_cv, x_emb, y):
        X_train_cv, X_test_cv, y_train, y_test = train_test_split(
            x_cv,
            y,
            test_size = 0.20,
            random_state=42
        )

        X_train_emb, X_test_emb, y_train, y_test = train_test_split(
            x_emb,
            y,
            test_size = 0.20,
            random_state=42
        )

        sp.sparse.save_npz(os.path.join(self.config.root_dir, 'X_train_cv.npz'), X_train_cv)
        sp.sparse.save_npz(os.path.join(self.config.root_dir,'X_test_cv.npz'), X_test_cv)
        pd.DataFrame(X_train_emb).to_csv(
            os.path.join(self.config.root_dir, "X_train_emb.csv"),
            index=False
        )
        pd.DataFrame(X_test_emb).to_csv(
            os.path.join(self.config.root_dir, "X_test_emb.csv"),
            index=False
        )
        pd.DataFrame(y_train).to_csv(
            os.path.join(self.config.root_dir, "y_train.csv"),
            index=False
        )
        pd.DataFrame(y_test).to_csv(
            os.path.join(self.config.root_dir, "y_test.csv"),
            index=False
        )

        logger.info("Splited data into training and test sets")
