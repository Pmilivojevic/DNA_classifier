import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from dnaseq.entity.config_entity import ModelTrainerConfig
from dnaseq import logger
import joblib

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.cv = CountVectorizer()
        self.clf1 = LogisticRegression()
        self.clf2 = MultinomialNB()
        self.config = config
        self.config.params[0]['classifier'] = [self.clf1]
        self.config.params[1]['classifier'] = [self.clf2]
        
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        
        train_x = train_data.drop([self.config.target_column], axis=1).values.astype('U')
        train_y = train_data[self.config.target_column]
        
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
        
        pipeline = Pipeline(
            [
                ('preprocessor', self.cv),
                ('classifier', self.clf1)
            ]
        )
        
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=self.config.params,
            scoring='accuracy',
            cv=outer_cv,
            n_jobs=1,
            verbose=3,
            return_train_score=True
        )
        
        grid.fit(train_x.ravel(), train_y.ravel())
        
        print(grid.best_params_)
        print(grid.best_score_)
        
        joblib.dump(grid.best_estimator_, os.path.join(self.config.root_dir, self.config.model_name))
