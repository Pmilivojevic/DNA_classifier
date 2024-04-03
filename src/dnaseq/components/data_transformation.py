import os
import pandas as pd
from sklearn.model_selection import train_test_split
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
        
        data = pd.DataFrame({'sequence': X_data, 'class': y_data})
        
        return data

    def train_test_spliting(self, data):
        train, test = train_test_split(data)
        
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
        
        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)
