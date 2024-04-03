from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
import numpy as np
import joblib
from dnaseq.entity.config_entity import ModelEvaluationConfig
from dnaseq.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        
        return rmse, mae, r2
    
    def evaluate(self):
        test_data = pd.read_csv('artifacts/data_transformation/test.csv')
        test_x = test_data.drop(self.config.target_column, axis=1).values.astype('U')
        test_y = test_data[self.config.target_column]
        
        estimator = joblib.load(self.config.model_path)
        
        test_x = estimator['preprocessor'].transform(test_x.ravel())
        pred = estimator['classifier'].predict(test_x)
        
        (rmse, mae, r2) = self.eval_metrics(test_y, pred)
        
        scores = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
        
        cm = confusion_matrix(test_y, pred)
        
        save_json(path=Path(self.config.metric_file_name), data=scores)
        
        with open(self.config.cm_name, 'w') as fw:
            fw.write(str(cm))
