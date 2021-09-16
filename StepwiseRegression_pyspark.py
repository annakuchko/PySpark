# Stepwise Regression with Pyspark

import pandas as pd
from pyspark.ml.base import _FitMultipleIterator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

class StepwiseRegression():
    def __init__(self, 
                 estimator, 
                 features, 
                 target, 
                 metric='roc_auc', 
                 forward=True,
                 verbose=False):
        self.estimator = estimator 
        self.features = features
        self.target = target
        self.metric = metric
        self.forward = forward
        self.verbose = verbose

    def fitMultiple_scores(self, dataset, features, selected_features):
        target = self.target
        estimator = self.estimator
        verbose = self.verbose 
        forward = self.forward
        dataset = dataset.cache()
        
        def fitSingleModel_score(index):
            col = features[index]
            if forward:
                if verbose:
                    print(f'Trying to add col: {col}')
                vectorAssembler = VectorAssembler().\
                    setParams(inputCols=selected_features + [col],
                              outputCol='features')
                if verbose:
                    print(selected_features + [col])
            else:
                if verbose:
                    print(f'Trying to drop col: {col}')
                vectorAssembler = VectorAssembler().\
                    setParams(inputCols=list(set(features)-set([col])),
                              outputCol='features')
            
            va_df = vectorAssembler.transform(dataset)
            estimator.setLabelCol(target).setFeaturesCol('features')
            return estimator.fit(va_df)
        return _FitMultipleIterator(fitSingleModel_score, len(features))

    def fit_scores(self, dataset, features, selected_features):
        models = [None] * len(features)
        # cols = [None] * len(features)
        for index, model in self.fitMultiple_scores(
                dataset, features, selected_features
                ):
            models[index] = model
        return models
    
    def scorer(self, dataset, features, selected_features):
        metric= self.metric
        scores = []
        lr_models = self.fit_scores(dataset, features, selected_features)
        for m in lr_models:
            if metric == 'roc_auc':
                score = m.summary.areaUnderRoc
            elif metric == 'accuracy':
                score = m.summary.accuracy
            elif metric == 'r2':
                score = m.summary.r2
            elif metric == 'rmse':
                score = m.summary.rmse
            scores.append(score)
        return pd.Series(scores, index=features, name='SCORES')

    def transform(self, dataset):
        metric = self.metric
        features = self.features
        verbose = self.verbose 
        forward = self.forward
        selected_features = []
        dropped = True
        prev_score = 0.0
        while dropped:
            dropped = False
            if verbose:
                print(f'Calculating {metric}...')
            scores = self.scorer(dataset, features, selected_features)
            if verbose:
                print(f'Calculating max {metric}...')
            maxscore = max(scores)
            if verbose:
                print(f'Max {metric}: {maxscore}')
            maxloc = scores[scores == maxscore].index
            if maxscore > prev_score:
                if forward:
                    if verbose:
                        print(f'Adding {maxloc[0]}'
                              f' with {metric}: {str(maxscore)}')
                    features = list(set(features) - set(maxloc[0]))
                    selected_features = selected_features + [(maxloc[0])]
                    prev_score = maxscore
                    dropped = True
                    if verbose:
                        print(f'Remaining variables:')
                        print(selected_features)
                else:
                    if maxscore > prev_score:
                        if verbose:
                            print(f'Dropping {maxloc[0]}'
                                  f' with {metric}: {str(maxscore)}')
                        features = list(set(features) - set([maxloc[0]]))
                        prev_score = maxscore
                        dropped = True
                        selected_features = features
                        if verbose:
                            print(f'Remaining variables:')
                            print(selected_features)
        return dataset.select(selected_features)

# Example                     
if __name__ == '__main__':
    lr = LogisticRegression(standardization=True)
    rocauc = StepwiseRegression(
        estimator=lr, 
        features=inputCols, 
        target='target_binary', 
        metric='roc_auc', 
        forward=True,
        verbose=True
        )
    df_rocauc = rocauc.transform(df)
                    