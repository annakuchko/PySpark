# Class for autocorrelation elimination using 
# Variance Inflation Factor (VIF) values

import pandas as pd
from pyspark.ml.base import _FitMultipleIterator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

class VIFElimination:
    def __init__(self, features, verbose=False):
        self.estimator = LinearRegression(
            standardization=True,
            maxIter=5
            )
        self.features = features
        self.verbose = verbose 
        
    def fitMultiple_vif(self, dataset, features):
        estimator = self.estimator
        dataset = dataset.cache()
        def fitSingleModel_vif(index):
            col = features[index]
            vectorAssembler = VectorAssembler()\
                .setParams(
                    inputCols=list(
                        set(features)-set([col])
                        ), 
                    outputCol='features'
                    )
            va_df = vectorAssembler.transform(dataset)
            estimator.setLabelCol(col).setFeaturesCol('features')
            return estimator.fit(va_df)
        return _FitMultipleIterator(
            fitSingleModel_vif,
            len(features)
            )
    
    def fit_vif(self, dataset, features):
        models = [None] * len(features)
        cols = [None] * len(features)
        for index, model in self.fitMultiple_vif(dataset, features):
            models[index] = model
        return models
    
    def variance_inflation_factors(self, dataset, features):
        vifs = []
        lr_models = self.fit_vif(dataset, features)
        for m in lr_models:
            vif = 1.0 / (1.0 - m.summary.r2)
            vifs.append(vif)
        return pd.Series(vifs,
                         index=features,
                         name='VIF')
    
    def transform(self, dataset, thresh=5.0):
        features = self.features
        dropped = True
        while dropped:
            
            if self.verbose:
                print('calculating VIF...')
            
            vifs = self.variance_inflation_factors(dataset, features)
            
            if self.verbose:
                print('Calculating max VIF...')
            
            maxvif = max(vifs)
            maxloc = vifs[vifs==maxvif].index
            if maxvif > thresh:
                
                if self.verbose:
                    print(f'Dropping {maxloc[0]} with '
                          f'VIF: {str(maxvif)}')
                
                features = list(set(features) - set([maxloc[0]]))
                dropped = True
                remaining_features = features
                if self.verbose:
                    print('Remaining variables: ')
                    print(features)
        
        return dataset.select(features)
    
# Example
if __name__ == '__main__':
    vif = VIFElimination(features=in_cols,
                         verbose=True)
    df_vif = vif.transform(df)

                



