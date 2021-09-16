# Custom class for time series cross-valdation with Pyspark

import bisect 
from datetime import date, timedelta 
import numpy as np 
import pandas as pd
from pyspark.ml.tuning import CrossValidatorModel, Estimator, ValidatorParams, \
    HasParallelism, MLReadable, MLWritable, keyword_only
from multiprocessing.pool import ThreadPool

class TimeSeriesCVSplit:
    def __init__(self, 
                 initial_start_dt, 
                 initial_end_dt, 
                 step_size_in_months, 
                 test_size_in_months):
        self._initial_start_dt = initial_start_dt
        self._initial_end_dt = initial_end_dt
        self._step_size_in_months = step_size_in_months
        self._test_size_in_months = test_size_in_months
        
    def split(self, df, step_count_limit=1):
        dates = sorted(list(pd.to_datetime(
            df.select('report_dt').toPandas()['report_dt'].unique(),
            errors='coerce'
            )))
        period_start = bisect.bisect_left(
            dates, 
            self._initial_start_dt
            )
        start_dt = dates[period_start]
        period_end = bisect.bisect_left(
            dates, 
            self._initial_end_dt
            ) + self._test_size_in_months
        train_end_dates = []
        test_start_dates = []
        test_end_dates = []
        size = len(dates)
        step = 0
        while period_end < size and \
            step < step_count_limit:
                train_end_dt = dates[
                    period_end - \
                    self._test_size_in_months
                    ]
                test_start_dt = dates[
                    period_end - \
                        self._test_size_in_months + 1
                    ]
                test_end_dt = dates[period_end]
                
                train_end_dates.append(train_end_dt)
                test_start_dates.append(test_start_dt)
                test_end_dates.append(test_end_dt) 
                
                period_end += self._step_size_in_months
                step += 1
        return {'train_ends': train_end_dates,
                'test_starts': test_start_dates,
                'test_ends': test_end_dates}
    
def _parallelFitTasks(est, 
                      train, 
                      eva, 
                      validation, 
                      epm, 
                      collectSubModel, 
                      verbose=False):
    
    modelIter = est.fitMultiple(train, epm)
    
    def singleTask():
        index, model = next(modelIter)
        metric = eva.evaluate(model.transform(validation, 
                                              epm[index]))
        if verbose:
            print(f'{eva.getMetricName()}: {metric}')
        return index, metric, model if collectSubModel else None
    return [singleTask] * len(epm)

class CustomCrossValidator(Estimator, 
                           ValidatorParams, 
                           HasParallelism,
                           MLReadable,
                           MLWritable):
    @keyword_only 
    def __init__(self,
                 estimator=None,
                 estimatorParamMaps=None,
                 evaluator=None,
                 numFolds=5,
                 seed=2021,
                 parallelism=1,
                 collectSubModel=False,
                 foldCol=''):
        super(CustomCrossValidator, self).__init__()
        self._setDefault(parallelism=1)
        kwargs = self._input_kwargs
        self._set(**kwargs)
        self.numFolds = numFolds
        
    def _fit(self, dataset, verbose=True):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        eva = self.getOrDefault(self.evaluator)
        numFolds = self.numFolds
        numModels = len(epm)
        metrics = [0.0] * numFolds
        
        initial_start_dt = np.datetime64(
            pd.to_datetime(
                dataset.select(
                    f.min('report_dt')
                    ).first()[0] - datetime.timedelta(days=365)
                )
            )
        initial_end_dt = np.datetime64(
            datetime.datetime(
                year=2014, 
                month=1, 
                day=3
                )
            )
        step_size_in_months = 12
        test_size_in_months = 12
        
        if verbose:
            print('Dplitting data...')
        
        splitter = TimeSeriesCVSplit(
            initial_start_dt, 
            initial_end_dt, 
            step_size_in_months, 
            test_size_in_months
            )
        timeSeriesFolds = splitter.split(
            dataset, 
            numFolds
            )
        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        subModels = None
        
        if verbose:
            print(f'Fitting {len(estimatorParamMaps)} '
                  f'candidates for {numFolds} folds')
            print(f'Total {numFolds * len(estimatorParamMaps)} fits')
        
    
        for i in range(numFolds):
            if verbose:
                print(f'Fold {i+1} out of {numFolds}')
            train_end_dt = timeSeriesFolds['train_ends'][i]
            test_start_dt = timeSeriesFolds['test_starts'][i]
            test_end_dt = timeSeriesFolds['test_ends'][i]
            training_data = dataset.where(
                f.col('report_dt') <= train_end_dt
                )
            validation_data = dataset.where(
                (f.col('report_dt') >= test_start_dt) & \
                (f.col('report_dt') < test_end_dt)
                )
            training_data = training_data.cache()
            validation_data = validation_data.cache()
            
            tasks = _parallelFitTasks(
                est, 
                training_data, 
                eva, 
                validation_data, 
                epm, 
                False,
                verbose)
            for j, metric, submodel in pool.imap_unordered(
                    lambda f: f(), tasks
                    ):
                metrics[j] += metric(numFolds)
                validation_data.unpersist()
                training_data.unpersist()
        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        
        bestModel = est.fit(dataset, epm[bestIndex])
        
        return self._copyValues(
            CrossValidatorModel(
                bestModel.metrics
                )
            )
    
    
# Example
if __name__ == '__main__':
    lr = RandomForestClassifier(featuresCol='features', 
                                labelCol='label')
    grid = ParamGridBuiler()\
        .addGrid(lr.maxDepth, [5, 10, 15])\
        .addGrid(lr.numTrees, [50, 100, 150])\
        .build()
    vectorAssembler = vectorAssembler()\
        .setParams(inputCols=in_cols,
                   outputCol='features')
    va_df_train = vectorAssembler.transform(df_train)
    va_df_test = vectorAssembler.transform(df_test)
    evaluator = BinaryClassificationEvaluator(metricName='areaUnderPR')
    
    cv = CustomCrossValidator(
        estimator=lr,
        estimatorParamMaps=grid,
        evaluator=evaluator
        )
    spark.conf.set('spark.sql.execution.arrow.enabled', 'true')
    sparl.sql('set spark.sql.parquet.enableVectorizedReader=true')
    cvModel = cv.fit(va_df_train)
    
    cvModel.avgMetrics
    print('Best Param (Max Depth): ',
          cvModel.bestModel._java_obj.getMaxDepth())
    cvModel.bestModel.save('randomforest.model')

    test_preds = cvModel.bestModel.transform(va_df_test) 
    print('Test areaUnderROC: ',
          evaluator.evaluate(test_preds,
                             {evaluator.metricName: 'areaUnderROC'})
          )
    
        
        
        
        
