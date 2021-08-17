# Selecting first historical value for each group/generation
'''
open_month - month of loan/loans generation issuance
report_date - date of record for each loan/generation
indicator - values to be selected
first_indicator - first value of indicator in available history
'''
df = df.join(
    df.withColumn(
        'rn', 
        f.row_number().over(
            Window.partitionBy('open_month').orderBy('report_date')
        )
    ).where('rn'==1).drop('rn').select(
        'open_month', 
        'indicator'
    ).withColumnRenamed(
        'open_month',
        'open_month'
    ).withColumnRenamed(
        'indicator',
        'first_indicator'
    ).on='open_month'
)