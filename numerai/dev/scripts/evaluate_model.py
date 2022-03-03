#####################
###### imports ######
#####################

import os
import dill
if not os.getcwd().endswith('trading'): os.chdir('../../..') # local machine
assert os.getcwd().endswith('trading'), 'Wrong path!'
import numerapi
from numerai.dev.configs.evaluate_model_cfg import *
import pandas as pd



model_obj = dill.load(open(MODEL_OBJ_FILEPATH, 'rb'))


########################
###### evaluation ######
########################

### run numerai analytics ###

pred_colname = RUN_MODEL_PARAMS['prediction_colname'] if "RUN_MODEL_PARAMS['prediction_colname']" in globals() else 'prediction'

importances = pd.DataFrame({(f, imp) for f, imp in zip(model_obj['final_features'], model_obj['model'].feature_importances_)})\
                .rename(columns={0: 'feature', 1: 'importance'})\
                .sort_values(by='importance', ascending=False)

### corr_coefs for train / val / test ###

train_era_scores = model_obj['df_pred'][model_obj['df_pred'][SPLIT_COLNAME].str.startswith('train')]\
                    .groupby(DATE_COL)\
                    .apply(calc_coef, TARGET, pred_colname)

val_era_scores = model_obj['df_pred'][model_obj['df_pred'][SPLIT_COLNAME].str.startswith('val')]\
                    .groupby(DATE_COL)\
                    .apply(calc_coef, TARGET, pred_colname)
test_era_scores = model_obj['df_pred'][model_obj['df_pred'][SPLIT_COLNAME].str.startswith('test')]\
                    .groupby(DATE_COL)\
                    .apply(calc_coef, TARGET, pred_colname)


### plot the coef scores / print the hit rates ###

train_era_scores = pd.DataFrame(train_era_scores, columns=['era_score']).assign(era='train')
val_era_scores = pd.DataFrame(val_era_scores, columns=['era_score']).assign(era='val')
test_era_scores = pd.DataFrame(test_era_scores, columns=['era_score']).assign(era='test')
era_scores = pd.concat([train_era_scores, val_era_scores, test_era_scores])

fig = px.line(era_scores.reset_index(), x="date", y="era_score", line_group='era')
fig.show()


if __name__ == '__main__':
    print('Done!')