import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

df = pd.read_csv('D:/wsd_predictions.tsv', sep='\t', na_values=np.nan, keep_default_na=False)

y_true = df['good_class'].to_list()
y_pred = df['predicted_class'].to_list()

f1_micro = f1_score(y_true, y_pred, average='micro')

print(f1_micro)
