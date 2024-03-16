import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
label_csv = '/tcdata/truth.csv'
pred_csv = '/root/tianchi_entry/results.csv'

label_df = pd.read_csv(label_csv,names=['uuid','label'])
pred_df = pd.read_csv(label_csv,names=['uuid','time1','time2','label'])
print('f1 score is:',f1_score(label_df['label'],pred_df['label']))
print('accuracy is:',accuracy_score(label_df['label'],pred_df['label']))
print('recall is:',recall_score(label_df['label'],pred_df['label']))
print('precision is:',precision_score(label_df['label'],pred_df['label']))


line message: 49798,"{""data"":[0.117529675],""shape"":[1]}"
orignal message: 0.117529675
