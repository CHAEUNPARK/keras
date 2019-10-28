import pandas as pd
from numpy import array
df = pd.read_csv('./test.csv')
# df = df.fillna(df.mean())

df = df.drop('Time', axis=1)
df = df.values

# df['Time'] = pd.to_datetime(df['Time'])
# df = df.set_index('Time')

from sklearn.decomposition import PCA

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_idx = i + n_steps
        if end_idx > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_idx, :], sequence[end_idx, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


n_steps = 24
x, y = split_sequence(df, n_steps)
#
# x = x.reshape(len())

from xgboost import XGBClassifier
model = XGBClassifier(random_state=0)
# model.fit(x, y)
