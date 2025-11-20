import pandas as pd

df_train = pd.read_csv("datasets/bigearthnet_s2/train_labels.csv")
class_cols = df_train.columns[1:]

freq = df_train[class_cols].sum().sort_values(ascending=False)
print(freq)
