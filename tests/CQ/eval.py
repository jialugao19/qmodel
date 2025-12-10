#%%
import numpy as np
import pandas as pd

path = "/home/xli/projects/qmodel/train_logs/1122/CQ2_1min_convgru1_clip_1122_233437/eval/eval_40000.feather"

path = "train_logs/1122/CQ2_1min_convgru1_clip_1122_221344/eval/eval_40000.feather"
df = pd.read_feather(path)
df["target2"] = np.sign(df.target) * (df.target/30) ** 2

#%%
q = 0.995
q1 = df.prediction.quantile(q)
q2 = df.prediction.quantile(1-q)

df1 = df[df.prediction > q1].target2
df2 = df[df.prediction < q2].target2

print(q1, q2, len(df1), len(df2), df1.mean(), df2.mean())

