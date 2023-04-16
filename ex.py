
import pandas as pd
from fanalysis.pca import PCA

df = pd.read_table("fanalysis/tests/pca_data.txt", header=0, index_col=0, delimiter="\t", encoding="utf-8")

print(df)