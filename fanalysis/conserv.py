import pandas as pd
from fanalysis.pcabis import PCA
%matplotlib inline

df = pd.read_table("fanalysis/tests/pca_data.txt", header=0, index_col=0, delimiter="\t", encoding="utf-8")

m = df.drop(['FINITION'], axis=1)

qsup = [7]
isup = [18]

my_pca = PCA(std_unit=True, n_components=3)

#,quanti_sup=qsup, ind_sup=isup
#, n_components=3

my_pca.fit(m,quanti_sup=qsup, ind_sup=isup)