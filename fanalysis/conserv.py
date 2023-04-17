def fit(self, X, y=None, quanti_sup = np.array(None), ):
        """ Fit the model to X

        Parameters
        ----------
        X : array of float, shape (n_rows, n_columns)
            Training data, where n_rows in the number of rows and
            n_columns is the number of columns
            (= the number of variables).
            X is a table containing numeric values.
        
        y : None
            y is ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Stats initialization
        self.row_contrib_ = None
        self.col_contrib_ = None
        self.row_cos2_ = None
        self.col_cos2_ = None

        self.quanti_sup = quanti_sup
        self.col_coord_sup_ = None
        
        # Compute SVD
        self._compute_svd(X,quanti_sup)
        
        return self








   self.Y = dfqs.iloc[:,:].values


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