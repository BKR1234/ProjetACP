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