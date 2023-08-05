# -*- coding: utf-8 -*-

""" pca module
"""

# Author: Olivier Garcia <o.garcia.dev@gmail.com>
# License: BSD 3 clause

import numpy as np
import scipy.stats as stat
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from fanalysis.basebis import Base
from IPython.display import display


class PCA(Base):
    """ Principal Components Analysis (PCA)
    
    This class inherits from the Base class.
    
    PCA performs a Principal Components Analysis, given a table of
    numeric variables ; shape= n_rows x n_columns.

    This implementation only works for dense arrays.

    Parameters
    ----------
    std_unit : bool
       - If True : the data are scaled to unit variance.
       - If False : the data are not scaled to unit variance.
    
    n_components : int, float or None
        Number of components to keep.
        - If n_components is None, keep all the components.
        - If 0 <= n_components < 1, select the number of components such
          that the amount of variance that needs to be explained is
          greater than the percentage specified by n_components.
        - If 1 <= n_components :
            - If n_components is int, select a number of components
              equal to n_components.
            - If n_components is float, select the higher number of
              components lower than n_components.
        
    row_labels : array of strings or None
        - If row_labels is an array of strings : this array provides the
          row labels.
              If the shape of the array doesn't match with the number of
              rows : labels are automatically computed for each row.
        - If row_labels is None : labels are automatically computed for
          each row.
    
    col_labels : array of strings or None
        - If col_labels is an array of strings : this array provides the
          column labels.
              If the shape of the array doesn't match with the number of 
              columns : labels are automatically computed for each
              column.
        - If col_labels is None : labels are automatically computed for
          each column.
    
    stats : bool
        - If stats is true : stats are computed : contributions and
          square cosines for rows and columns.
        - If stats is false : stats are not computed.

    Attributes
    ----------
    n_components_ : int
        The estimated number of components.
    
    row_labels_ : array of strings
        Labels for the rows.
    
    col_labels_ : array of strings
        Labels for the columns.
    
    col_labels_short_ : array of strings
        Short labels for the columns.
        Useful only for MCA, which inherits from Base class. In that
        case, the short labels for the columns at not prefixed by the
        names of the variables.
    
    eig_ : array of float
        A 3 x n_components_ matrix containing all the eigenvalues
        (1st row), the percentage of variance (2nd row) and the
        cumulative percentage of variance (3rd row).
    
    eigen_vectors_ : array of float
        Eigen vectors extracted from the Principal Components Analysis.
    
    row_coord_ : array of float
        A n_rows x n_components_ matrix containing the row coordinates.
    
    col_coord_ : array of float
        A n_columns x n_components_ matrix containing the column
        coordinates.
        
    row_contrib_ : array of float
        A n_rows x n_components_ matrix containing the row
        contributions.
    
    col_contrib_ : array of float
        A n_columns x n_components_ matrix containing the column
        contributions.
    
    row_cos2_ : array of float
        A n_rows x n_components_ matrix containing the row cosines.
    
    col_cos2_ : array of float
        A n_columns x n_components_ matrix containing the column
        cosines.
    
    col_cor_ : array of float
        A n_columns x n_components_ matrix containing the correlations
        between variables (= columns) and axes.
    
    means_ : array of float
        The mean for each variable (= for each column).

    std_ : array of float
        The standard deviation for each variable (= for each column).
    
    ss_col_coord_ : array of float
        The sum of squared of columns coordinates.

    model_ : string
        The model fitted = 'pca'
    """
    def __init__(self, std_unit=True, n_components=None, row_labels=None,
                 col_labels=None, stats=True):
        self.std_unit = std_unit
        self.n_components = n_components
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.stats = stats

    def fit(self, df, ind_sup = None, quanti_sup = None, quali_sup = None):
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

        self.ind_sup = ind_sup
        self.row_coord_sup_ = None

        self.quali_sup = quali_sup
        self.quali_coord_sup_ = None

        self.df = df

        if type(self.quali_sup)==int :
            self.quali_sup = [self.quali_sup]

        if self.quali_sup != None :
            iqls = [i-1 for i in self.quali_sup]
            if self.ind_sup == None :
                if self.quanti_sup == None :
                    dt = df.drop(df.columns[iqls],axis=1)
                else:
                    iqs = [i-1 for i in self.quanti_sup]
                    dt = df.drop(df.columns[iqls],axis=1).drop(df.columns[iqs],axis=1)
            else :
                iis = [i-1 for i in self.ind_sup]
                if self.quanti_sup == None :
                    dt = df.drop(df.columns[iqls],axis=1).drop(df.index[iis], axis=0)
                else:
                    iqs = [i-1 for i in self.quanti_sup]
                    dt = df.drop(df.columns[iqls],axis=1).drop(df.columns[iqs],axis=1).drop(df.index[iis], axis=0)
                    
            
            mean_qual = []
            for j in iqls : 
                for i in np.unique(df[df.columns[j]]) :    
                    mean_qual.append(np.mean(dt[df[df.columns[j]]==i],axis=0))
            
            self.quali_sup_val = pd.DataFrame(mean_qual).values
            self.quali_sup_labels = np.unique(df[df.columns[iqls]])

        ndt = df.drop(df.columns[iqls],axis=1)

        if type(self.quanti_sup)==int:
            self.quanti_sup = [self.quanti_sup]

        if type(self.ind_sup) == int :
            self.ind_sup = [self.ind_sup]

        if self.ind_sup == None :
            self.ind_sup_val = np.array(None)
            self.row_labels_sup = None
            if self.quanti_sup == None :
                self.X = ndt.values
                self.col_labels = ndt.columns
                self.row_labels = ndt.index
                self.Y = np.array(None)
                self.col_labels_sup = None
            else :
                
                ndf = ndt.drop(df.columns[iqs],axis=1)
                self.X = ndf.values
                self.col_labels = ndf.columns
                self.row_labels = ndf.index
                dfqs = ndt[df.columns[iqs]]
                self.Y = dfqs.values
                self.col_labels_sup = dfqs.columns
        else :
            if self.quanti_sup == None :
                ndf = ndt.drop(df.index[iis], axis=0)
                self.X = ndf.values
                self.col_labels = ndf.columns
                self.row_labels = ndf.index
                self.Y = np.array(None)
                self.col_labels_sup = None
                self.ind_sup_val = ndf.values[iis]
                self.row_labels_sup = ndf.index[iis]


            else :
                ndf = ndt.drop(df.columns[iqs],axis=1).drop(df.index[iis], axis=0)
                self.X = ndf.values
                self.col_labels = ndf.columns
                self.row_labels = ndf.index
                dfqs = ndt[df.columns[iqs]].drop(df.index[iis], axis=0)
                self.Y = dfqs.values
                self.col_labels_sup = dfqs.columns
                self.ind_sup_val = ndt.drop(df.columns[iqs],axis=1).values[iis]
                self.row_labels_sup = ndt.index[iis]
        
        self.iis = iis
        self.iqs = iqs
        self.iqls = iqls

        # Compute SVD
        self._compute_svd()
        
        return self

    def transform(self, X, y=None):
        """ Apply the dimensionality reduction on X.

        X is projected on the first axes previous extracted from a
        training set.

        Parameters
        ----------
        X : array of float, shape (n_rows_sup, n_columns)
            New data, where n_rows_sup is the number of supplementary
            row points and n_columns is the number of columns.
            X rows correspond to supplementary row points that are
            projected on the axes.
            X is a table containing numeric values.
        
        y : None
            y is ignored.

        Returns
        -------
        X_new : array of float, shape (n_rows_sup, n_components_)
            X_new : coordinates of the projections of the supplementary
            row points on the axes.
        """
        if self.std_unit:
            Z = (X - self.means_) / self.std_
        else:
            Z = X - self.means_
        
        return Z.dot(self.eigen_vectors_)
        
    def _compute_svd(self):
        """ Compute a Singular Value Decomposition
        
        Then, this function computes :
            n_components_ : number of components.
            eig_ : eigen values.
            eigen_vectors_ : eigen vectors.
            row_coord_ : row coordinates.
            col_coord_ : column coordinates.
            _compute_stats(X) : if stats_ is True.
            row_labels_ : row labels.
            col_labels_ : columns labels.

        Parameters
        ----------
        X : array of float, shape (n_rows, n_columns)
            Training data, where n_rows is the number of rows and
            n_columns is the number of columns.
            X is a table containing numeric values.

        Returns
        -------
        None
        """
        # Initializations
        self.means_ = np.mean(self.X, axis=0).reshape(1, -1)
        if self.std_unit:
            self.std_ = np.std(self.X, axis=0, ddof=0).reshape(1, -1)
            Z = (self.X - self.means_) / self.std_
        else:
            Z = self.X - self.means_        
                
        # SVD
        U, lambdas, V = np.linalg.svd(Z, full_matrices=False)
        
        # Eigen values - first step
        eigen_values = lambdas ** 2 / Z.shape[0]
        eigen_values_percent = 100 * eigen_values / np.sum(eigen_values)
        eigen_values_percent_cumsum = np.cumsum(eigen_values_percent)
        
        # Set n_components_
        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = len(eigen_values)
        elif (self.n_components_ >= 0) and (self.n_components_ < 1):
            i = 0
            threshold = 100 * self.n_components_
            while eigen_values_percent_cumsum[i] < threshold:
                i = i + 1
            self.n_components_ = i
        elif ((self.n_components_ >= 1)
                and (self.n_components_ <= len(eigen_values))
                and (isinstance(self.n_components_, int))):
            self.n_components_ = int(np.trunc(self.n_components_))
        elif ((self.n_components_ >= 1)
                and (self.n_components_ <= len(eigen_values))
                and (isinstance(self.n_components_, float))):
            self.n_components_ = int(np.floor(self.n_components_))
        else:
            self.n_components_ = len(eigen_values)
        
        # Eigen values - second step
        self.eig_ = np.array([eigen_values[:self.n_components_],
                             eigen_values_percent[:self.n_components_],
                             eigen_values_percent_cumsum[:self.n_components_]])
        
        self.eig = np.array([eigen_values,
                             eigen_values_percent,
                             eigen_values_percent_cumsum])
        
        # Eigen vectors
        self.eigen_vectors_ = V.T[:, :self.n_components_]
        
        # Factor coordinates for rows - first step
        row_coord = U * lambdas.reshape(1, -1)
        
        # Factor coordinates for columns - first step
        col_coord = V.T.dot(np.diag(eigen_values**(0.5)))
        self.ss_col_coord_ = (np.sum(col_coord ** 2, axis=1)).reshape(-1, 1)

        # Factor coordinates for rows - second step
        self.row_coord_ = row_coord[:, :self.n_components_]

        # Factor coordinates for columns - second step
        self.col_coord_ = col_coord[:, :self.n_components_]

        # Individu sup
        if self.ind_sup_val.all() != None :
            if self.std_unit:
                I = (self.ind_sup_val - self.means_) / self.std_
            else :
                I =  self.ind_sup_val - self.means_

            row_coord_sup_ = np.zeros([len(self.ind_sup), self.n_components_])
            for i in range(len(self.ind_sup)) :
                for j in range(self.n_components_):
                    row_coord_sup_[i,j] = sum(I[i,:] * self.eigen_vectors_[:,j])
            self.row_coord_sup_ = row_coord_sup_[:, :self.n_components_]



        # Variable sup
        #Inititalisation
        if self.Y.all() != None :   #mieux all ou any ?
            self.means_qsup_ = np.mean(self.Y, axis=0).reshape(1, -1)
            if self.std_unit:
                self.std_qsup_ = np.std(self.Y, axis=0, ddof=0).reshape(1, -1)
                W = (self.Y - self.means_qsup_) / self.std_qsup_  
            else :
                W =  self.Y - self.means_qsup_


            col_coord_sup_ = W.T.dot(row_coord/(np.sqrt(eigen_values))/len(row_coord[:,0]))
            self.col_coord_sup_ = col_coord_sup_[:, :self.n_components_]


        #Quali sup
        if self.quali_sup_val.all() != None :
            if self.std_unit:
                I = (self.quali_sup_val - self.means_) / self.std_
            else :
                I =  self.quali_sup_val - self.means_

        quali_coord_sup_ = np.zeros([self.quali_sup_val.shape[0], self.n_components_])
        for i in range(self.quali_sup_val.shape[0]) :
            for j in range(self.n_components_):
                quali_coord_sup_[i,j] = sum(I[i,:] * self.eigen_vectors_[:,j])

        self.quali_coord_sup_ = quali_coord_sup_[:, :self.n_components_]

        # Compute stats
        if self.stats:
            self._compute_stats(Z)
        
        # Set row labels
        nrows = self.X.shape[0]
        self.row_labels_ = self.row_labels
        self.row_labels_.name = None
        if (self.row_labels_ is None) or (len(self.row_labels_) != nrows):
            self.row_labels_ = ["row " + str(x) for x in np.arange(0, nrows)]

        if self.ind_sup_val.all() != None :
            nrows_sup = len(self.ind_sup)
            self.row_labels_sup_ = self.row_labels_sup
            self.row_labels_sup_.name = None
            if (self.row_labels_sup_ is None) or (len(self.row_labels_sup_) != nrows_sup):
                self.row_labels_sup_ = ["row " + str(x) for x in np.arange(0, nrows_sup)]


        # Set column labels
        ncols = self.X.shape[1]
        self.col_labels_ = self.col_labels
        self.col_labels_.name = None
        if (self.col_labels_ is None) or (len(self.col_labels_) != ncols):
            self.col_labels_ = ["col " + str(x) for x in np.arange(0, ncols)]
        self.col_labels_short_ = self.col_labels_

        if self.Y.all() != None :
            ncols_sup = len(self.quanti_sup)
            self.col_labels_sup_ = self.col_labels_sup
            self.col_labels_sup_.name = None
            if (self.col_labels_sup_ is None) or (len(self.col_labels_sup_) != ncols_sup):
                self.col_labels_sup_ = ["col " + str(x) for x in np.arange(0, ncols_sup)]

        self.model_ = "pca"

    def _compute_stats(self, Z):
        """ Compute statistics : 
                row_contrib_ : row contributions.
                col_contrib_ : column contributions.
                row_cos2_ : row cosines.
                col_cos2_ : column cosines.
                col_cor_ : correlations between variables (= columns)
                and axes.
        
        Parameters
        ----------
        X : array of float, shape (n_rows, n_columns)
            Training data, where n_rows is the number of rows and
            n_columns is the number of columns.
            X is a table containing numeric values.
        Z : array of float, shape (n_rows, n_columns)
            Transformed training data (centered and -optionnaly- scaled
            values)
            where n_rows is the number of rows and n_columns is the
            number of columns.

        Returns
        -------
        None
        """
        # Contributions
        n = Z.shape[0]
        row_contrib = 100 * ((1 / n)
                          * (self.row_coord_ ** 2)
                          * (1 / self.eig_[0].T))
        col_contrib = 100 * (self.col_coord_ ** 2) * (1 / self.eig_[0].T)
        self.row_contrib_ = row_contrib[:, :self.n_components_]
        self.col_contrib_ = col_contrib[:, :self.n_components_]
        
        # Cos2
        row_cos2 = ((self.row_coord_ ** 2)
                    / (np.linalg.norm(Z, axis=1).reshape(-1, 1) ** 2))
        self.row_cos2_ = row_cos2[:, :self.n_components_]
        col_cos2 = (self.col_coord_ ** 2) / self.ss_col_coord_
        self.col_cos2_ = col_cos2[:, :self.n_components_]
        self.ss_col_coord_ = None

        #Cos2 ind sup
        t = self.df.iloc[self.iis].drop(self.df.columns[self.iqls],axis=1).drop(self.df.columns[self.iqs],axis=1)
        if self.std_unit:
            W = (t - self.means_) / self.std_
        else :
            W = (t - self.means_)
        
        ind_sup_cos2 = ((self.row_coord_sup_ ** 2) / (np.linalg.norm(W, axis=1).reshape(-1, 1) ** 2))
        self.ind_sup_cos2 = ind_sup_cos2[:, :self.n_components_]

        #Cos2 var quali sup
        if self.std_unit:
            W = (self.quali_sup_val - self.means_) / self.std_
        else :
            W = (self.quali_sup_val - self.means_) 

        quali_sup_cos2 = ((self.quali_coord_sup_ ** 2) / (np.linalg.norm(W, axis=1).reshape(-1, 1) ** 2))
        self.quali_sup_cos2 = quali_sup_cos2[:, :self.n_components_]

        #Cos2 var quanti sup
        if self.std_unit:
            W = (self.Y - np.mean(self.Y,axis=0).reshape(1,-1)) / np.std(self.Y, axis=0, ddof=0).reshape(1, -1)
        else :
            W = (self.Y - np.mean(self.Y,axis=0).reshape(1,-1)) 
        
        quanti_sup_cos2 = (self.col_coord_sup_ ** 2) / (np.std(W) ** 2)
        self.quanti_sup_cos2 = quanti_sup_cos2[:, :self.n_components_]

        
        # Correlations between variables and axes
        nvars = self.means_.shape[1]
        self.col_cor_ = np.zeros(shape=(nvars, self.n_components_))
        for i in np.arange(0, nvars):
            for j in np.arange(0, self.n_components_):
                self.col_cor_[i, j] = stat.pearsonr(self.X[:,i],
                                                    self.row_coord_[:,j])[0]
        
        # Correlations between variables sup and axes
        if self.Y.all() != None :
            nvars = self.means_qsup_.shape[1]
            self.col_cor_sup_ = np.zeros(shape=(nvars, self.n_components_))
            for i in np.arange(0, nvars):
                for j in np.arange(0, self.n_components_):
                    self.col_cor_sup_[i, j] = stat.pearsonr(self.Y[:,i], self.row_coord_[:,j])[0]

            #self.col_cos2_sup_ = (self.col_coord_sup_ ** 2) / self.ss_col_coord_
        
        
    def correlation_circle(self, num_x_axis, num_y_axis, figsize=None):
        """ Plot the correlation circle
        
        Parameters
        ----------
        num_x_axis : int
            Select the component to plot as x axis.
        
        num_y_axis : int
             Select the component to plot as y axis.
        
        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, aspect="equal")
        ax.add_artist(patches.Circle((0, 0), 1.0, color="black", fill=False))
        
        x_serie = self.col_cor_[:, num_x_axis - 1]
        y_serie = self.col_cor_[:, num_y_axis - 1]
        labels = self.col_labels_
        
        for i in np.arange(0, x_serie.shape[0]):
            x = x_serie[i]
            y = y_serie[i]
            label = labels[i]
            delta = 0.1 if y >= 0 else -0.1
            ax.annotate("", xy=(x, y), xytext=(0, 0),
                        arrowprops={"facecolor": "black",
                                    "width": 0.5,
                                    "headwidth": 4})
            ax.text(x, y + delta, label,
                    horizontalalignment="center", verticalalignment="center",
                    color="blue")
            
        if self.Y.all() != None :
            x_sup_serie = self.col_cor_sup_[:, num_x_axis - 1]
            y_sup_serie = self.col_cor_sup_[:, num_y_axis - 1]
            labels_sup = self.col_labels_sup
            


            for i in np.arange(0, x_sup_serie.shape[0]):
                x = x_sup_serie[i]
                y = y_sup_serie[i]
                label_sup = labels_sup[i]
                delta = 0.1 if y >= 0 else -0.1
                ax.annotate("", xy=(x, y), xytext=(0, 0),
                            arrowprops={"color": "red",
                                        "width": 0.5,
                                        "headwidth": 4})
                ax.text(x, y - delta, label_sup,
                        horizontalalignment="center", verticalalignment="center", 
                        color="red")
            
        
        plt.axvline(x=0, linestyle="--", linewidth=0.5, color="k")
        plt.axhline(y=0, linestyle="--", linewidth=0.5, color="k")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        plt.title("Correlation circle")
        plt.xlabel("Dim " + str(num_x_axis) + " ("
                    + str(np.around(self.eig_[1, num_x_axis - 1], 2)) + "%)")
        plt.ylabel("Dim " + str(num_y_axis) + " ("
                    + str(np.around(self.eig_[1, num_y_axis - 1], 2)) + "%)")
        plt.show()



    def mapping_row(self, num_x_axis, num_y_axis, figsize=None, type = None):
        """ Plot the Factor map for rows only
        
        Parameters
        ----------
        num_x_axis : int
            Select the component to plot as x axis.
        
        num_y_axis : int
             Select the component to plot as y axis.
        
        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)

        x=self.row_coord_[:, num_x_axis - 1]
        y=self.row_coord_[:, num_y_axis - 1]
        

        if type == "cos2" :
            h= 0.1
            color = "black"
            cos2=self.row_cos2_[:, num_x_axis - 1]
            df=pd.DataFrame({'x':x,'y':y,'cos2':cos2})
            sns.relplot(x='x',y='y',hue='cos2',data=df, palette= "rocket_r") 

        elif type == "contrib" :
            h= 0.1
            color = "black"
            contrib = self.row_contrib_[:, num_x_axis - 1]
            df=pd.DataFrame({'x':x,'y':y,'contrib':contrib})
            sns.relplot(x='x',y='y',hue='contrib',data=df, palette= "rocket_r") 


        elif type == "sup" :
            h= 0.1
            color = "black"
            iqls = [i-1 for i in self.quali_sup]
            iis = [i-1 for i in self.ind_sup]
            sup = self.df[self.df.columns[iqls]].drop(self.df.index[iis], axis=0)
            edf=pd.DataFrame({'x':x,'y':y})
            sup.index = range(edf.shape[0])
            edf = edf.join(sup)
            edf.columns = ["x","y","sup"]
            sns.relplot(x='x',y='y',hue='sup',data=edf) 



        else :
            h = 0
            color = "black"
            plt.scatter(self.row_coord_[:, num_x_axis - 1],
                    self.row_coord_[:, num_y_axis - 1],
                    marker=".", color="white")


        
        for i in np.arange(0, self.row_coord_.shape[0]):
            plt.text(self.row_coord_[i, num_x_axis - 1],
                     self.row_coord_[i, num_y_axis - 1] + h,
                     self.row_labels_[i],
                     horizontalalignment="center", verticalalignment="center",
                     color=color)

        if self.ind_sup_val.all() != None :
            for i in np.arange(0, self.row_coord_sup_.shape[0]):
                plt.text(self.row_coord_sup_[i, num_x_axis - 1],
                        self.row_coord_sup_[i, num_y_axis - 1],
                        self.row_labels_sup_[i],
                        horizontalalignment="center", verticalalignment="center",
                        color="red")
        
        if self.quali_sup_val.all() != None :
            for i in np.arange(0, self.quali_coord_sup_.shape[0]):
                plt.text(self.quali_coord_sup_[i, num_x_axis - 1],
                        self.quali_coord_sup_[i, num_y_axis - 1],
                        self.quali_sup_labels[i],
                        horizontalalignment="center", verticalalignment="center",
                        color="blue")
        

        plt.title("Factor map for rows")
        plt.xlabel("Dim " + str(num_x_axis) + " ("
                    + str(np.around(self.eig_[1, num_x_axis - 1], 2)) + "%)")
        plt.ylabel("Dim " + str(num_y_axis) + " ("
                    + str(np.around(self.eig_[1, num_y_axis - 1], 2)) + "%)")
        plt.axvline(x=0, linestyle="--", linewidth=0.5, color="k")
        plt.axhline(y=0, linestyle="--", linewidth=0.5, color="k")
        plt.show()




    def show_eig(self):
        L = [ 'Comp ' + str(i+1) for i in range(len(self.eig[0]))]
        df=pd.DataFrame(self.eig, index = ['Eigenvalue','Percentage of variance (%)','Cumulative percentage of variance (%)'], columns=L)
        df = df.style.set_caption("Eigen values").set_table_styles([{
            'selector': 'caption',
            'props': [
                ('font-size', '20px'),
                ('font-weight', 'bold'),
                ('text-align', 'center')
            ]
        }])
        return(df)


    def show_eigen_vectors(self):
        L = [ 'Comp ' + str(i+1) for i in range(len(self.eig_[0]))]
        df=pd.DataFrame(self.eigen_vectors_, columns=L, index=self.col_labels_)
        df = df.style.set_caption("Eigen vectors").set_table_styles([{
            'selector': 'caption',
            'props': [
                ('font-size', '20px'),
                ('font-weight', 'bold'),
                ('text-align', 'center')
            ]
        }])
        return(df)

    def show_row(self):
        if self.stats :
            df = pd.DataFrame()
            row_name = []

            for i in range(self.row_coord_.shape[1]):
                comp = 'Comp ' + str(i+1)
                coord = 'Comp ' + str(i+1) + " (coord)"
                contrib = 'Comp ' + str(i+1) + " (contrib)"
                cos = 'Comp ' + str(i+1) + " (cos2)"
                #df[comp] = " | " 
                df[coord] = np.round(self.row_coord_[:,i],3)
                df[contrib] = np.round(self.row_contrib_[:,i],3)
                df[cos] = np.round(self.row_cos2_[:,i],3)
                #df[comp] = np.round(self.row_coord_[:,i],3)
                row_name = row_name + ["Comp" + str(i+1) + " coord","contrib","cos2"]

            df.index = self.row_labels_
            df.columns = row_name

        else :
            L = [ 'Comp ' + str(i+1) for i in range(len(self.eig_[0]))]
            df=pd.DataFrame(self.row_coord_, columns = L, index = self.row_labels_)

        df = df.style.set_caption("Individuals").set_table_styles([{
            'selector': 'caption',
            'props': [
                ('font-size', '20px'),
                ('font-weight', 'bold'),
                ('text-align', 'center')
            ]
        }])
        return(df)

    def show_row_sup(self):
        if self.stats :
            df = pd.DataFrame()
            row_name = []

            for i in range(self.row_coord_sup_.shape[1]):
                comp = 'Comp ' + str(i+1)
                coord = 'Comp ' + str(i+1) + " (coord)"
                cos = 'Comp ' + str(i+1) + " (cos2)"
                #df[comp] = " | " 
                df[coord] = np.round(self.row_coord_sup_[:,i],3)
                df[cos] = np.round(self.ind_sup_cos2[:,i],3)
                #df[comp] = np.round(self.row_coord_[:,i],3)
                row_name = row_name + ["Comp" + str(i+1) + " coord", "cos2"]

            df.index = self.row_labels_sup_
            df.columns = row_name

        else :
            L = [ 'Comp ' + str(i+1) for i in range(len(self.eig_[0]))]
            df=pd.DataFrame(self.row_coord_sup_, columns = L, index = self.row_labels_sup_)
            
        df = df.style.set_caption("Supplementary individual").set_table_styles([{
            'selector': 'caption',
            'props': [
                ('font-size', '20px'),
                ('font-weight', 'bold'),
                ('text-align', 'center')
            ]
        }])
        return(df)    

    def show_col(self):
        if self.stats :
            df = pd.DataFrame()
            col_name = []

            for i in range(self.col_coord_.shape[1]):
                comp = 'Comp ' + str(i+1)
                coord = 'Comp ' + str(i+1) + " (coord)"
                contrib = 'Comp ' + str(i+1) + " (contrib)"
                cos = 'Comp ' + str(i+1) + " (cos2)"
                #df[comp] = " | " 
                df[coord] = np.round(self.col_coord_[:,i],3)
                df[contrib] = np.round(self.col_contrib_[:,i],3)
                df[cos] = np.round(self.col_cos2_[:,i],3)
                #df[comp] = np.round(self.col_coord_[:,i],3)
                col_name = col_name + ["Comp" + str(i+1) + " coord","contrib","cos2"]

            df.index = self.col_labels_
            df.columns = col_name

        else :
            L = [ 'Comp ' + str(i+1) for i in range(len(self.eig_[0]))]
            df=pd.DataFrame(self.row_coord_, columns = L, index = self.row_labels_)

        df = df.style.set_caption("Individuals").set_table_styles([{
            'selector': 'caption',
            'props': [
                ('font-size', '20px'),
                ('font-weight', 'bold'),
                ('text-align', 'center')
            ]
        }])
        return(df)
    

    def show_col_sup(self):
        if self.stats :
            df = pd.DataFrame()
            row_name = []

            for i in range(self.col_coord_sup_.shape[1]):
                coord = 'Comp ' + str(i+1) + " (coord)"
                cos = 'Comp ' + str(i+1) + " (cos2)"
                df[coord] = np.round(self.col_coord_sup_[:,i],3)
                df[cos] = np.round(self.quanti_sup_cos2[:,i],3)
                row_name = row_name + ["Comp" + str(i+1) + " coord", "cos2"]

            df.index = self.col_labels_sup_
            df.columns = row_name

        else :
            L = [ 'Comp ' + str(i+1) for i in range(len(self.eig_[0]))]
            df=pd.DataFrame(self.col_coord_sup_, columns = L, index = self.col_labels_sup_)
        
        df = df.style.set_caption("Supplementary continuous variable").set_table_styles([{
            'selector': 'caption',
            'props': [
                ('font-size', '20px'),
                ('font-weight', 'bold'),
                ('text-align', 'center')
            ]
        }])
        return(df)
    

    def show_qual_sup(self):
        if self.stats :
            df = pd.DataFrame()
            row_name = []

            for i in range(self.quali_coord_sup_.shape[1]):
                coord = 'Comp ' + str(i+1) + " (coord)"
                cos = 'Comp ' + str(i+1) + " (cos2)" 
                df[coord] = np.round(self.quali_coord_sup_[:,i],3)
                df[cos] = np.round(self.quali_sup_cos2[:,i],3)
                row_name = row_name + ["Comp" + str(i+1) + " coord", "cos2"]

            df.index = self.quali_sup_labels
            df.columns = row_name

        else :
            L = [ 'Comp ' + str(i+1) for i in range(len(self.eig_[0]))]
            df=pd.DataFrame(self.quali_coord_sup_, columns = L, index = self.quali_sup_labels)
            
        df = df.style.set_caption("Supplementary categories").set_table_styles([{
            'selector': 'caption',
            'props': [
                ('font-size', '20px'),
                ('font-weight', 'bold'),
                ('text-align', 'center')
            ]
        }])
        return(df)



    def summary(self):
        eig = self.show_eig().format(precision = 3)
        row = self.show_row().format(precision = 3)
        col = self.show_col().format(precision = 3)
        if self.quanti_sup != None :
            sup_col = self.show_col_sup().format(precision = 3)
        else :
            sup_col = ""
        if self.ind_sup != None :
            sup_row = self.show_row_sup().format(precision = 3)
        else : 
            sup_row = ""
        if self.quali_sup != None :
            sup_qual = self.show_qual_sup().format(precision = 3)
        else :
            sup_qual = ""
        display(eig,row,sup_row,col,sup_col,sup_qual)

