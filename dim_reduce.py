'''
Vector Dimension Class for testing different methods.
'''

import numpy as np
from sklearn.decomposition import PCA

class DimensionReduction:
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim
        self.type = None

        if self.name == 'PCA':
            self.type = PCA(n_components=dim)
    
    def fit(self, data):
        """
        data format (n_samples, n_features)
        """
        self.type.fit(data)
    
    def transform(self, data):
        self.type.transform(data)