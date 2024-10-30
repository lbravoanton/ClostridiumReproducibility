from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
import numpy as np

class PIKE():
    def __init__(self, t, n_jobs=10):
        self.t = t
        self.n_jobs = n_jobs

    def __call__(self, X_mz, X_i, Y_mz=None, Y_i=None, distance_TH=1e-6):
        if Y_mz is None and Y_i is None:
            Y_mz = X_mz
            Y_i = X_i

        K = np.zeros((X_mz.shape[0], Y_mz.shape[0]))

        # Precompute distances matrix
        positions_x = X_mz[0,:].reshape(-1, 1)
        positions_y = Y_mz[0,:].reshape(-1, 1)
        distances = pairwise_distances(
                positions_x,
                positions_y,
                metric='sqeuclidean'
            )
        distances = np.exp(-distances / (4 * self.t))
        d = np.where(distances[0] < distance_TH)[0][0]

        def compute_partial_sum(i, x, X_i, Y_i, distances, d):
            intensities_y = Y_i.T[:(i+d), :]
            di = distances[i, :(i+d)].reshape(-1, 1)
            prod = intensities_y * di
            x = np.broadcast_to(x, (np.minimum(i+d,  X_i.shape[1]), X_i.shape[0])).T
            return np.matmul(x, prod)

        # Parallel computation
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_partial_sum)(i, x, X_i, Y_i, distances, d) 
            for i, x in enumerate(X_i.T)
        )

        K = np.sum(results, axis=0) / (4 * self.t * np.pi)
        return K
     
                    
            
            
