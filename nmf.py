"""
Non-negative Matrix Factorization algorithms.
Many can be unified under a single framework based on multiplicative updates.
Implementation is inspired by NMFlib by Graham Grindlay
"""
import numpy as np
import ipdb



class NMF():
    def __init__(self, rank, update_func="kl", iterations=100,
                 threshold=None, norm_D=1, norm_H=0,
                 update_D=True, update_H=True):

        self.rank = rank
        self.update = getattr(self, update_func + "_updates")
        self.iterations = iterations
        self.threshold = threshold
        self.norm_D = norm_D
        self.norm_H = norm_H
        self.update_D = update_D
        self.update_H = update_H
        self.update_func = update_func

    """
  Compute divergence between reconstruction and original
  """

    def compute_error(self, X, D, H):
        eps = np.spacing(1)
        R = np.dot(D, H)
        if self.update_func == "kl" or "kl_mix":
            err = np.sum(np.multiply(X, np.log((X + eps) / (R + eps))) - X + R)
        elif self.update_func == "is":
            err = np.sum((X + eps) / (R + eps) - np.log((X + eps) / (R + eps)) - 1)
        elif self.update_func == "eucl":
            err = np.sum((X - R) ** 2)
        else:
            raise XalueError("Unknown metric" + self.update_func)
        return err

    """
  Normalize D and/or H depending on initialization options
  """

    def normalize_D(self, D):
        if self.norm_D == 1:
            D = D / (np.sum(D, axis=0))
        if self.norm_D == 2:
            D = D / (np.sqrt(np.sum(D ** 2, axis=0)))
        return D

    def normalize_H(self, H):
        if self.norm_H == 1:
            H = H / (np.sum(H, axis=1))
        if self.norm_H == 2:
            H = H / (np.sqrt(np.sum(H ** 2, axis=1)))
        return H

    def normalize(self, D, H):
        D = self.normalize_D(D)
        H = self.normalize_H(H)
        return [D, H]

    """
  Initialize and compute multiplicative updates iterations
  """

    def process(self, X, lam, D0=None, H0=None):
        eps = np.spacing(1)
        D = D0 if D0 is not None else np.random.rand(X.shape[0], self.rank) + eps
        H = H0 if H0 is not None else np.random.rand(self.rank, X.shape[1]) + eps
        self.ones_X = np.ones(X.shape)
        self.ones_XX = np.ones(X.shape[0])
        [D,H] = self.normalize(D,H)
        for i in range(self.iterations):
            [X, D, H] = self.update(X, D, H,lam)
            err = self.compute_error(X, D, H)
            print(i,err)
            if self.threshold is not None:
                err = self.compute_error(X, D, H)
                if err <= self.threshold:
                    print("The error is below " + str(self.threshold))
                    return [D, H, err]
        return [D, H, err]

    def process_mix(self, X, lam_s, lam_n, D_n, N_s, N_n):
        eps = np.spacing(1)
        D_s= np.random.rand(X.shape[0], N_s) + eps
        H_s = np.random.rand(N_s, X.shape[1]) + eps
        H_n = np.random.rand(N_n, X.shape[1]) + eps
        self.ones_X = np.ones(X.shape)
        self.ones_XX = np.ones(X.shape[0])
        D=np.concatenate((D_n,D_s),axis=1)
        H=np.concatenate((H_n,H_s),axis=0)
        [D,H] = self.normalize(D,H)
        for i in range(self.iterations):
            [X, D, H] = self.update(X, D, H, lam_s, lam_n)
            err = self.compute_error(X, D, H)
            print(i,err)
            if self.threshold is not None:
                err = self.compute_error(X, D, H)
                if err <= self.threshold:
                    print("The error is below " + str(self.threshold))
                    return [D, H, err]
        return [D, H, err]

    """
  Optimize Kullback-Leibler divergence
  """

    def kl_updates(self, X, D, H, lam):
        eps = np.spacing(1)
        if self.update_D:
            R = np.dot(D, H)
            #RH=np.dot(R,H.T)
            #XH=np.dot(X,H.T)
            #D *= (XH+D*(np.dot(self.ones_XX,(np.dot(R,H.T)*D))))/(np.dot(R,H.T)+D*(np.dot(self.ones_XX,(np.dot(X,H.T)*D))))
            #numerator=XH+D*np.dot(self.ones_XX,RH*D)
            #denominator=RH+D*np.dot(self.ones_XX, XH*D)
            #D *= numerator/denominator
            D *= np.dot(np.divide(X, R + eps), H.T) / (np.dot(self.ones_X, H.T) + eps)
            D = self.normalize_D(D)
        if self.update_H:
            R = np.dot(D, H)
            #ipdb.set_trace()
            #H *= np.dot(D.T, X) / (np.dot(D.T, R) + eps + lam)
            H *= np.dot(D.T, np.divide(X, R + eps)) / (np.dot(D.T, self.ones_X) + eps + lam)
            #H = self.normalize_H(H)
        return [X, D, H]


    def kl_mix_updates(self, X, D, H, lam_s, lam_n):
        eps = np.spacing(1)
        D_s=D[:,-self.rank:]
        D_n=D[:,:-self.rank]
        H_s=H[-self.rank:,:]
        H_n=H[:-self.rank,:]

        R=np.dot(D,H)
        # updating mixture dictionary D_s
        D_s *= (np.dot(X, H_s.T) + D_s * (np.dot(self.ones_XX, (np.dot(R, H_s.T) * D_s)))) / (
        np.dot(R, H_s.T) + D_s * (np.dot(self.ones_XX,(np.dot(X,H_s.T)*D_s))))
        #D_s *= np.dot(np.divide(X, R + eps), H.T) / (np.dot(self.ones_X, H.T) + eps)

        D_s = self.normalize_D(D_s)

        # update total dictionary
        D = np.concatenate((D_n, D_s), axis=1)
        R=np.dot(D,H)

        # updating mixture activation
        #H_s *= np.dot(D_s.T, X) / (np.dot(D_s.T,R) + eps + lam_s)
        H_s *= np.dot(D_s.T, np.divide(X, R + eps)) / (np.dot(D_s.T, self.ones_X) + eps + lam_s)

        # updating noise activation
        #H_n *= np.dot(D_n.T, X) / (np.dot(D_n.T,R) + eps + lam_n)
        H_n *= np.dot(D_n.T, np.divide(X, R + eps)) / (np.dot(D_n.T, self.ones_X) + eps + lam_n)

        # update total activation
        H = np.concatenate((H_n, H_s), axis=0)
        return [X, D, H]