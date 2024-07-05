import numpy as np
import scipy
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import torch


class GaussianMixtureModel:
    '''
    A class for Gaussian Mixture Model
    '''

    def __init__(self, samples: np.ndarray, n_components=5, use_pytorch: bool = False, torch_device="cpu", dtype=torch.float64) -> None:
        if use_pytorch:
            self.samples = torch.as_tensor(samples, device=torch_device, dtype=dtype)
            self.np_samples = self.samples.clone().detach().cpu().numpy()
        else:
            self.samples = samples
            self.np_samples = np.asarray(samples)
        self.gmm = GaussianMixture(n_components=n_components)
        self.n_components = n_components
        self.gmm.fit(self.np_samples)
        self.use_pytorch = use_pytorch
        self.torch_device = torch_device
        self.dtype = dtype
        self.set_up_tensors()

    def set_up_tensors(self):
        n_dim = self.samples.shape
        self.torch_means = torch.tensor(self.gmm.means_, dtype=self.dtype, device=self.torch_device)
        self.torch_weights = torch.tensor(self.gmm.weights_, dtype=self.dtype, device=self.torch_device)
        self.torch_covariances = torch.tensor(self.gmm.covariances_, dtype=self.dtype, device=self.torch_device)
        self.pi_c = torch.tensor((2*torch.pi)**(n_dim/2), device=self.torch_device, dtype=self.dtype)
        self.inv_covariance = torch.linalg.inv(self.torch_covariances).unsqueeze(dim=0)
        self.det_covariance = torch.linalg.det(self.torch_covariances)

    def select_n_components(self, ns, set_gmm: bool = True, progress_bar=False, record_history = True):
        best_score = -1
        best_n = -1
        best_gmm = self.gmm
        self.n_components_history = np.empty(0)
        self.s_score_history = np.empty(0)
        if progress_bar:
            from tqdm import tqdm
        else:
            def tqdm(iterable):
                return iterable
        for n in tqdm(ns):
            gmm_tmp = GaussianMixture(n_components=n)
            gmm_tmp.fit(self.np_samples)
            labels = gmm_tmp.predict(self.np_samples)
            s_score = silhouette_score(self.np_samples, labels)
            if record_history:
                self.n_components_history = np.append(self.n_components_history,n)
                self.s_score_history = np.append(self.s_score_history, s_score)
            if s_score > best_score:
                best_score = s_score
                best_gmm = gmm_tmp
                best_n = n
        if set_gmm:
            self.gmm = best_gmm
            self.set_up_tensors()
        return best_n, best_score

    def __torchpdf(self, x: torch.Tensor):

        diff = x.unsqueeze(dim=1) - self.torch_means
        # diff: (n_samples,n_components,n_dim). inv_covariance:(n_components, n_dim,n_dim)
        # originaly it is a 1x3 x 3x3 x 3x1 -> 1x1.
        # Now we need (n_samples,n_components,1,n_dim)x(n_components, n_dim, n_dim)->(n_samples,n_components,1,n_dim)
        # and then (n_samples,n_components,1,n_dim)x(n_samples,n_components,n_dim,1)->(n_samples,n_components,1,1)
        # diff result: n_samples,
        diff = diff.unsqueeze(dim=-2)
        diff_t = torch.transpose(diff, -2, -1)
        m_dist = diff @ self.inv_covariance @ diff_t  # diff: (n_samples,n_components,n_dim). inv_covariance:(n_components, n_dim,n_dim)
        m_dist = m_dist.squeeze()
        densities = torch.exp(-0.5*m_dist)/(self.pi_c*torch.sqrt(self.det_covariance))
        densities_samples = torch.matmul(densities,self.torch_weights.unsqueeze(dim=1)).squeeze()
        return densities_samples

    def pdf(self, x: np.ndarray | torch.Tensor):
        if self.use_pytorch:
            return self.__torchpdf(x)
        try:
            x = np.reshape(x, (1, 3))
        except:
            raise ValueError(f"x should only contain 3 elements, x is now {x} with shape {np.shape(x)}")

        return np.exp(self.gmm.score_samples(x)).item()
