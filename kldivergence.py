import numpy as np
import scipy as sp
from tools import decorators
import torch
from typing import Callable


class DifferentiableKLDivergence:
    def __init__(self, p: Callable = None, q: Callable = None,
                 p_center: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 q_center: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 enable_move_pdf_to_center: bool = False,
                 initial_t: torch.Tensor = torch.tensor([0.0, 0.0, 0.0], requires_grad=True),
                 init_alpha: torch.Tensor = torch.tensor([0.0], requires_grad=True),
                 init_beta: torch.Tensor = torch.tensor([0.0], requires_grad=True),
                 init_gamma: torch.Tensor = torch.tensor([0.0], requires_grad=True),
                 s: int = 6,    # the s for phi_ijk = sum_{i+j+k <= s} (sin(...)*sin(...)*sin(...))
                 step_size=0.01,
                 interval: torch.Tensor = torch.tensor([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]),
                 torch_device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 enable_rotation: bool = True, enable_translation: bool = True,
                 integration_num_steps: int = 50,
                 enable_monte_carlo_integration: bool = False,
                 randomize_trapezoidal_rule_integration_eval_points: bool = False) -> None:
        self.origin_p = p
        self.origin_q = q
        self.t = torch.nn.Parameter(initial_t)
        self.alpha = torch.nn.Parameter(init_alpha)
        self.beta = torch.nn.Parameter(init_beta)
        self.gamma = torch.nn.Parameter(init_gamma)
        self.s = s
        self.torch_device = torch_device
        self.dtype = dtype
        self.R = self.get_rotation_matrix()
        self.step_size = step_size
        self.interval = interval
        self.__check_requires_grad()
        self.enable_rotation = enable_rotation
        self.enable_translation = enable_translation
        self.__define_optimize_upper_level_one_step()
        self.__integration_num_steps = integration_num_steps
        self.enable_move_pdf_to_center = enable_move_pdf_to_center
        self.p_center = p_center
        self.q_center = q_center
        self.__move_pdf_to_center()
        self.__enable_monte_carlo_integration = enable_monte_carlo_integration
        self.__randomize_trapezoidal_rule_integration_eval_points = randomize_trapezoidal_rule_integration_eval_points
        self.__set_integration_method()
        self.calculate_KL_divergence()

    def __check_requires_grad(self):
        if not all([self.t.requires_grad, self.alpha.requires_grad, self.beta.requires_grad, self.gamma.requires_grad]):
            raise ValueError("t, alpha, beta, gamma should be set to requires_grad=True")

    def __define_optimize_upper_level_one_step(self):
        if self.enable_rotation == False and self.enable_translation == False:
            raise ValueError("optimize_rotation and optimize_translation can not be False at the same time")
        elif self.enable_rotation == False:
            self.__optimize_upper_level_one_step = self.__optimize_upper_level_one_step_translate
        elif self.enable_translation == False:
            self.__optimize_upper_level_one_step = self.__optimize_upper_level_one_step_rotation
        else:
            self.__optimize_upper_level_one_step = self.__optimize_upper_level_one_step_both

        self.__optimizer, self.__closure = self.__optimize_upper_level_one_step()

    def set_optimize_upper_level_one_step(self, enable_rotation=None, enable_translation=None):
        if not enable_rotation == None:
            self.enable_rotation = enable_rotation
        if not enable_translation == None:
            self.enable_translation = enable_translation
        self.__define_optimize_upper_level_one_step()

    def __move_pdf_to_center(self):
        if self.enable_move_pdf_to_center:
            self.p_center = self.p_center.to(dtype=self.dtype, device=self.torch_device)
            self.q_center = self.q_center.to(dtype=self.dtype, device=self.torch_device)

            def p(x):
                return self.origin_p(x + self.p_center)

            def q(x):
                return self.origin_q(x + self.q_center)
            self.p = p
            self.q = q
            self.get_transformed_points = self.get_transformed_points_rotate_along_global_coord
        else:

            def p(x):
                return self.origin_p(x)

            def q(x):
                return self.origin_q(x)
            self.p = p
            self.q = q
            self.get_transformed_points = self.get_transformed_points_rotate_along_point_cloud_center

    def get_transformed_points_rotate_along_point_cloud_center(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Call this method after setting correct self.t and self.R
        '''
        return (self.R.unsqueeze(dim=0).unsqueeze(dim=0) @ (x-self.p_center).unsqueeze(dim=-1)).squeeze() + self.p_center + self.t

    def get_transformed_points_rotate_along_global_coord(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Call this method after setting correct self.t and self.R
        '''
        return (self.R.unsqueeze(dim=0).unsqueeze(dim=0) @ x.unsqueeze(dim=-1)).squeeze() + self.t

    def get_t(self):
        return self.t.clone().detach()

    def get_rotation(self):
        return self.alpha.clone().detach(), self.beta.clone().detach(), self.gamma.clone().detach()

    def calculate_KL_divergence(self):
        self.R = self.get_rotation_matrix()
        self.kldivergence = self.__calc_integration(self.__integrand)
        return self.kldivergence

    def get_KL_divergence(self):
        return self.kldivergence.clone().detach()

    def generate_meshgrid_eval_points(self) -> torch.Tensor:
        x = torch.linspace(self.interval[0][0], self.interval[0][1], self.__integration_num_steps+1, device=self.torch_device, dtype=self.dtype)
        y = torch.linspace(self.interval[1][0], self.interval[1][1], self.__integration_num_steps+1, device=self.torch_device, dtype=self.dtype)
        z = torch.linspace(self.interval[2][0], self.interval[2][1], self.__integration_num_steps+1, device=self.torch_device, dtype=self.dtype)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        eval_points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        return eval_points

    def generate_random_eval_points(self) -> torch.Tensor:
        random_points = torch.rand(size=(self.__integration_num_steps, 3), dtype=self.dtype, device=self.torch_device)
        scale_x = self.interval[0][1] - self.interval[0][0]
        scale_y = self.interval[1][1] - self.interval[1][0]
        scale_z = self.interval[2][1] - self.interval[2][0]
        scales = torch.tensor([scale_x, scale_y, scale_z], dtype=self.dtype, device=self.torch_device)
        offsets = torch.tensor([self.interval[0][0], self.interval[1][0], self.interval[2][0]], dtype=self.dtype, device=self.torch_device)
        random_points = random_points * scales + offsets
        return random_points

    def set_integration_num_steps(self, integration_num_steps: int):
        self.__integration_num_steps = integration_num_steps
        self.__integration_eval_points = self.generate_meshgrid_eval_points()

    def __set_integration_method(self):
        if self.__enable_monte_carlo_integration:
            self.__calc_integration = self.__monte_carlo_integration
        else:
            if self.__randomize_trapezoidal_rule_integration_eval_points:
                raise ValueError("randomize_trapezoidal_rule_integration_eval_points is buggy now, don't set to True!")
                self.__integration_eval_points = self.generate_random_eval_points()
            else:
                self.__integration_eval_points = self.generate_meshgrid_eval_points()
            self.__calc_integration = self.__trapezoidal_rule_integration

    def get_rotation_matrix(self) -> torch.Tensor:

        Rx = torch.stack([
            torch.stack([torch.tensor(1.0, dtype=self.dtype, device=self.torch_device), torch.tensor(0.0, dtype=self.dtype,
                        device=self.torch_device), torch.tensor(0.0, dtype=self.dtype, device=self.torch_device)]),
            torch.stack([torch.tensor(0.0, dtype=self.dtype, device=self.torch_device), torch.cos(self.alpha).squeeze(), -torch.sin(self.alpha).squeeze()]),
            torch.stack([torch.tensor(0.0, dtype=self.dtype, device=self.torch_device), torch.sin(self.alpha).squeeze(), torch.cos(self.alpha).squeeze()])
        ])

        Ry = torch.stack([
            torch.stack([torch.cos(self.beta).squeeze(), torch.tensor(0.0, dtype=self.dtype, device=self.torch_device), torch.sin(self.beta).squeeze()]),
            torch.stack([torch.tensor(0.0, dtype=self.dtype, device=self.torch_device), torch.tensor(1.0, dtype=self.dtype,
                        device=self.torch_device), torch.tensor(0.0, dtype=self.dtype, device=self.torch_device)]),
            torch.stack([-torch.sin(self.beta).squeeze(), torch.tensor(0.0, dtype=self.dtype, device=self.torch_device), torch.cos(self.beta).squeeze()])
        ])

        Rz = torch.stack([
            torch.stack([torch.cos(self.gamma).squeeze(), -torch.sin(self.gamma).squeeze(), torch.tensor(0.0, dtype=self.dtype, device=self.torch_device)]),
            torch.stack([torch.sin(self.gamma).squeeze(), torch.cos(self.gamma).squeeze(), torch.tensor(0.0, dtype=self.dtype, device=self.torch_device)]),
            torch.stack([torch.tensor(0.0, dtype=self.dtype, device=self.torch_device), torch.tensor(
                0.0, dtype=self.dtype, device=self.torch_device), torch.tensor(1.0, dtype=self.dtype, device=self.torch_device)])
        ])

        # Combined rotation matrix, NOTE: xyz Euler angle!
        R = Rz @ Ry @ Rx
        return R

    def get_transformed_points(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Call this method after setting correct self.t and self.R
        '''
        return (self.R.unsqueeze(dim=0).unsqueeze(dim=0) @ (x+self.t).unsqueeze(dim=-1)).squeeze()  # self.R.T: (3,3); (x-self.t): (n_x,3) -> return: (n_x,3)

    def __integrand(self, x: torch.Tensor) -> torch.Tensor:
        p_x = self.p(x)
        tiny_tensor = torch.tensor(1e-16, device=self.torch_device, requires_grad=False)
        return p_x*torch.log(torch.maximum(p_x/(torch.maximum(self.q(self.get_transformed_points(x)), tiny_tensor)), tiny_tensor))



    def __monte_carlo_integration(self, integrand) -> torch.Tensor:
        fxyz = integrand(self.generate_random_eval_points())
        volume = (self.interval[0][1] - self.interval[0][0])*(self.interval[1][1] - self.interval[1][0])*(self.interval[2][1] - self.interval[2][0])
        return torch.sum(fxyz, dim=0) * volume / self.__integration_num_steps

    def __trapezoidal_rule_integration(self, integrand) -> torch.Tensor:
        dx = (self.interval[0][1] - self.interval[0][0])/self.__integration_num_steps
        dy = (self.interval[1][1] - self.interval[1][0])/self.__integration_num_steps
        dz = (self.interval[2][1] - self.interval[2][0])/self.__integration_num_steps
        fxyz = integrand(self.__integration_eval_points)
        return torch.sum(fxyz, dim=0) * dx * dy * dz

    def backward(self, differentiable_tensor: torch.Tensor) -> None:
        differentiable_tensor.backward(retain_graph=True)

    def __optimize_upper_level_one_step_both(self):
        params = [self.t, self.alpha, self.beta, self.gamma]  # Ensure these are tensors with requires_grad=True
        optimizer = torch.optim.LBFGS(params, lr=self.step_size, tolerance_change=1e-20, tolerance_grad=1e-20, max_iter=10)

        def closure():
            optimizer.zero_grad()  # Clear gradients
            self.R = self.get_rotation_matrix()
            one_tensor = torch.tensor(1, dtype=self.dtype, device=self.torch_device)
            kld = self.calculate_KL_divergence()
            J = kld  + torch.abs(torch.maximum(torch.max(self.t),one_tensor)) + torch.abs(torch.minimum(torch.min(self.t),-one_tensor))
            J.backward()
            return J

        # Perform the optimization step
        return optimizer, closure

    def __optimize_upper_level_one_step_rotation(self):
        params = [self.alpha, self.beta, self.gamma]  # Ensure these are tensors with requires_grad=True
        optimizer = torch.optim.LBFGS(params, lr=self.step_size, tolerance_change=1e-20, tolerance_grad=1e-20, max_iter=10)

        def closure():
            optimizer.zero_grad()  # Clear gradients
            self.R = self.get_rotation_matrix()
            one_tensor = torch.tensor(1, dtype=self.dtype, device=self.torch_device)
            kld = self.calculate_KL_divergence()
            J = kld + torch.pow(torch.maximum(torch.max(self.t),one_tensor),4) + torch.pow(torch.minimum(torch.min(self.t),-one_tensor),4)
            J.backward()
            return J

        # Perform the optimization step
        return optimizer, closure

    def __optimize_upper_level_one_step_translate(self):
        params = [self.t, self.alpha, self.beta, self.gamma]  # Ensure these are tensors with requires_grad=True
        optimizer = torch.optim.LBFGS(params, lr=self.step_size, tolerance_change=1e-20, tolerance_grad=1e-20, max_iter=10)

        def closure():
            optimizer.zero_grad()  # Clear gradients
            self.R = self.get_rotation_matrix()
            one_tensor = torch.tensor(1, dtype=self.dtype, device=self.torch_device)
            kld = self.calculate_KL_divergence()
            J = kld + torch.pow(torch.maximum(torch.max(self.t),one_tensor),4) + torch.pow(torch.minimum(torch.min(self.t),-one_tensor),4)
            J.backward()
            return J

        # Perform the optimization step
        return optimizer, closure
    
    def update_transformation(self):
        self.__optimizer.step(self.__closure)
