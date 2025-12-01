import numpy as np
import torch

class ForwardImaging:
    """Forward model that simulates Poisson measurements with Gaussian blur and downsampling."""

    def __init__(self, resolution, device = 'cpu', vertical_super_res = False, photons = 20) -> None:
        self.resolution = resolution
        self.photons = photons
        
        self.padding = [0, 4, 5]
        
        self.stride = [2, 3, 6] if vertical_super_res else [1, 3, 6]
            
        self.kernel_size = 11

        self.sigma = 6/2.35

        self.kernel = torch.from_numpy(self.gaussian_kernel(self.kernel_size, self.sigma)).unsqueeze(0).float()
        self.kernel = torch.ones(2 if vertical_super_res else 1, self.kernel_size, self.kernel_size) * self.kernel
        self.kernel = self.kernel.to(device)

    def gaussian_kernel(self, size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1/ (2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
            (size, size)
        )
        return kernel / np.sum(kernel)
    
    def forward_imaging(self, high_res, mask = None):
        """Apply blur + downsampling and return Poisson distribution and the aggregated volume."""

        high_res = high_res.unsqueeze(1)
        volume_shape = high_res.shape
        high_res = torch.nn.functional.interpolate(
            high_res,
            size=[volume_shape[-3], volume_shape[-2] * self.resolution, volume_shape[-1] * self.resolution],
            mode='nearest'
        )
        
        aggregated = torch.nn.functional.conv3d(high_res, self.kernel.unsqueeze(0).unsqueeze(0), 
                                         stride=self.stride, 
                                         padding=self.padding).squeeze()
        
        aggregated = torch.relu(aggregated)
        aggregated = aggregated / (aggregated.sum(0) + 1e-6)

        if mask is not None:
            # print(aggregated.shape, mask.shape)
            distribution = torch.distributions.Poisson(self.photons*aggregated[mask == 1])
        else:
            distribution = torch.distributions.Poisson(self.photons*aggregated)

        return distribution, aggregated
    
    def forward_imaging_poisson_fix(self, high_res, mask = None):

        high_res = high_res.unsqueeze(1)
        volume_shape = high_res.shape
        high_res = torch.nn.functional.interpolate(
            high_res,
            size=[volume_shape[-3], volume_shape[-2] * self.resolution, volume_shape[-1] * self.resolution],
            mode='nearest'
        )
        
        aggregated = torch.nn.functional.conv3d(high_res, self.kernel.unsqueeze(0).unsqueeze(0), 
                                         stride=self.stride, 
                                         padding=self.padding).squeeze()
        
        aggregated = torch.relu(aggregated)
        aggregated = aggregated / (aggregated.sum(0) + 1e-6)

        if mask is not None:
            return self.photons*aggregated[mask == 1]
        else:
            return self.photons*aggregated
    
    def forward_imaging_gaussian(self, high_res, mask = None):

        high_res = high_res.unsqueeze(1)
        volume_shape = high_res.shape
        high_res = torch.nn.functional.interpolate(
            high_res,
            size=[volume_shape[-3], volume_shape[-2] * self.resolution, volume_shape[-1] * self.resolution],
            mode='nearest'
        )
        
        aggregated = torch.nn.functional.conv3d(high_res, self.kernel.unsqueeze(0).unsqueeze(0), 
                                         stride=self.stride, 
                                         padding=self.padding).squeeze()
        
        aggregated = torch.relu(aggregated)
        aggregated = (aggregated / (aggregated.sum(0) + 1e-6))

        if mask is not None:
            distribution = torch.distributions.Normal((self.photons*aggregated[mask == 1]), (self.photons*aggregated[mask == 1]).sqrt()+1e-6)
        else:
            distribution = torch.distributions.Normal((self.photons*aggregated),(self.photons*aggregated).sqrt()+1e-6)

        return distribution, aggregated
    
    def forward_imaging_multinomial(self, high_res, pattern):

        high_res = high_res.unsqueeze(1)
        volume_shape = high_res.shape
        high_res = torch.nn.functional.interpolate(
            high_res,
            size=[volume_shape[-3], volume_shape[-2] * self.resolution, volume_shape[-1] * self.resolution],
            mode='nearest'
        )
        
        aggregated = torch.nn.functional.conv3d(high_res, self.kernel.unsqueeze(0).unsqueeze(0), 
                                         stride=self.stride, 
                                         padding=self.padding).squeeze()
        
        aggregated = pattern @ aggregated.reshape(aggregated.shape[0],-1).T

        distribution = torch.distributions.Multinomial(self.photons, aggregated + 1e-6)
        return distribution, aggregated
    
    def forward_imaging_anscombe(self, high_res, mask=None):
        # same pre-processing as your function
        high_res = high_res.unsqueeze(1)
        volume_shape = high_res.shape
        high_res = torch.nn.functional.interpolate(
            high_res,
            size=[volume_shape[-3], volume_shape[-2] * self.resolution, volume_shape[-1] * self.resolution],
            mode='nearest'
        )

        aggregated = torch.nn.functional.conv3d(
            high_res, self.kernel.unsqueeze(0).unsqueeze(0),
            stride=self.stride, padding=self.padding
        ).squeeze()

        aggregated = torch.relu(aggregated)
        aggregated = aggregated / (aggregated.sum(0) + 1e-6)

        # Poisson rate λ
        lam = self.photons * (aggregated if mask is None else aggregated[mask == 1])

        # Anscombe mean μ_A and unit variance
        mu_A = 2.0 * torch.sqrt(lam + 3.0/8.0)

        # Gaussian in Anscombe domain
        dist_A = torch.distributions.Normal(mu_A, torch.ones_like(mu_A))

        return dist_A, aggregated
