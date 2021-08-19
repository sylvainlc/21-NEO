#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:19:26 2020

@author: achillethin
"""
import math
import pdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.stats import gamma, invgamma


class Target(nn.Module):
    """
    Base class for a custom target distribution
    """

    def __init__(self, kwargs):
        super(Target, self).__init__()
        self.device = kwargs.device
        self.torchType = kwargs.torchType
        self.device_zero = torch.tensor(0., dtype=kwargs.torchType, device=self.device)
        self.device_one = torch.tensor(1., dtype=kwargs.torchType, device=self.device)

    def get_density(self, x, z):
        """
        The method returns target density, estimated at point x
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        density - p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def get_logdensity(self, x, z, prior=None, args=None, prior_flow=None):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        log_density - log p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def get_samples(self, n):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

class Gaussian(Target):
    """
    1 gaussian (multivariate)
    """

    def __init__(self, kwargs):
        super(Gaussian_mixture, self).__init__(kwargs)
        self.device = kwargs.device
        self.mu = kwargs['mu']  # list of locations for each of these gaussians
        self.cov = kwargs['cov']  # list of covariance matrices for each of these gaussians
        self.dist = torch.distributions.MultivariateNormal(loc=self.mu, covariance_matrix=self.cov)

    def get_density(self, z, x=None):
        """
        The method returns target density
        Input:
        x - datapoint
        Output:
        density - p(x)
        """
        density = self.get_logdensity(z).exp()
        return density

    def get_logdensity(self, z, x=None, prior=None, args=None, prior_flow=None):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        Output:
        log_density - log p(x)
        """
        return self.dist.log_prob(x)



class Gaussian_mixture(Target):
    """
    Mixture of n gaussians (multivariate)
    """

    def __init__(self, kwargs):
        super(Gaussian_mixture, self).__init__(kwargs)
        self.device = kwargs.device
        self.num = kwargs['num_gauss']
        self.pis = kwargs['p_gaussians']
        self.locs = kwargs['locs']  # list of locations for each of these gaussians
        self.covs = kwargs['covs']  # list of covariance matrices for each of these gaussians
        self.peak = [None] * self.num
        for i in range(self.num):
            self.peak[i] = torch.distributions.MultivariateNormal(loc=self.locs[i], covariance_matrix=self.covs[i])

    def get_density(self, z, x=None):
        """
        The method returns target density
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        density - p(x)
        """
        density = self.get_logdensity(z).exp()
        return density

    def get_logdensity(self, z, x=None, prior=None, args=None, prior_flow=None):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        log_density - log p(x)
        """
        log_p = torch.tensor([], device=self.device)
        for i in range(self.num):
            log_paux = (torch.log(self.pis[i]) + self.peak[i].log_prob(z)).view(-1, 1)
            log_p = torch.cat([log_p, log_paux], dim=-1)
        log_density = torch.logsumexp(log_p, dim=1)  # + torch.tensor(1337., device=self.device)
        return log_density


