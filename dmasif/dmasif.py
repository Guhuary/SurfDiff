import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import os
from .geometry_processing import (
	curvatures,
	atoms_to_points_normals,
)
# from helper import soft_dimension, diagonal_ranges
from .benchmark_models import dMaSIFConv_seg

import time


def extract_surf_feature(atoms):
	batch_atoms = torch.zeros(atoms.shape[0]).long().to(atoms.device)
	xyz, normals, batch, batch_for_atom = atoms_to_points_normals(
		atoms, 
		batch_atoms,
		atomtypes=None,
		resolution=1,
		sup_sampling=20,
	)
	# Estimate the curvatures using the triangles or the estimated normals:
	P_curvatures = curvatures(
		xyz,
		triangles=None,
		normals=normals,
		scales=[1.0, 2.0, 3.0, 5.0, 10.0],
		batch=batch,
	)
	return xyz, normals, batch, batch_for_atom, P_curvatures
	
class Surf_Extract(nn.Module):
	"""docstring for Surf_Extract"""
	def __init__(self, in_channels=10, out_channels=32, resolution=1, sup_sampling=20):
		super(Surf_Extract, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.dropout = nn.Dropout(p=0.25)
		self.resolution = resolution
		self.sup_sampling = sup_sampling
		self.curvature_scales = [1.0, 2.0, 3.0, 5.0, 10.0]
		self.Conv = dMaSIFConv_seg(in_channels=in_channels, out_channels=out_channels, n_layers=1)
		self.orientation_scores = nn.Sequential(
			nn.Linear(in_channels, out_channels),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(out_channels, 1),
		)

	def features(self, atoms, batch_for_protein):
		"""Estimates geometric and chemical features from a protein surface or a cloud of atoms."""
		fail = 0
		for _ in range(5):
			xyz, normals, batch, batch_for_atom = atoms_to_points_normals(
				atoms,
				batch_for_protein,
				atomtypes=None,
				resolution=self.resolution,
				sup_sampling=self.sup_sampling,
			)

			# Estimate the curvatures using the triangles or the estimated normals:
			P_curvatures = curvatures(
				xyz,
				triangles=None,
				normals=normals,
				scales=self.curvature_scales,
				batch=batch,
			)
			if not P_curvatures.isnan().any():
				break
			else:
				fail += 1

		if fail == 5:
			print('P_curvatures exist nan')
			np.savez('exp.npz', xyz=xyz.detach().cpu().numpy(), normals=normals.detach().cpu().numpy(), batch=batch.detach().cpu().numpy())
		if P_curvatures.isnan().any():
			P_curvatures[P_curvatures.isnan()] = 0
		# Concatenate our features:
		self.Conv.load_mesh(
				xyz,
				triangles=None,
				normals=normals,
				weights=self.orientation_scores(P_curvatures),
				batch=batch,
			)
		return xyz, P_curvatures, batch, batch_for_atom

	def forward(self, atoms, batch_for_protein):
		xyz, features, batch, batch_for_atom = self.features(atoms, batch_for_protein)
		output = self.Conv(features)
		return xyz, output, batch


	'''
	def forward_with_feature(self, xyz, normals, P_curvatures, batch_for_protein):
		self.Conv.load_mesh(xyz, 
			triangles=None, 
			normals=normals, 
			weights=self.orientation_scores(P_curvatures), 
			batch=batch_for_protein)
		output = self.Conv(P_curvatures)
		return output
	'''