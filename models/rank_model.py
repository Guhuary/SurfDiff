import math

from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean
import numpy as np
from e3nn.nn import BatchNorm

from datasets.process_mols import lig_feature_dims, rec_residue_feature_dims, rec_atom_feature_dims
from torch_geometric.utils import to_dense_batch

class AtomEncoder(torch.nn.Module):
	def __init__(self, emb_dim, feature_dims, lm_embedding_type= None):
		# first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
		super(AtomEncoder, self).__init__()
		self.atom_embedding_list = torch.nn.ModuleList()
		self.num_categorical_features = len(feature_dims[0])
		self.num_scalar_features = feature_dims[1]
		self.lm_embedding_type = lm_embedding_type
		for i, dim in enumerate(feature_dims[0]):
			emb = torch.nn.Embedding(dim, emb_dim)
			torch.nn.init.xavier_uniform_(emb.weight.data)
			self.atom_embedding_list.append(emb)

		if self.num_scalar_features > 0:
			self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
		if self.lm_embedding_type is not None:
			if self.lm_embedding_type == 'esm':
				self.lm_embedding_dim = 1280
			else: 
				raise ValueError('LM Embedding type was not correctly determined. LM embedding type: ', self.lm_embedding_type)
			self.lm_embedding_layer = torch.nn.Linear(self.lm_embedding_dim + emb_dim, emb_dim)

	def forward(self, x):
		x_embedding = 0
		if self.lm_embedding_type is not None:
			assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.lm_embedding_dim
		else:
			assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
		for i in range(self.num_categorical_features):
			x_embedding += self.atom_embedding_list[i](x[:, i].long())

		if self.num_scalar_features > 0:
			# print(x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features].device)
			x_embedding += self.linear(x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])
		if self.lm_embedding_type is not None:
			x_embedding = self.lm_embedding_layer(torch.cat([x_embedding, x[:, -self.lm_embedding_dim:]], axis=1))
		return x_embedding

class TensorProductConvLayer(torch.nn.Module):
	def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
				 hidden_features=None):
		super(TensorProductConvLayer, self).__init__()
		self.in_irreps = in_irreps
		self.out_irreps = out_irreps
		self.sh_irreps = sh_irreps
		self.residual = residual
		if hidden_features is None:
			hidden_features = n_edge_features

		self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

		self.fc = nn.Sequential(
			nn.Linear(n_edge_features, hidden_features),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_features, tp.weight_numel)
		)
		self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

	def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean',p=False):
		edge_src, edge_dst = edge_index
		tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

		out_nodes = out_nodes or node_attr.shape[0]
		out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

		if self.residual:
			padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
			out = out + padded

		if self.batch_norm:
			out = self.batch_norm(out)
		return out

class Rank_Model(torch.nn.Module):
	def __init__(self, in_lig_edge_features=4, sh_lmax=2,
				 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
				 distance_embed_dim=32, cross_distance_embed_dim=32, 
				 use_second_order_repr=False, batch_norm=True,
				 dropout=0.0, lm_embedding_type=None):
		super(Rank_Model, self).__init__()
		self.in_lig_edge_features = in_lig_edge_features                # 4
		self.lig_max_radius = lig_max_radius                            # 5
		self.rec_max_radius = rec_max_radius                            # 30
		self.cross_max_distance = cross_max_distance                    # 10
		self.distance_embed_dim = distance_embed_dim                    # 64
		self.cross_distance_embed_dim = cross_distance_embed_dim        # 64
		self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
		self.ns, self.nv = ns, nv                                       # 48, 10
		self.num_conv_layers = num_conv_layers                          # 6

		self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims)
		self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + distance_embed_dim, ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

		self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, lm_embedding_type=lm_embedding_type)
		self.rec_edge_embedding = nn.Sequential(nn.Linear(distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

		self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
		self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)

		if use_second_order_repr:
			irrep_seq = [
				f'{ns}x0e',
				f'{ns}x0e + {nv}x1o + {nv}x2e',
				f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
				f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
			]
		else:
			irrep_seq = [
				f'{ns}x0e',
				f'{ns}x0e + {nv}x1o',
				f'{ns}x0e + {nv}x1o + {nv}x1e',
				f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
			]

		rec_conv_layers = []
		# ar_conv_layers, ra_conv_layers = [], []
		# atom_conv_layers = []
		lig_conv_layers = []
		for i in range(num_conv_layers):
			in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
			out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
			parameters = {
				'in_irreps': in_irreps,
				'sh_irreps': self.sh_irreps,
				'out_irreps': out_irreps,
				'n_edge_features': 3 * ns,
				'hidden_features': 3 * ns,
				'residual': False,
				'batch_norm': batch_norm, 
				'dropout': dropout
			}

			lig_conv_layers.append(TensorProductConvLayer(**parameters))
			rec_conv_layers.append(TensorProductConvLayer(**parameters))


		self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
		self.rec_conv_layers = nn.ModuleList(rec_conv_layers)

		self.MLP = nn.Sequential(
						nn.Linear(2 * ns, ns),
						nn.ReLU(),
						nn.Dropout(dropout),
						nn.Linear(ns, ns)
					)
		n_gaussians = 10
		self.z_pi = nn.Linear(ns, n_gaussians)
		self.z_sigma = nn.Linear(ns, n_gaussians)
		self.z_mu = nn.Linear(ns, n_gaussians)

	def forward(self, data):
		# build receptor graph
		rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
		rec_src, rec_dst = rec_edge_index
		rec_node_attr = self.rec_node_embedding(rec_node_attr)
		rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

		# build ligand graph
		lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
		lig_src, lig_dst = lig_edge_index
		lig_node_attr = self.lig_node_embedding(lig_node_attr)
		lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

		for l in range(self.num_conv_layers):
			# intra ligand graph message passing
			lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
			lig_intra_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)

			# padding original features and update features with residual updates
			lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))
			lig_node_attr = lig_node_attr + lig_intra_update
			
			rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
			rec_intra_update = self.rec_conv_layers[l](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)

			rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
			rec_node_attr = rec_node_attr + rec_intra_update

		h_l_x, l_mask = to_dense_batch(lig_node_attr[:, :self.ns], data['ligand'].batch)
		h_t_x, t_mask = to_dense_batch(rec_node_attr[:, :self.ns], data['receptor'].batch)

		h_l_pos, _ = to_dense_batch(data['ligand'].pos, data['ligand'].batch)
		h_t_pos, _ = to_dense_batch(data['receptor'].pos, data['receptor'].batch)

		assert h_l_x.size(0) == h_t_x.size(0), 'Encountered unequal batch-sizes'
		(B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)

		# Combine and mask
		h_l_x = h_l_x.unsqueeze(-2) # [B, N_l, 1, C_out]
		h_l_x = h_l_x.repeat(1, 1, N_t, 1) # [B, N_l, N_t, C_out]

		h_t_x = h_t_x.unsqueeze(-3)
		h_t_x = h_t_x.repeat(1, N_l, 1, 1) # [B, N_l, N_t, C_out]
		C = torch.cat((h_l_x, h_t_x), -1) # [B, N_l, N_t, 2 x C_out]
		C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
		C = C[C_mask]
		C = self.MLP(C)
		
		# Outputs
		pi = F.softmax(self.z_pi(C), -1)
		sigma = F.elu(self.z_sigma(C)) + 1.1
		mu = F.elu(self.z_mu(C)) + 1
		dist = self.compute_euclidean_distances_matrix(h_l_pos, h_t_pos)[C_mask]

		return pi, sigma, mu, dist.unsqueeze(1)

	def compute_euclidean_distances_matrix(self, X, Y):
		# Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
		# (X-Y)^2 = X^2 + Y^2 -2XY
		dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2,    axis=-1).unsqueeze(1) + \
				torch.sum(X**2, axis=-1).unsqueeze(-1)
		return dists ** 0.5

	def build_lig_conv_graph(self, data):
		# compute edges
		radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch)  # find edges with dis < 5
		edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
		edge_attr = torch.cat([
			data['ligand', 'ligand'].edge_attr,
			torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device)
		], 0)
		# edge_index: 2 * (bond + new[dis < 5])
		# edge_attr: (bond + new[dis < 5]) x 4

		# compute initial features
		node_attr = data['ligand'].x

		src, dst = edge_index
		edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
		edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

		edge_attr = torch.cat([edge_attr, edge_length_emb], 1)      # # edge_attr: (bond + new[dis < 5]) x (4 + 64 + 64)
		edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

		return node_attr, edge_index, edge_attr, edge_sh

	def build_rec_conv_graph(self, data):
		# builds the receptor initial node and edge embeddings
		node_attr = data['receptor'].x

		# this assumes the edges were already created in preprocessing since protein's structure is fixed
		edge_index = data['receptor', 'receptor'].edge_index
		src, dst = edge_index
		edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

		edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
		edge_attr = edge_length_emb
		edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

		return node_attr, edge_index, edge_attr, edge_sh

	def build_atom_conv_graph(self, data):
		# build the graph between receptor atoms
		node_attr = data['atom'].x

		# this assumes the edges were already created in preprocessing since protein's structure is fixed
		edge_index = data['atom', 'atom'].edge_index
		src, dst = edge_index
		edge_vec = data['atom'].pos[dst.long()] - data['atom'].pos[src.long()]

		edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))
		edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

		return node_attr, edge_index, edge_attr, edge_sh

	def build_cross_conv_graph(self, data, lr_cross_distance_cutoff):
		# build the cross edges between ligan atoms, receptor residues and receptor atoms

		# ATOM to RECEPTOR
		ar_edge_index = data['atom', 'receptor'].edge_index
		ar_edge_vec = data['receptor'].pos[ar_edge_index[1].long()] - data['atom'].pos[ar_edge_index[0].long()]
		ar_edge_attr = self.rec_distance_expansion(ar_edge_vec.norm(dim=-1))
		ar_edge_sh = o3.spherical_harmonics(self.sh_irreps, ar_edge_vec, normalize=True, normalization='component')

		return ar_edge_index, ar_edge_attr, ar_edge_sh


class GaussianSmearing(torch.nn.Module):
	# used to embed the edge distances
	def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
		super().__init__()
		offset = torch.linspace(start, stop, num_gaussians)
		self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
		self.register_buffer('offset', offset)

	def forward(self, dist):
		dist = dist.view(-1, 1) - self.offset.view(1, -1)
		return torch.exp(self.coeff * torch.pow(dist, 2))
