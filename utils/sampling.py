import numpy as np
import torch
from torch_geometric.loader import DataLoader

from utils.diffusion_utils import modify_conformer, set_time
from utils.torsion import modify_conformer_torsion_angles
from scipy.spatial.transform import Rotation as R

from datasets.process_mols import write_mol_with_coords
from torch.distributions import Normal

def log_prob(pi, sigma, mu, y):
	normal = Normal(mu, sigma)
	loglik = normal.log_prob(y.expand_as(normal.loc))
	loss = torch.logsumexp(torch.log(pi+1e-10) + loglik, dim=1)
	return loss

def randomize_position(data_list, no_torsion, no_random, tr_sigma_max):
	# in place modification of the list
	if not no_torsion:
		# randomize torsion angles
		for complex_graph in data_list:
			torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
			complex_graph['ligand'].pos = \
				modify_conformer_torsion_angles(complex_graph['ligand'].pos,
												complex_graph['ligand', 'ligand'].edge_index.T[
													complex_graph['ligand'].edge_mask],
												complex_graph['ligand'].mask_rotate[0], torsion_updates)

	for complex_graph in data_list:
		# randomize position
		molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
		random_rotation = torch.from_numpy(R.random().as_matrix()).float()
		# complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T
		
		# Add prior position to the ligand
		complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T
		
		# base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())
		complex_graph['ligand'].pos += complex_graph.ligand_center
		if not no_random:  # note for now the torsion angles are still randomised  # if False:
			tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))     # 
			complex_graph['ligand'].pos += tr_update
		

def sampling(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, device, t_to_sigma, model_args,
			 no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
			 confidence_model_args=None, batch_size=32, no_final_step_noise=False, rank_model=None):
	N = len(data_list)
	visualization_ = [ [] for _ in range(40)]
	for t_idx in range(inference_steps):
		t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
		dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
		dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
		dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

		loader = DataLoader(data_list, batch_size=batch_size)
		new_data_list = []

		for complex_graph_batch in loader:
			b = complex_graph_batch.num_graphs
			complex_graph_batch = complex_graph_batch.to(device)

			tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
			set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, model_args.all_atoms, device)
			
			with torch.no_grad():
				tr_score, rot_score, tor_score = model(complex_graph_batch)

			tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
			rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))

			if ode:             # F
				tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
				rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
			else:
				tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
					else torch.normal(mean=0, std=1, size=(b, 3))
				tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

				rot_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
					else torch.normal(mean=0, std=1, size=(b, 3))
				rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()

			if not model_args.no_torsion:
				tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
				if ode:
					tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score.cpu()).numpy()
				else:
					tor_z = torch.zeros(tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
						else torch.normal(mean=0, std=1, size=tor_score.shape)
					tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
				torsions_per_molecule = tor_perturb.shape[0] // b
			else:
				tor_perturb = None

			# Apply noise
			new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1], rot_perturb[i:i + 1].squeeze(0),
										  tor_perturb[i * torsions_per_molecule:(i + 1) * torsions_per_molecule] if not model_args.no_torsion else None)
						 for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
		data_list = new_data_list
		
		for i in range(len(data_list)):
			visualization_[i].append((data_list[i]['ligand'].pos + data_list[i].original_center).detach().cpu().numpy())
		
		if visualization_list is not None:
			for idx, visualization in enumerate(visualization_list):
				visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
								  part=1, order=t_idx + 2)

	with torch.no_grad():
		if rank_model is not None:
			loader = DataLoader(data_list, batch_size=batch_size)
			confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
			confidence = []
			for complex_graph_batch in loader:
				complex_graph_batch = complex_graph_batch.to(device)
				if confidence_data_list is not None:
					confidence_complex_graph_batch = next(confidence_loader).to(device)
					confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
					pi, sigma, mu, dist = rank_model(confidence_complex_graph_batch)
					
					n_lig = confidence_complex_graph_batch['ligand'].pos.shape[0] // batch_size
					n_rec = confidence_complex_graph_batch['receptor'].pos.shape[0] // batch_size
					dist_sdf = dist.reshape(batch_size, n_lig, n_rec)

					likelihood = log_prob(pi, sigma, mu, dist)
					print(likelihood)
					sdf = - 0.5 *((- dist_sdf / 0.5).logsumexp(dim=-1, keepdim=True))  # bs x n_lig x 1
					mask = 1 		# sdf > 5
					confidence.append((mask * likelihood.reshape(batch_size, n_lig, n_rec)).mean(dim=(1, 2)))
				else:
					pi, sigma, mu, dist = rank_model(complex_graph_batch)
					
					n_lig = complex_graph_batch['ligand'].pos.shape[0] // batch_size
					n_rec = complex_graph_batch['receptor'].pos.shape[0] // batch_size
					dist_sdf = dist.reshape(batch_size, n_lig, n_rec)

					likelihood = log_prob(pi, sigma, mu, dist)
					sdf = - 0.5 *((- dist_sdf / 0.5).logsumexp(dim=-1, keepdim=True))  # bs x n_lig x 1
					mask = sdf > 1
					confidence.append((mask * likelihood.reshape(batch_size, n_lig, n_rec)).mean(dim=(1, 2)))
			confidence_me = torch.cat(confidence, dim=0).unsqueeze(1)

		if confidence_model is not None:
			loader = DataLoader(data_list, batch_size=batch_size)
			confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
			confidence = []
			for complex_graph_batch in loader:
				complex_graph_batch = complex_graph_batch.to(device)
				if confidence_data_list is not None:
					confidence_complex_graph_batch = next(confidence_loader).to(device)
					confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
					set_time(confidence_complex_graph_batch, 0, 0, 0, N, confidence_model_args.all_atoms, device)
					confidence.append(confidence_model(confidence_complex_graph_batch))
				else:
					confidence.append(confidence_model(complex_graph_batch))
			confidence = torch.cat(confidence, dim=0)
		else:
			confidence = None

	return data_list, confidence, visualization_


def sampling_new_confidence(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, device, t_to_sigma, model_args,
			 no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
			 confidence_model_args=None, batch_size=32, no_final_step_noise=False, rank_model=None):
	N = len(data_list)
	visualization_ = [ [] for _ in range(40)]
	for t_idx in range(inference_steps):
		t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
		dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
		dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
		dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

		loader = DataLoader(data_list, batch_size=batch_size)
		new_data_list = []

		for complex_graph_batch in loader:
			b = complex_graph_batch.num_graphs
			complex_graph_batch = complex_graph_batch.to(device)

			tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
			set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, model_args.all_atoms, device)

			with torch.no_grad():
				tr_score, rot_score, tor_score = model(complex_graph_batch)

			tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
			rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))

			if ode:             # F
				tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
				rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
			else:
				tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
					else torch.normal(mean=0, std=1, size=(b, 3))
				tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

				rot_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
					else torch.normal(mean=0, std=1, size=(b, 3))
				rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()

			if not model_args.no_torsion:
				tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
				if ode:
					tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score.cpu()).numpy()
				else:
					tor_z = torch.zeros(tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
						else torch.normal(mean=0, std=1, size=tor_score.shape)
					tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
				torsions_per_molecule = tor_perturb.shape[0] // b
			else:
				tor_perturb = None

			del complex_graph_batch['protein_surface']
			del complex_graph_batch['ligand_surface']
			# Apply noise
			new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1], rot_perturb[i:i + 1].squeeze(0),
										  tor_perturb[i * torsions_per_molecule:(i + 1) * torsions_per_molecule] if not model_args.no_torsion else None)
						 for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
		data_list = new_data_list
		
		for i in range(len(data_list)):
			visualization_[i].append((data_list[i]['ligand'].pos + data_list[i].original_center).detach().cpu().numpy())
		
		if visualization_list is not None:
			for idx, visualization in enumerate(visualization_list):
				visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
								  part=1, order=t_idx + 2)

	with torch.no_grad():
		if rank_model is not None and confidence_model is not None:
			loader = DataLoader(data_list, batch_size=batch_size)
			confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
			confidence = []
			for complex_graph_batch in loader:
				complex_graph_batch = complex_graph_batch.to(device)
				if confidence_data_list is not None:
					confidence_complex_graph_batch = next(confidence_loader).to(device)
					confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
					pi, sigma, mu, dist = rank_model(confidence_complex_graph_batch)
					n_lig = confidence_complex_graph_batch['ligand'].pos.shape[0] // batch_size
					n_rec = confidence_complex_graph_batch['receptor'].pos.shape[0] // batch_size
					dist_sdf = dist.reshape(batch_size, n_lig, n_rec)
					likelihood = log_prob(pi, sigma, mu, dist)
					sdf = - 0.35 *((- dist_sdf / 0.35).logsumexp(dim=-1, keepdim=True))  # bs x n_lig x 1
					mask = sdf > 0.5
					set_time(confidence_complex_graph_batch, 0, 0, 0, N, confidence_model_args.all_atoms, device)
					confidence.append(
						(mask * likelihood.reshape(batch_size, n_lig, n_rec)).mean(dim=(1, 2)) * 0.5 + 
						confidence_model(confidence_complex_graph_batch)[:, 0] * 0.5
						)
				else:
					pi, sigma, mu, dist = rank_model(complex_graph_batch)
					
					n_lig = complex_graph_batch['ligand'].pos.shape[0] // batch_size
					n_rec = complex_graph_batch['receptor'].pos.shape[0] // batch_size
					dist_sdf = dist.reshape(batch_size, n_lig, n_rec)

					likelihood = log_prob(pi, sigma, mu, dist)
					sdf = - 0.35 *((- dist_sdf / 0.35).logsumexp(dim=-1, keepdim=True))  # bs x n_lig x 1
					mask = sdf > 0.5
					confidence.append((mask * likelihood.reshape(batch_size, n_lig, n_rec)).mean(dim=(1, 2)) * 0.5 + 
						confidence_model(complex_graph_batch))
			confidence = torch.cat(confidence, dim=0).unsqueeze(1)

	return data_list, confidence, visualization_
