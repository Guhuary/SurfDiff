import copy
import math
import os
from functools import partial
import random

import wandb
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

import yaml

from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from datasets.pdbbind import construct_loader
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch, loss_function, inference_epoch
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model, ExponentialMovingAverage

from torch.utils.tensorboard import SummaryWriter
'''
def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

Seed_everything(seed=42)
'''
print('no seed')
def train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, writer, restart_epoch):
	best_val_loss = math.inf
	best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0 # 0 
	best_epoch = 0
	best_val_inference_epoch = 0
	loss_fn = partial(loss_function, tr_weight=args.tr_weight, rot_weight=args.rot_weight,
					  tor_weight=args.tor_weight, no_torsion=args.no_torsion)

	print("Starting training...")
	start_val = 400
	for epoch in range(restart_epoch+1, args.n_epochs):
		if epoch % 5 == 0: 
			print("Run name: ", args.run_name)
		logs = {}
		train_losses = train_epoch(model, train_loader, optimizer, device, t_to_sigma, loss_fn, ema_weights)
		print("Epoch {}: Training loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}"
			  .format(epoch, train_losses['loss'], train_losses['tr_loss'], train_losses['rot_loss'],
					  train_losses['tor_loss']))
		writer.add_scalar('Train/loss', train_losses['loss'], epoch)
		writer.add_scalar('Train/tr', train_losses['tr_loss'], epoch)
		writer.add_scalar('Train/rot', train_losses['rot_loss'], epoch)
		writer.add_scalar('Train/tor', train_losses['tor_loss'], epoch)

		ema_weights.store(model.parameters())
		if args.use_ema: 
			ema_weights.copy_to(model.parameters()) # load ema parameters into model for running validation and inference
		if epoch >= start_val:
			val_losses = test_epoch(model, val_loader, device, t_to_sigma, loss_fn, args.test_sigma_intervals)
			print("Epoch {}: Validation loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}"
				  .format(epoch, val_losses['loss'], val_losses['tr_loss'], val_losses['rot_loss'], val_losses['tor_loss']))
			writer.add_scalar('Val/loss', val_losses['loss'], epoch)
			writer.add_scalar('Val/tr', val_losses['tr_loss'], epoch)
			writer.add_scalar('Val/rot', val_losses['rot_loss'], epoch)
			writer.add_scalar('Val/tor', val_losses['tor_loss'], epoch)
		
		if args.val_inference_freq != None and (epoch + 1) % args.val_inference_freq == 0 and epoch >= start_val:
			inf_metrics = inference_epoch(model, val_loader.dataset.complex_graphs[:args.num_inference_complexes], device, t_to_sigma, args)
			print("Epoch {}: Val inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f}"
				  .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5']))
			logs.update({'valinf_' + k: v for k, v in inf_metrics.items()}, step=epoch + 1)

		if not args.use_ema: 
			ema_weights.copy_to(model.parameters())
		ema_state_dict = copy.deepcopy(model.module.state_dict() if device.type == 'cuda' else model.state_dict())
		ema_weights.restore(model.parameters())

		if args.wandb:
			logs.update({'train_' + k: v for k, v in train_losses.items()})
			logs.update({'val_' + k: v for k, v in val_losses.items()})
			logs['current_lr'] = optimizer.param_groups[0]['lr']
			wandb.log(logs, step=epoch + 1)

		state_dict = model.module.state_dict() if device.type == 'cuda' else model.state_dict()
		if args.inference_earlystop_metric in logs.keys() and \
				(args.inference_earlystop_goal == 'min' and logs[args.inference_earlystop_metric] <= best_val_inference_value or
				 args.inference_earlystop_goal == 'max' and logs[args.inference_earlystop_metric] >= best_val_inference_value) and epoch >= start_val:
			best_val_inference_value = logs[args.inference_earlystop_metric]
			best_val_inference_epoch = epoch
			torch.save({'epoch': epoch, 
				'state_dict': state_dict}, os.path.join(run_dir, 'best_inference_epoch_model.pt'))
			torch.save({'epoch': epoch, 
				'state_dict': ema_state_dict}, os.path.join(run_dir, 'best_ema_inference_epoch_model.pt'))
		if epoch >= start_val:
			if val_losses['loss'] <= best_val_loss:
				best_val_loss = val_losses['loss']
				best_epoch = epoch
				torch.save({'epoch': epoch,
					'state_dict': state_dict}, os.path.join(run_dir, 'best_model.pt'))
				torch.save({'epoch': epoch,
					'state_dict': ema_state_dict}, os.path.join(run_dir, 'best_ema_model.pt'))

		#### save models for every 50 epochs
		if (epoch + 1) % 10 == 0:
			torch.save(state_dict, os.path.join(run_dir, 'epoch{}_inference_model.pt'.format(epoch + 1)))
			torch.save(ema_state_dict, os.path.join(run_dir, 'epoch{}_ema_inference_model.pt'.format(epoch + 1)))

		if scheduler:
			if args.val_inference_freq is not None:
				scheduler.step(best_val_inference_value)
			else:
				scheduler.step(val_losses['loss'])

		torch.save({
			'epoch': epoch,
			'model': state_dict,
			'optimizer': optimizer.state_dict(),
			'ema_weights': ema_weights.state_dict(),
		}, os.path.join(run_dir, 'last_model.pt'))

	print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
	print("Best inference metric {} on Epoch {}".format(best_val_inference_value, best_val_inference_epoch))


'''
Namespace(config=None, log_dir='workdir', restart_dir=None, cache_path='data/cache', 
data_dir='data/PDBBind_processed/', split_train='data/splits/timesplit_no_lig_overlap_train', 
split_val='data/splits/timesplit_no_lig_overlap_val', split_test='data/splits/timesplit_test', 
test_sigma_intervals=True, val_inference_freq=5, train_inference_freq=None, inference_steps=20, 
num_inference_complexes=500, inference_earlystop_metric='valinf_rmsds_lt2', inference_earlystop_goal='max', 
wandb=False, project='difdock_train', run_name='try', cudnn_benchmark=True, num_dataloader_workers=1, 
pin_memory=False, n_epochs=850, batch_size=16, scheduler='plateau', scheduler_patience=30, lr=0.001, 
restart_lr=None, w_decay=0.0, num_workers=1, use_ema=True, ema_rate=0.999, limit_complexes=0, all_atoms=False, 
receptor_radius=15.0, c_alpha_max_neighbors=24, atom_radius=5, atom_max_neighbors=8, matching_popsize=20, 
matching_maxiter=20, max_lig_size=None, remove_hs=True, num_conformers=1, esm_embeddings_path='data/esm2_3billion_embeddings.pt', 
tr_weight=0.33, rot_weight=0.33, tor_weight=0.33, rot_sigma_min=0.03, rot_sigma_max=1.55, tr_sigma_min=0.1, tr_sigma_max=19.0, 
tor_sigma_min=0.0314, tor_sigma_max=3.14, no_torsion=False, num_conv_layers=6, max_radius=5.0, scale_by_sigma=True, ns=48, nv=10, 
distance_embed_dim=64, cross_distance_embed_dim=64, no_batch_norm=False, use_second_order_repr=False, cross_max_distance=80, 
dynamic_max_cross=True, dropout=0.1, embedding_type='sinusoidal', sigma_embed_dim=64, embedding_scale=1000)
'''
def main_function():
	args = parse_train_args()
	if args.config:
		config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
		arg_dict = args.__dict__
		for key, value in config_dict.items():
			if isinstance(value, list):
				for v in value:
					arg_dict[key].append(v)
			else:
				arg_dict[key] = value
		args.config = args.config.name
	assert (args.inference_earlystop_goal == 'max' or args.inference_earlystop_goal == 'min')
	if args.val_inference_freq is not None and args.scheduler is not None:
		assert (args.scheduler_patience > args.val_inference_freq) # otherwise we will just stop training after args.scheduler_patience epochs
	if args.cudnn_benchmark:
		torch.backends.cudnn.benchmark = True

	# construct loader
	t_to_sigma = partial(t_to_sigma_compl, args=args)
	train_loader, val_loader = construct_loader(args, t_to_sigma)

	model = get_model(args, device, t_to_sigma=t_to_sigma)
	optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.inference_earlystop_goal if args.val_inference_freq is not None else 'min')
	ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)

	import datetime
	now = datetime.datetime.now().strftime("%Y_%m_%d %H:%M")
	writer = SummaryWriter(log_dir=os.path.join('./log', now))

	if args.restart_dir:
		try:
			dict = torch.load(f'{args.restart_dir}/last_model.pt', map_location=torch.device('cpu'))
			if args.restart_lr is not None: 
				dict['optimizer']['param_groups'][0]['lr'] = args.restart_lr
			optimizer.load_state_dict(dict['optimizer'])
			model.module.load_state_dict(dict['model'], strict=True)
			if hasattr(args, 'ema_rate'):
				ema_weights.load_state_dict(dict['ema_weights'], device=device)
			restart_epoch = dict['epoch']
			print("Restarting from epoch", dict['epoch'])
		except Exception as e:
			print("Exception", e)
			dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
			model.module.load_state_dict(dict, strict=True)
			print("Due to exception had to take the best epoch and no optimiser")
	else:
		restart_epoch = 0
	numel = sum([p.numel() for p in model.parameters()])
	print('Model with', numel, 'parameters')

	if args.wandb:
		wandb.init(
			entity='entity',
			settings=wandb.Settings(start_method="fork"),
			project=args.project,
			name=args.run_name,
			config=args
		)
		wandb.log({'numel': numel})

	# record parameters
	run_dir = os.path.join(args.log_dir, args.run_name)
	yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
	save_yaml_file(yaml_file_name, args.__dict__)
	args.device = device

	train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, writer=writer, restart_epoch=restart_epoch)


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # torch.device('cpu')
	main_function()