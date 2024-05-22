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
from utils.training import train_epoch_rank as train_epoch
from utils.training import test_epoch_rank as test_epoch
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_rank_model, ExponentialMovingAverage

from torch.utils.tensorboard import SummaryWriter

def train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, writer, restart_epoch):
	best_val_loss = math.inf
	best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0 # 0 
	best_epoch = 0
	best_val_inference_epoch = 0
	loss_fn = torch.nn.BCELoss()

	print("Starting training...")
	start_val = 0
	for epoch in range(restart_epoch+1, args.n_epochs):
		if epoch % 5 == 0: 
			print("Run name: ", args.run_name)
		logs = {}
		
		train_losses = train_epoch(model, train_loader, optimizer, device, ema_weights, loss_fn)
		print(f"Epoch {epoch}: loss {train_losses['loss']}")

		
		ema_weights.store(model.parameters())
		if args.use_ema: 
			ema_weights.copy_to(model.parameters()) # load ema parameters into model for running validation and inference
		
		if epoch >= start_val:
			val_losses = test_epoch(model, val_loader, device, loss_fn)
			print(f"Epoch {epoch}: loss {val_losses['loss']}")
			writer.add_scalar('Val/loss', val_losses['loss'], epoch)

		if not args.use_ema: 
			ema_weights.copy_to(model.parameters())
		ema_state_dict = copy.deepcopy(model.module.state_dict() if device.type == 'cuda' else model.state_dict())
		ema_weights.restore(model.parameters())

		state_dict = model.module.state_dict() if device.type == 'cuda' else model.state_dict()

		if epoch >= start_val:
			if val_losses['loss'] <= best_val_loss:
				best_val_loss = val_losses['loss']
				best_epoch = epoch
				torch.save({'epoch': epoch,
					'state_dict': state_dict}, os.path.join(run_dir, 'best_model.pt'))
				torch.save({'epoch': epoch,
					'state_dict': ema_state_dict}, os.path.join(run_dir, 'best_ema_model.pt'))

		#### save models for every 50 epochs
		if (epoch) % 5 == 0:
			torch.save(state_dict, os.path.join(run_dir, 'epoch{}_inference_model.pt'.format(epoch)))
			torch.save(ema_state_dict, os.path.join(run_dir, 'epoch{}_ema_inference_model.pt'.format(epoch)))

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

	model = get_rank_model(args, device)
	optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.inference_earlystop_goal if args.val_inference_freq is not None else 'min')
	ema_weights = ExponentialMovingAverage(model.parameters(),decay=args.ema_rate)

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

	# ema_weights.store(model.parameters())
	# if args.use_ema: 
	# 	ema_weights.copy_to(model.parameters()) # load ema parameters into model for running validation and inference
	# if not args.use_ema: 
	# 	ema_weights.copy_to(model.parameters())
	# ema_state_dict = copy.deepcopy(model.module.state_dict() if device.type == 'cuda' else model.state_dict())
	# ema_weights.restore(model.parameters())
	# state_dict = model.module.state_dict() if device.type == 'cuda' else model.state_dict()
	# epoch = 0
	# torch.save(state_dict, os.path.join(run_dir, 'epoch{}_inference_model.pt'.format(epoch)))
	# torch.save(ema_state_dict, os.path.join(run_dir, 'epoch{}_ema_inference_model.pt'.format(epoch)))
if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # torch.device('cpu')
	main_function()