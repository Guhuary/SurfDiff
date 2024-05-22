# SurfDiff: Enhancing Protein-Ligand Predictions through Initial Pocket Identification and Surface-Aware Ligand Optimization

This project is build upon [DiffDock](https://github.com/gcorso/DiffDock). Please follow DiffDock to set up Dataset and Environment (make sure to use the correct pytorch, pytorch-geometric, pykeops and cuda versions). 

## Run SurfDiff
### Generate the ESM2 embeddings for the proteins
First run:

    python datasets/pdbbind_lm_embedding_preparation.py

Use the generated file `data/pdbbind_sequences.fasta` to generate the ESM2 language model embeddings using the library https://github.com/facebookresearch/esm by installing their repository and executing the following in their repository:

    python scripts/extract.py esm2_t33_650M_UR50D pdbbind_sequences.fasta embeddings_output --repr_layers 33 --include per_tok --truncation_seq_length 4096

This generates the `embeddings_output` directory which you have to copy into the `data` folder of our repository to have `data/embeddings_output`.
Then run the command:

    python datasets/esm_embeddings_to_pt.py

### Using the provided model weights for evaluation
We first generate the language model embeddings for the testset, then run inference with SurfDiff, and then evaluate the files that SurfDiff produced:
	
	python datasets/esm_embedding_preparation.py --protein_ligand_csv data/testset_csv.csv --out_file data/prepared_for_esm_testset.fasta
	git clone https://github.com/facebookresearch/esm 
	cd esm
	pip install -e .
	cd ..
	python esm/scripts/extract.py esm2_t33_650M_UR50D data/prepared_for_esm_testset.fasta data/esm2_output --repr_layers 33 --include per_tok
	python -m inference_rank --protein_ligand_csv data/testset_csv.csv --out_dir results/surfdiff --inference_steps 20 --samples_per_complex 40 --batch_size 5 --actual_steps 18 --no_final_step_noise
	python evaluate_files.py --results_path results/surfdiff --file_to_exclude rank1.sdf --num_predictions 40

One can also run model for a single protein (6qqw for example) as:

	python -m inference_rank --complex_name 6qqw --protein_path data/6qqw/6qqw_protein_processed.pdb --ligand_description data/6qqw/6qqw_ligand.mol2 --out_dir results/6qqw --inference_steps 20 --samples_per_complex 40 --batch_size 5 --actual_steps 18 --no_final_step_noise
