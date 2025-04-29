#!/bin/bash
#SBATCH -p ei-short
#SBATCH --mem 1G
#SBATCH --time=0-00:10:00
#SBATCH -c 1	
#SBATCH --output %x.out 		# STDOUT and STDERR
#SBATCH --mail-type=END,FAIL			# notifications for job done & fail
#SBATCH --mail-user=lucy.mahony@earlham.ac.uk	# send-to address

source ~/.bashrc 
mamba activate /hpc-home/mahony/miniforge3
conda run -n transformers python multibranch_nn.py

