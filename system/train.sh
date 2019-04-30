srun --job-name=senti -p AD -n1 --gres=gpu:$2 --ntasks-per-node=1 --cpus-per-task=5 python -u main.py --config configs/$1.yaml
