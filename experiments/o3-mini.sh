################ COT experiments
python -m experiments.run_experiment --dataset_name DiscoveryBench --model_name o3-mini --agent_type COT --overwrite
python -m experiments.run_experiment --dataset_name QRData --model_name o3-mini --agent_type COT --overwrite
python -m experiments.run_experiment --dataset_name StatQA --model_name o3-mini --agent_type COT --overwrite

################ ROT experiments
python -m experiments.run_experiment --dataset_name DiscoveryBench --model_name o3-mini --agent_type ROT --overwrite
python -m experiments.run_experiment --dataset_name QRData --model_name o3-mini --agent_type ROT --overwrite
python -m experiments.run_experiment --dataset_name StatQA --model_name o3-mini --agent_type ROT --overwrite

################ ReAct experiments
python -m experiments.run_experiment --dataset_name DiscoveryBench --model_name o3-mini --agent_type REACT --overwrite
python -m experiments.run_experiment --dataset_name QRData --model_name o3-mini --agent_type REACT --overwrite
python -m experiments.run_experiment --dataset_name StatQA --model_name o3-mini --agent_type REACT --overwrite

################ Reflexion
python -m experiments.run_experiment --dataset_name DiscoveryBench --model_name o3-mini --agent_type REFLEXION --overwrite
python -m experiments.run_experiment --dataset_name QRData --model_name o3-mini --agent_type REFLEXION --overwrite
python -m experiments.run_experiment --dataset_name StatQA --model_name o3-mini --agent_type REFLEXION --overwrite
