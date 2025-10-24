# filepath: /home/manuelcastiblan/academic/mlflow-deploy/mlflow-deploy/Makefile
preproces:
	python preprocesSAT.py
train:
	python trainSAT.py 
validate:
	python validateSAT.py
pathModelo:
	python down_modelSAT.py