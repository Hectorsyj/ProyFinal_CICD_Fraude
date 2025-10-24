# filepath: /home/manuelcastiblan/academic/mlflow-deploy/mlflow-deploy/Makefile
preproces:
	python src/preprocesSAT.py
train:
	python src/trainSAT.py 
validate:
	python src/validateSAT.py
pathModelo:
	python down_modelSAT.py