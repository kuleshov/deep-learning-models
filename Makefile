PYTHON=/usr/bin/python27
PYTHON=python

EPOCHS=200
NAME=experiment

DATASET=mnist
MODEL=ssadgm
ALG=adam

LR=3e-4
B1=0.9
B2=0.999
SUPERBATCH=1024
NB=128

# ----------------------------------------------------------------------------

train:
	$(PYTHON) run.py train \
	  --dataset $(DATASET) \
	  --model $(MODEL) \
	  -e $(EPOCHS) \
	  -l $(DATASET).$(MODEL).$(ALG).$(LR).$(NB).$(NAME) \
	  --alg $(ALG) \
	  --lr $(LR) \
	  --b1 $(B1) \
	  --b2 $(B2) \
	  --n_superbatch $(SUPERBATCH) \
	  --n_batch $(NB)

grid:
	$(PYTHON) run.py grid \
	  --dataset $(DATASET) \
	  --model $(MODEL) \
	  -e 50 \
	  -l grid \
	  --alg adam \
	  --lr 1e-4 1e-3 5e-3 1e-2 \
	  --b1 0.9 \
	  --n_batch 128
