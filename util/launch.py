import itertools

# ----------------------------------------------------------------------------

def print_grid(args):
  base_cmd = """python run.py train \
    --dataset {dataset} \
    --model {model} \
    -e {epochs} \
    -l {logname}.{dataset}.{model}.{alg}.{lr}.{b1}.{b2}.{nb}.{nsb} \
    --alg {alg} \
    --lr {lr} \
    --b1 {b1} \
    --b2 {b2} \
    --n_batch {nb} \
    --n_subbatch {nsb}"""

  for alg, lr, n_batch, n_subbatch, b1, b2 \
    in itertools.product(args.alg, args.lr, args.n_batch, args.n_subbatch,
                         args.b1, args.b2):
    if n_subbatch >= n_batch : continue
    cmd =  base_cmd.format(dataset=args.dataset, model=args.model, epochs=args.epochs,
                           logname=args.logname, alg=alg, lr=lr, nb=n_batch, 
                           nsb=n_subbatch, b1=b1, b2=b2)
    print cmd