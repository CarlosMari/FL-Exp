# FL-Exp

Experiments with aggregation methods in federated learning, using plain files to exchange weights.

- `train.py N`: a client node (`N` = 1 or 2) trains a ResNet-18 on its chunk of CIFAR-10, writes its
  parameters to `weights_N.pickle`, and signals with `readyN.b`.
- `central_node.py`: waits for both nodes, averages their parameters (FedAvg with equal weights),
  saves the global model to `out/model.pth`, and reports validation loss each round.
- `config.py`: number of nodes, data chunking, batch size, epochs, seed.

## Run

```bash
pip install torch torchvision tqdm
mkdir -p out
python central_node.py &   # one terminal
python train.py 1 &        # one per node
python train.py 2
```

The scripts default to CUDA.
