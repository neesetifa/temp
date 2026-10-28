# Consolidate six task models into one before on-device deployment

A product ships six small classifiers, one per task, all fine-tuned by different teams from the
same pretrained **base** network. The next device release has room for only **one** network of
the base's size. Your job is to build that single **merged model**: one shared trunk with six
task heads. Once it ships, it replaces all six task models on the device, so each head has to
reproduce its original model's predictions as faithfully as possible.

## What you have

All files are in `/app/data`:

| file | content |
|---|---|
| `base.npz` | the pretrained base trunk (`W1 b1 W2 b2 W3 b3`) |
| `expert_{k}.npz` | task model k (k = 0..5): trunk + `head_W (64,10)`, `head_b (10,)` |
| `unlabeled_inputs.npz` | `task_{k}`: **64 inputs logged from task k's production traffic** |

There are no labels and no other data. The input distribution of each task's traffic over R^48 is
unknown, and some tasks may share the same input distribution. The 64 logged inputs per task
are all you have from it, but you can run the six task models (and the base) on any input you
like.

The six task models started from the base, but not all from the same export of it. Some teams
fine-tuned from a re-exported copy whose neurons are stored in a different order. That copy
computes exactly the same function as the base, but its weights are not directly aligned with
the base or with each other.

## Architecture

The base, the task models and the merged model share the trunk shape (see `/app/mlp.py`):

```
x (48) -> ReLU(x W1 + b1) (64) -> ReLU(. W2 + b2) (64) -> ReLU(. W3 + b3) (64) = h
task model k : logits = h @ head_W + head_b                       (10 classes)
merged model : logits_k = h @ HW[k] + Hb[k]   for task k = 0..5    (one trunk, six heads)
```

## What matters

The merged model will serve the same production traffic the 64 logged inputs came from, i.e. many
more inputs from each task's real input distribution. For task k, what counts is how closely head
k matches task model k on that traffic, measured as the mean **KL(task_model_k(x) ‖ merged
head k(x))** over the softmax outputs, averaged over task k's real inputs. Lower is better.

All six tasks ship together, so the release is only as good as its weakest head. A head that is
far off its task model on its traffic (for example an untrained head, or one task model's weights
reused for every task) breaks that product feature, however good the other five heads are. A
release whose six heads are all close to their task models beats one with five excellent heads
and one poor one. Getting the divergence down on every task, relative to how hard that task is,
is the goal.

Tools in `/app`:

- `mlp.py`: loaders, the forward pass, the KL divergence above, a gradient for the merged model and
  Adam, all in numpy. You may install and use anything else, e.g. torch.
- `make_merged.py`: a working but weak starting point (naive weight averaging).
- `evaluate.py`: reports each task's mean KL on the 64 logged inputs. If you train on those inputs,
  this number is optimistic, so hold some out for an honest estimate of how the model will do on
  real traffic.
- `tests/test_public.py`: checks that your file loads in the deployment format.

## Deliverable

The device build pipeline loads **`/app/merged.npz`**, which must contain these float arrays:
`W1 (48,64) b1 (64,) W2 (64,64) b2 (64,) W3 (64,64) b3 (64,) HW (6,64,10) Hb (6,10)`, all finite.
The pipeline ignores any other arrays in the file and never reads them. The file itself must be
a regular file of at most 20 MB. `mlp.save_merged()` writes the right format. The pipeline cannot
ship a file with missing arrays, wrong shapes or dtypes, NaN/inf values or pickled objects, or a
file that is a symlink or larger than 20 MB.

Only the weights file ships; none of your code runs on the device. You can spend your own
session's compute on producing it.
