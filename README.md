# DHL
We propose the **D**ual-space **H**ierarchical **L**earning (DHL) to leverage multi-level goal sequences and their hierarchical relationships for conversational recommendation.

## Install
```bash
pip install jieba
pip install higher
pip install torch
```

## Experiments

```bash
python -u DHL.py --mode mw-h-mlp --attention both
```

You could change the above command to reproduce the experimental results in the Ablation Study section.
