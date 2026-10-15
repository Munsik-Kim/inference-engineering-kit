"""Frozen post-hoc target and contrasts; never generic core policy."""
H = 'model.layers.7.mlp.down_proj'
ARMS = ('R00', 'R10', 'R01', 'R11', 'BF16_REF', 'GPTQ_REF')
COEFFICIENTS = {
    'H_without_R': {'R10': 1, 'R00': -1},
    'H_with_R': {'R11': 1, 'R01': -1},
    'R_without_H': {'R01': 1, 'R00': -1},
    'R_with_H': {'R11': 1, 'R10': -1},
    'full_clipping': {'R11': 1, 'R00': -1},
    'interaction': {'R11': 1, 'R10': -1, 'R01': -1, 'R00': 1},
}
