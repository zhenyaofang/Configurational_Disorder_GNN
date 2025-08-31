# Towards accurate prediction of configurational disorder properties in materials using graph neural networks

## Description

This work combines graph neural networks and Monte Carlo simulations to predict configurational disorder properties in materials.

## Dependencies

Some key dependencies include:

- `python==3.11`
- `torch==2.3.1`
- `torch_geometric==2.5.3`

## Usage

For graph neural network training and testing, navigate to the `GNN/` directory, and run the script:
```bash
python GNN.py
```

For Monte-Carlo simulation, nativate to the `MC/` directory, and run the script:
```bash
python Wang_Landau.py
```

## Citation

If you find this work useful, please consider cite the following reference:

Fang, Z., Yan, Q. Towards accurate prediction of configurational disorder properties in materials using graph neural networks. npj Comput Mater 10, 91 (2024). https://doi.org/10.1038/s41524-024-01283-w
