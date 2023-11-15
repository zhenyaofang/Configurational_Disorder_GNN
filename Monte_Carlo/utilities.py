import os
import io
import copy
import random
import numpy as np
import pickle as pkl
import logging

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

import torch
import torch_geometric
from torch_geometric.data import Data

R_CUTOFF = 10
MAXNEIGHBORS = 20

class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def get_logger(name):
    logger = logging.getLogger(name)
    filename = name + '.log'
    fh = logging.FileHandler(filename, mode='w')
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    return logger

def calculateEdgeAttributes(dist, dr):
    if dr == 0:
        return dist
    else:
        rgrid = np.arange(0, R_CUTOFF, dr)
        sigma = R_CUTOFF / 3
        attr = np.exp(-0.5 * (rgrid - dist)**2 /sigma**2) / np.sqrt(2 * np.pi) / sigma
        return attr

def structureToGraph(structure, dr=0.1):
    atom_dict = {k: k for k in range(100)}
    # structure = Structure.from_file(file_loc)
    neighbors = structure.get_all_neighbors(R_CUTOFF, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x.nn_distance) for nbrs in neighbors]

    nbr_fea_idx, nbr_fea = [], []
    max_neighbors = min(MAXNEIGHBORS, len(all_nbrs[0]))
    for nbr in all_nbrs:
        nbr_fea_idx.append(list(map(lambda x: x.index, nbr[ : max_neighbors])))
        nbr_fea.append(list(map(lambda x: x.nn_distance, nbr[ : max_neighbors])))

    x = []
    edge_index = []
    edge_attr = []
    for i in range(len(structure.atomic_numbers)):
        elemi = atom_dict[structure.atomic_numbers[i]]
        x.append(elemi)
        for j in range(len(nbr_fea_idx[i])):
            edge_index.append([i, nbr_fea_idx[i][j]])
            edge_attr.append(calculateEdgeAttributes(nbr_fea[i][j], dr=dr))

    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(np.array(edge_index), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros((data.x.shape[0]), dtype=int)
    # assert data.validate(raise_on_error=False), 'structure file ' + file_loc + ' cannot be converted into Data object.'
    return data

def graphToStructure(graph, pristine_structure):
    atoms = graph.x
    assert len(atoms) == pristine_structure.num_sites, "Cannot convert the graph into a structure"

    structure = copy.deepcopy(pristine_structure)
    for i in range(len(atoms)):
        structure[i].species = Element.from_Z(atoms[i])
    structure.sort()
    return structure

def generateSupercell(unitcell, target_length):
    l = np.array([unitcell.lattice.a, unitcell.lattice.b, unitcell.lattice.c])
    n = np.rint(target_length / l)
    unitcell.make_supercell(n)

def setConcentration(cell, concentration):
    Cu, Au = cell.types_of_species[0], cell.types_of_species[1]
    num_Au = int(cell.num_sites * concentration)
    indices = random.sample(range(cell.num_sites), num_Au)
    for j in range(cell.num_sites):
        if j in indices:
            cell[j].species = Au
        else:
            cell[j].species = Cu

def loadModel(filename):
    assert filename in os.listdir('./save'), 'model {} does not exist'.format(filename)
    with open('./save/' + filename, 'rb') as f:
        model = CPU_Unpickler(f).load()
    return model

def getEntropy(self):
    energies = np.array(self.energies)
    energies = energies - np.min(energies)
    T = self.temperature
    factor = np.exp(-energies / (kB * T))

    Z = np.sum(factor)
    E = np.sum(energies * factor) / Z
    F = -kB * T * np.log(Z)
    S = (E - F) / T
    return S


def getOrderParameter(self, index=None):
    ordered = np.array(self.ordered.atomic_numbers)
    if index == None:
        disordered = np.array(self.getGroundStateStructure().atomic_numbers)
        return 1 - np.sum(ordered == disordered) / len(ordered)
    elif isinstance(index, int):
        disordered = np.array(utilities.graphToStructure(self.graphs[index], self.ordered).atomic_numbers)
        return 1 - np.sum(ordered == disordered) / len(ordered)
    elif isinstance(index, tuple):
        structures = [utilities.graphToStructure(self.graphs[i], self.ordered) for i in range(index[0], index[1])]
        disordered = [np.array(struct.atomic_numbers) for struct in structures]
        order_parameters = [1 - np.sum(ordered == dis) / len(ordered) for dis in disordered]
        return np.mean(order_parameters)

if __name__ == '__main__':
    unitcell = Structure.from_file('unitcell.vasp')
    generateSupercell(unitcell, 15)
    setConcentration(unitcell, 0.201)
    print(unitcell)
