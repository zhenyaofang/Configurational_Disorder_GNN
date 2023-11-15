import os
import pickle as pkl
import numpy as np
import copy
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

from pymatgen.core import Structure

import utilities

kB = 8.617 * 10**-5
NUM_ATOMS = 200

class WangLandau():
    def __init__(self, modelname='grid', unitcell='POSCAR', concentration=0.5, flatness_criterion=0.8, steps_check_flatness=1000, max_steps=50000,
                 Emin=float('-inf'), Emax=float('inf'), dE=0.01, max_distortion=0):
        self.MC_name = 'WL_' + modelname + '_' + str(concentration)
        utilities.generateSupercell(unitcell, target_length=15)
        self.parent_structure = unitcell
        self.model = self._loadModel(modelname)
        self.max_distortion = max_distortion
        self.steps_check_flatness = steps_check_flatness
        self.flatness_criterion = flatness_criterion
        self.logger = utilities.get_logger('./save/' + self.MC_name)
        self.max_steps = max_steps
        self.concentration = concentration

        self.E = (Emin, Emax, dE)
        self.Egrid = np.arange(Emin, Emax, dE)
        self.histogram = np.zeros(len(self.Egrid))
        self.lnDOS = np.zeros(len(self.Egrid))
        self.lnf = 1

    def _loadModel(self, modelname):
        return utilities.loadModel(modelname + '.pkl')

    def evaluateEnergy(self, graph):
        with torch.no_grad():
            energy = self.model(graph).numpy() * len(graph.x)
        return energy

    def getIndexByEnergy(self, energy):
        return int((energy - self.E[0]) // self.E[2])

    def getEnergyByIndex(self, index):
        return index * self.E[2] + self.E[0]

    def swap(self, graph):
        elem_list = np.array(graph.x)
        elem_kind = list(set(elem_list))
        index0 = np.random.choice(np.where(elem_list == elem_kind[0])[0])
        index1 = np.random.choice(np.where(elem_list == elem_kind[1])[0])
        graph.x[index0] = torch.tensor(elem_kind[1])
        graph.x[index1] = torch.tensor(elem_kind[0])

    def addRandomDistortion(self, structure):
        for j in range(len(structure)):
            displacement = np.random.random_sample(3) * self.max_distortion - self.max_distortion / 2
            structure[j].coords += displacement

    def accept(self, energy, energy_new):
        index = self.getIndexByEnergy(energy)
        index_new = self.getIndexByEnergy(energy_new)
        lnDOS = self.lnDOS[index]
        lnDOS_new = self.lnDOS[index_new]
        if lnDOS >= lnDOS_new:
            return True
        return np.exp(lnDOS - lnDOS_new) > np.random.uniform(0, 1)

    def runMC(self):
        start_configuration = copy.deepcopy(self.parent_structure)
        utilities.setConcentration(start_configuration, self.concentration)
        self.addRandomDistortion(start_configuration)
        graph_cur = utilities.structureToGraph(start_configuration)
        energy_cur = self.evaluateEnergy(graph_cur)

        for step_counter in tqdm(range(1, self.max_steps)):
            graph_next = copy.deepcopy(graph_cur)
            self.swap(graph_next)
            energy_next = self.evaluateEnergy(graph_next)
            print(energy_next)
            if self.accept(energy_cur, energy_next):
                energy_cur = energy_next
                graph_cur = graph_next

            index = self.getIndexByEnergy(energy_cur)
            self.lnDOS[index] += self.lnf
            self.histogram[index] += 1

            if step_counter % self.steps_check_flatness == 0:
                histogram = self.histogram[self.lnDOS > 0]
                print('histogram min = {}, max = {}, mean = {}.'.format(histogram.min(), histogram.max(), histogram.mean()))
                if len(histogram) >= 2 and (histogram > self.flatness_criterion * histogram.mean()).all():
                    self.histogram[:] = 0
                    self.lnf = self.lnf / 2
                    print('the flatness criterion is satisfied. The current modification factor is exp({})'.format(self.lnf))

        results = {
            'energy_levels': self.Egrid,
            'lnDOS': self.lnDOS
        }
        with open('./save/'+ self.MC_name + '_results.pkl', 'wb') as f:
            pkl.dump(results, f)

class PostProcessing():
    def __init__(self, MCfile=None, Tgrid=np.arange(10, 1000, 10), concentration=0.5):
        with open(MCfile, 'rb') as file:
            MC = pkl.load(file)

        self.Tgrid = Tgrid

        lnDOS = MC['lnDOS']
        lnDOS = lnDOS - lnDOS.min()
        DOS = np.exp(lnDOS - lnDOS.max()) / np.sum(np.exp(lnDOS - lnDOS.max()))
        DOS = DOS * scipy.special.comb(NUM_ATOMS, int(concentration * NUM_ATOMS)) / np.sum(DOS)
        nonzero_index = np.nonzero(DOS)[0][0]
        self.DOS = DOS[nonzero_index : ]

        Egrid = MC['energy_levels'][nonzero_index :]
        self.Egrid = Egrid - Egrid.min()


    def _getNormalizedDOS(self, lng):
        '''
        lnDOS = [ln g1, ln g2, ln g3]
        lnDOS_max = ln g3
        rel_DOS = [g1/g3, g2/g3, g3/g3 = 1]
        rel_DOS_sum = (g1 + g2 + g3) / g3
        normalized_DOS = [g1 / (g1 + g2 + g3), g2 / (g1 + g2 + g3), g3 / (g1 + g2 + g3)]
        '''
        lnDOS = np.array(list(lng.values()))
        lnDOS_max = np.max(lnDOS)

        rel_DOS = np.exp(lnDOS - lnDOS_max)
        rel_DOS_sum = np.sum(rel_DOS)
        normalized_DOS = rel_DOS / rel_DOS_sum
        return normalized_DOS

    def partitionFunction(self):
        Z = np.array([np.sum(self.DOS * np.exp(-self.Egrid / (kB * T))) for T in self.Tgrid])
        return Z

    def averageEnergy(self):
        Z = self.partitionFunction()
        E = np.array([np.sum(self.DOS * self.Egrid * np.exp(-self.Egrid / (kB * T))) for T in self.Tgrid]) / Z
        return E

    def averageEnergySqaured(self):
        Z = self.partitionFunction()
        E2 = np.array([np.sum(self.DOS * self.Egrid**2 * np.exp(-self.Egrid / (kB * T))) for T in self.Tgrid]) / Z
        return E2

    def heatCapacity(self):
        E2 = self.averageEnergySqaured()
        E = self.averageEnergy()
        return (E2 - E**2) / (kB * self.Tgrid**2)

    def freeEnergy(self):
        Z = self.partitionFunction()
        return  - kB * self.Tgrid * np.log(Z) + self.Egrid[0]

    def configurationalEntropy(self):
        E = self.averageEnergy()
        F = self.freeEnergy()
        return (E - F) / self.Tgrid
    


if __name__ == '__main__':
    structure = Structure.from_file('unitcell.vasp')
    modelname = 'distortion_best_model'
    concentration = 0.5

    # Emin = -830
    # Emax = -790
    # dE = (Emax - Emin) / 100
    # WL = WangLandau(modelname=modelname,
    #                 unitcell=structure,
    #                 concentration=concentration,
    #                 max_steps=int(1E8),
    #                 Emin=Emin,
    #                 Emax=Emax,
    #                 dE=dE,
    #                 max_distortion=0.2
    #                 )
    # WL.runMC()

    Tgrid = np.arange(10, 5000, 10)
    PP = PostProcessing(MCfile='./save/WL_mean_no_bn_0.5_results.pkl', Tgrid=Tgrid)

    x = PP.Egrid
    y = PP.DOS / np.sum(PP.DOS)

    def gaussian(x, A, mean, stddev):
        return A / (stddev * np.sqrt(2 * np.pi)) * np.exp(-(x - mean) ** 2 / (2 * stddev ** 2))

    params, cov = scipy.optimize.curve_fit(gaussian, x, y, p0=[1, 3, 1])
    A, mean, stddev = params
    Egrid = np.linspace(0, 15, 1000)
    DOS = gaussian(Egrid, A, mean, stddev)
    # print(params)
    # A = 0.5948, mean = 3.5526, stddev = 0.87303
    #
    S = PP.configurationalEntropy()
    Cv = PP.heatCapacity()

    # plt.bar(PP.Egrid, y, align='center', alpha=1, color='lightgrey')
    # plt.plot(Egrid, DOS, color='black', linewidth=2)
    # plt.xlim((0, 12))
    # plt.ylim((0, 0.3))

    plt.plot(Tgrid, S, 'k', label='entropy', linewidth=2)
    plt.xlim((10, 5000))
    axs = plt.gca()
    # axs.set_ylim((0.01, 0.02))
    # axs.set_aspect('equal')
    axs1b = axs.twinx()
    axs1b.plot(Tgrid, Cv, 'r', label='heat capacity', linewidth=2)

    lines, labels = axs.get_legend_handles_labels()
    lines2, labels2 = axs1b.get_legend_handles_labels()
    axs1b.legend(lines + lines2, labels + labels2, loc='upper left', prop={'size': 10})

    plt.show()
