import numpy as np
import pickle
import copy

import scipy
from scipy.integrate import odeint
from odeintw import odeintw

class Hamiltonian:
    def __init__(self, bases, U, energy_mapping, coupling_dict_samecell,
                 coupling_dict_difcell, periodic=False, n=2):
        self.bases = bases
        self.U = U
        self.energy_mapping = energy_mapping
        self.coupling_dict_samecell = coupling_dict_samecell
        self.coupling_dict_difcell = coupling_dict_difcell
        self.periodic = periodic
        self.n = n
        self.num_unitcells = int(bases[-1][0])

        self.H = np.zeros((len(self.bases), len(self.bases)),dtype=complex)

        self.generate_cross_products()

    def differences_between_the_two(self, separated_list1, separated_list2):
        '''
        from two lists of string values, it will return a list that contains the elements that the two lists differ by

        '''
        sep_list1 = separated_list1.copy()
        sep_list2 = separated_list2.copy()

        overlap_list = list(set(sep_list1) & set(sep_list2))  # remove repeats. Not necessary in this case

        while len(overlap_list) > 0:
            for x in overlap_list:
                sep_list1.remove(x)
                sep_list2.remove(x)
            overlap_list = list(set(sep_list1) & set(sep_list2))
        return (sep_list1, sep_list2)

    def separated_list(self, single_basis):
        '''
        splits the basis into a list of individual states

        '1u1d' -> ['1u', '1d']
        '''
        return ([single_basis[i:i + self.n] for i in range(0, len(single_basis), self.n)])

    def differences_func(self, basis1, basis2):
        basis1_sep = self.separated_list(basis1)
        basis2_sep = self.separated_list(basis2)

        return (self.differences_between_the_two(basis1_sep, basis2_sep))

    def compare_differences(self, differences):
        if len(differences[0]) != 1 or len(differences[1]) != 1:
            return (0)
        val1, val2 = differences[0][0], differences[1][0]
        val1_label = val1[-1]
        val1_index = int(val1[0:-1])

        val2_label = val2[-1]
        val2_index = int(val2[0:-1])
        if abs(val1_index - val2_index) == 0:
            if val1_label + val2_label in self.coupling_dict_samecell:
                return (self.coupling_dict_samecell[val1_label + val2_label])
            if val2_label + val1_label in self.coupling_dict_samecell:
                return (self.coupling_dict_samecell[val2_label + val1_label])
            return (0)
        if np.abs(val2_index - val1_index) == 1:
            if val1_label + val2_label in self.coupling_dict_difcell and val2_index > val1_index:
                return (self.coupling_dict_difcell[val1_label + val2_label])
            if val2_label + val1_label in self.coupling_dict_difcell and val1_index > val2_index:
                return (self.coupling_dict_difcell[val2_label + val1_label])
            return (0)
        # print(val2_index, val1_index, self.num_unitcells)
        # if val2_index - val1_index + 1 == self.num_unitcells and self.periodic != None:
        if val2_index - val1_index + 1 == self.num_unitcells and self.periodic != False:

            # print(val1_label, val2_label, self.periodic)
            if val2_label + val1_label in self.periodic:
                return (self.periodic[val2_label + val1_label])
            # if val1_label + val2_label in self.periodic:
            #     return (self.periodic[val1_label + val2_label])
            return (0)
        return (0)

    def measure_coefficient(self, differences, separated_second):
        coefficient1 = separated_second.count(differences[1][0])
        coefficient2 = separated_second.count(differences[0][0]) + 1
        return (np.sqrt(coefficient1 * coefficient2))

    def diagonal(self, basis, disorder = False):
        total_energy = 0
        separated_list_diag = self.separated_list(basis)
        for val in separated_list_diag:
            # val_label = val[1]
            # total_energy += (self.energy_mapping[val_label])
            total_energy += (self.energy_mapping[val])
            if disorder:
                # print(val, self.disorder_energy_map[val])
                total_energy += (self.disorder_energy_map[val])

        for single in set(separated_list_diag):
            photon_number = separated_list_diag.count(single)
            total_energy += (self.U / 2) * photon_number * (photon_number - 1)

        #Short version of this..
        # for val in set(separated_list_diag):
        #     photon_number = separated_list_diag.count(val)
        #     val_label = val[1]
        #     onsite_energy = self.energy_mapping[val_label]
        #     total_energy += photon_number * onsite_energy + (self.U / 2) * photon_number * (photon_number - 1)
        #     if disorder:
        #         total_energy += (self.disorder_energy_map[val])

        return (total_energy)

    def generate_disorder(self, disorder_energy_map):
        '''
        Generates the on-site energy diagonal terms of the Hamiltonian
        :param disorder_energy_map: dictionary where the keys are the basis labels and values are onsite energies
        '''
        self.disorder_energy_map = disorder_energy_map
        for i, state1 in enumerate(self.bases):
            self.H[i, i] = self.diagonal(state1, True)

    def generate_onsite(self):
        '''
        Generates the on-site energy diagonal terms of the Hamiltonian
        :param disorder_energy_map: dictionary where the keys are the basis labels and values are onsite energies
        '''
        for i, state1 in enumerate(self.bases):
            self.H[i, i] = self.diagonal(state1, False)

    def generate_cross_products(self):
        '''
        Generates the Hamiltonian

        basis_states is a list of strings that has all of the basis states for a given Hamiltoinan subset you
        want to generate

        U is a float- whatever you set U to be

        t is a float- whatever you set t to be

        periodic is boolean True of False depending if you want periodic boundary condition or not. If not
        if only allows nearest neighbor hopping. If True, it will allow  hopping from lattice site 1 to 4

        Returns the matrix
        '''
        for i, state1 in enumerate(self.bases):
            for j, state2 in enumerate(self.bases[i:]):
                # print(j, state1, state2)
                if j == 0:
                    self.H[i, i] = self.diagonal(state1)
                    continue
                if len(state1) != len(state2):
                    self.H[i, i + j] = 0
                    self.H[i + j, i] = 0
                    continue

                separated1 = self.separated_list(state1)
                separated2 = self.separated_list(state2)

                differences = self.differences_between_the_two(separated1, separated2)
                couplingterm = self.compare_differences(differences)

                if couplingterm != 0:
                    couplingterm *= self.measure_coefficient(differences, separated2)
                # print(state1, state2, couplingterm)
                self.H[i, i + j] = couplingterm
                self.H[i + j, i] = couplingterm.conjugate()
        return (self.H)


def RecursiveBasisState(number_of_recursive, single_basis_state, state = ''):
    empty_list = []
    if number_of_recursive == 0:
        return(state)
    for i, single_element in enumerate(single_basis_state):
        mixed_state = RecursiveBasisState(number_of_recursive - 1, single_basis_state[i: ],
                                          state + single_element)
        if type(mixed_state) == list:
            for element in mixed_state:
                empty_list.append(element)
        else:
            empty_list.append(mixed_state)
    return(empty_list)

def H_beamsplitter(current_basis_states, detuning_amount, H, energy_map):
    energy_map = copy.copy(energy_map)
    for key in energy_map.keys():
        if key not in current_basis_states:
            energy_map[key] = detuning_amount
            continue
        energy_map[key] = 0
    H.energy_mapping = energy_map
    H.generate_onsite()
    return(H.H)

def H_Beamsplitter_Correlation(current_basis_states, detuning_amount, H, energy_map):
    '''
    Asssumption that your current correlations
    :param current_basis_states:
    :param detuning_amount:
    :param H:
    :param energy_map:
    :return:
    '''
    energy_map = copy.copy(energy_map)
    for key in energy_map.keys():
        if key not in current_basis_states[0]:
            energy_map[key] = detuning_amount / 2
            continue
        energy_map[key] = -detuning_amount / 2
    H.energy_mapping = energy_map
    H.generate_onsite()
    return(H.H)

class Current_Operator():
    def __init__(self, basis_states, basis_labels, current_raising, current_lowering):
        self.basis_states = basis_states
        self.basis_labels = basis_labels
        self.current_raising = current_raising
        self.current_lowering = current_lowering
        self.state = np.zeros(len(basis_states))
        self.n = len(basis_labels[0])

    def separated_list(self, single_basis):
        '''
        splits the basis into a list of individual states

        '1u1d' -> ['1u', '1d']
        '''
        return ([single_basis[i:i + self.n] for i in range(0, len(single_basis), self.n)])

    def count_instances(self, separated_basis, location_label, additional = 0):
        return(np.sqrt(separated_basis.count(location_label) + additional))

    def find_index(self, individual_basis):
        return(self.basis_states.index(individual_basis))

    def first_current(self, state):
        state_after = np.zeros(len(self.basis_states), dtype=np.complex_)

        for i, val in enumerate(state):
            sep_ind = self.separated_list(self.basis_states[i])
            mult_raising = 1
            for c_rais in self.current_raising:
                number_inst = self.count_instances(sep_ind, c_rais, additional = 1)
                mult_raising *= number_inst
                sep_ind.append(c_rais)
                sep_ind.sort()

            for c_low in self.current_lowering:
                number_inst = self.count_instances(sep_ind, c_low)
                if number_inst == 0:
                    mult_raising = 0
                    mult_raising *= number_inst
                    continue
                mult_raising *= number_inst
                sep_ind.remove(c_low)
                sep_ind.sort()

            if mult_raising != 0:
                index_of_new_index = self.find_index(''.join(sep_ind))
                state_after[index_of_new_index] = val * mult_raising

        return(np.array(state_after))

    def second_current(self, state):
        state_after = np.zeros(len(self.basis_states), dtype=np.complex_)

        for i, val in enumerate(state):
            sep_ind = self.separated_list(self.basis_states[i])
            mult_raising = 1
            for c_rais in self.current_lowering:
                number_inst = self.count_instances(sep_ind, c_rais, additional = 1)
                if number_inst == 0:
                    mult_raising = 0
                    continue
                mult_raising *= number_inst
                sep_ind.append(c_rais)
                sep_ind.sort()
            for c_low in self.current_raising:
                number_inst = self.count_instances(sep_ind, c_low)
                if number_inst == 0:
                    mult_raising = 0
                    continue
                mult_raising *= number_inst
                sep_ind.remove(c_low)
                sep_ind.sort()
            if mult_raising != 0:
                index_of_new_index = self.find_index(''.join(sep_ind))
                state_after[index_of_new_index] = val * mult_raising
        return(np.array(state_after))

    def conj_mult(self, state, new_state):
        state = np.array(state)
        return(np.inner(state.conj(), new_state))

    def current_calculation(self, state):
        first_state_meas = self.first_current(state)
        second_state_meas = self.second_current(state)
        return(self.conj_mult(state, first_state_meas) - self.conj_mult(state, second_state_meas))

    def operator_measurement(self, state, raising_operators, lowering_operators):
        self.current_raising = raising_operators
        self.current_lowering = lowering_operators
        final_state = self.first_current(state)
        return(self.conj_mult(state, final_state))


