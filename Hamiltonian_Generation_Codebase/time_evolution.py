import numpy as np
import pickle
import copy

import scipy
from scipy.integrate import odeint
from odeintw import odeintw

def density_matrix(state):
    return(np.outer(state, state))

def function(rho, t, Ham):
    return(1j * (np.matmul(rho, Ham) - np.matmul(Ham, rho)))


import scipy
from scipy.integrate import odeint
from odeintw import odeintw
def function_vector(state, t, Ham):
    return(-1j * Ham.dot(state))

def States_time_evolution(Hamiltonian, time_list, initial_state):
    sol = odeintw(function_vector, initial_state, time_list, args=(Hamiltonian,), full_output = 1)[0]
    return(sol)

def Expectation_values(time_evolutions, labels_to_find, basis):
    '''
    Measure expectation values of the basis_states For example measures in fock space land
    if you want to measure '1a1a', labels to find must be '1a1a'. Otherwise it will not find it
    Similarly, if your basis is called '1a1b', it will not find it if you look up '1b1a'
    :param time_evolutions: State vectors as a function of time
    :param labels_to_find: fock space labels you want to extract
    :param basis: all of the basis_states
    :return: Array of fock space number operators
    '''
    all_expectation_values = [[] for i in range(len(labels_to_find))]
    time_evolutions_amplitude = (time_evolutions.conj() * time_evolutions).real
    basis = np.array(basis)
    for i, label in enumerate(labels_to_find):
        index = np.where(basis == label)[0][0]

        all_expectation_values[i].append(time_evolutions_amplitude[:, index])
    return(np.array(all_expectation_values)[:, 0])

def Expectation_values_basis_labels_OLDVersion(time_evolutions, labels_to_find, basis_states):
    # Question:
    # is labels_to_find specific to what I want
    all_expectation_values = [[] for i in range(len(labels_to_find))]
    time_evolutions_amplitude = (time_evolutions.conj() * time_evolutions).real #amplitude squared
    basis = np.array(basis_states)

    for i, label in enumerate(labels_to_find):
        value = np.zeros(len(time_evolutions_amplitude))
        for j, base in enumerate(basis_states):
            separated_base = [base[i:i + len(labels_to_find[0])] for i in range(0, len(base), len(labels_to_find[0]))]
            count = separated_base.count(label)  #ex 1a1a would count 2 for two photons
            value += count * time_evolutions_amplitude[:, j]
        all_expectation_values[i].append(value)

        # index = np.where(basis == label)[0][0]
        # all_expectation_values[i].append(time_evolutions_amplitude[:, index])

    return(np.array(all_expectation_values)[:, 0])

def Expectation_values_basis_labels(time_evolutions, labels_to_find, basis_states):
    all_expectation_values = [[] for i in range(len(labels_to_find))]
    time_evolutions_amplitude = (time_evolutions.conj() * time_evolutions).real #amplitude squared
    for i, label in enumerate(labels_to_find):
        value = np.zeros(len(time_evolutions_amplitude)) #zero for each time step
        if type(label) == list:
            for j, base in enumerate(basis_states):
                separated_base = [base[i:i + len(labels_to_find[0][0])] for i in range(0, len(base), len(labels_to_find[0][0]))]
                count = 1
                for ind_label in label:
                    count *= separated_base.count(ind_label)
                value += count * time_evolutions_amplitude[:, j]  # Count just puts the occupation number
        else:
            for j, base in enumerate(basis_states):
                separated_base = [base[i:i + len(labels_to_find[0])] for i in range(0, len(base), len(labels_to_find[0]))]
                count = separated_base.count(label)  #ex 1a1a would count 2 for two photons
                value += count * time_evolutions_amplitude[:, j] #Count just puts the occupation number
        all_expectation_values[i].append(value)
    return(np.array(all_expectation_values)[:, 0])

# photon_number = separated_list_diag.count(single)
# return ([single_basis[i:i + self.n] for i in range(0, len(single_basis), self.n)])
