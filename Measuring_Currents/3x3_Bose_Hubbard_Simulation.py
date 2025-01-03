from Hamiltonian_Generation_Codebase._BaseCodeHam import *

from Hamiltonian_Generation_Codebase.state_initialization import *
from Hamiltonian_Generation_Codebase.time_evolution import *
from Hamiltonian_Generation_Codebase.Measurement_Functions import *

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 25})
plotsize = (10, 6)
legend_size = 12
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = 'tight'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
basis_labels = ['1a', '1b', '1c', '1d', '2a', '2b', '2c', '2d', '3a', '3b', '3c', '3d', '4a', '4b', '4c', '4d']
# basis_labels = ['1a', '1b', '2a', '2b']
basis_labels = ['1a', '1b', '2a', '2b', '3a', '3b']



g = -20 * 2 * np.pi    #In MHz and time in us
U= -200 * 2 * np.pi
# U= -70 * 2 * np.pi
# U= -30 * 2 * np.pi


e = 3000

coupling_dictionary_NN = {'aa': g, 'bb': g, 'cc': g, 'dd': g}     # Coupling to nearest unit cell
# coupling_dictionary_NN = {'aa': 2.5 * g, 'ba': g, 'bb': 2.5 * g}     # Coupling to nearest unit cell

coupling_dictionary_same = {'ab' : g, 'bc': g, 'cd': g}  #Coupling within a unit cell
coupling_periodic = False #{'aa': -3 * g, 'ba': g, 'bb': -3 * g}


#one photon basis states
basis_states = RecursiveBasisState(3, basis_labels, state = '', hardcore = True)
# initial_state = state_generation(basis_states, {'1a1a1a': (0.013690768-0j), '1a1a1b': (0.0755303879-0j), '1a1a2a': (-0.0484900235-0j), '1a1a2b': (-0.0001712278-0j), '1a1a3a': (-0.0483280617-0j), '1a1a3b': (-0.0753999194-0j), '1a1b1b': (0.0348905793-0j), '1a1b2a': (-0.0005168125-0j), '1a1b2b': (0.1487884197-0j), '1a1b3a': (-0.2808469975-0j), '1a1b3b': (-0.2752951269-0j), '1a2a2a': (-0.0483280617-0j), '1a2a2b': (-0.2808469975-0j), '1a2a3a': (0.2692711695-0j), '1a2a3b': (0.2814076211-0j), '1a2b2b': (-0.0912094311-0j), '1a2b3a': (0.2814076211-0j), '1a2b3b': (0.1498439421-0j), '1a3a3a': (-0.0484900235-0j), '1a3a3b': (-0.0005168125-0j), '1a3b3b': (0.0346072373-0j), '1b1b1b': (2.57017e-05-0j), '1b1b2a': (0.0346072373-0j), '1b1b2b': (0.04304595-0j), '1b1b3a': (-0.0912094311-0j), '1b1b3b': (-0.0432277069-0j), '1b2a2a': (-0.0753999194-0j), '1b2a2b': (-0.2752951269-0j), '1b2a3a': (0.2814076211+0j), '1b2a3b': (0.1498439421-0j), '1b2b2b': (-0.0432277069-0j), '1b2b3a': (0.1498439421-0j), '1b2b3b': (0.0005055037-0j), '1b3a3a': (-0.0001712278-0j), '1b3a3b': (0.1487884197-0j), '1b3b3b': (0.04304595-0j), '2a2a2a': (0.013690768-0j), '2a2a2b': (0.0755303879-0j), '2a2a3a': (-0.0484900235-0j), '2a2a3b': (-0.0001712278-0j), '2a2b2b': (0.0348905793-0j), '2a2b3a': (-0.0005168125-0j), '2a2b3b': (0.1487884197-0j), '2a3a3a': (-0.0483280617-0j), '2a3a3b': (-0.2808469975-0j), '2a3b3b': (-0.0912094311-0j), '2b2b2b': (2.57017e-05-0j), '2b2b3a': (0.0346072373-0j), '2b2b3b': (0.04304595-0j), '2b3a3a': (-0.0753999194-0j), '2b3a3b': (-0.2752951269-0j), '2b3b3b': (-0.0432277069-0j), '3a3a3a': (0.013690768-0j), '3a3a3b': (0.0755303879-0j), '3a3b3b': (0.0348905793-0j), '3b3b3b': (2.57017e-05-0j)}
# )
initial_state = 0
print(len(basis_states))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

eigenstates = True
only_eigenvalues = True
print_populations = True
eigenstate_population = -10
values_to_see = basis_labels
eigenvalue_start = 0
eigenvalue_end = None

max_number_of_states = 0
# onsite_disorder = 5

energy_map = {'1a': 200}
for key in basis_labels:
    if key not in energy_map.keys():
        energy_map[key] = e

energy_map.update((key, value * 2 * np.pi) for key, value in energy_map.items())#

print(energy_map)

H = Hamiltonian(basis_states, U, energy_map, coupling_dictionary_same,
                         coupling_dictionary_NN, coupling_periodic)
Evolution_H = copy.copy(H.H)

if eigenstates:
    print('Hamiltonian real:')
    print(np.round(H.H.real / (2 * np.pi), 3))
    print('Hamiltonian imaginary:')
    print(np.round(H.H.imag / (2 * np.pi), 3))

    eigval, eigvect = np.linalg.eig(H.H)
    sorted_eig = []
    for k, val in enumerate(eigval):
        sorted_eig.append((val, k))
    sorted_eig.sort()

    # if Fock_Space_Measure:
    #     all_exp_values = Expectation_values(sol, values_to_see, basis_states).real
    #     vmax = 1
    # else:
    #     values_to_see = basis_labels
    #     all_exp_values = Expectation_values_basis_labels(sol, values_to_see, basis_states).real
    #     vmax = len(basis_states[0]) / len(basis_labels[0])

    print(np.round(np.array(np.real(sorted_eig[eigenvalue_start: eigenvalue_end]))[:, 0] / (2 * np.pi), 5))

    plt.plot(np.array(np.real(sorted_eig[eigenvalue_start: eigenvalue_end]))[:, 0] / (2 * np.pi), '.')
    plt.show()
    if not only_eigenvalues:
        for i in range(len(sorted_eig)):
            if i > max_number_of_states:
                continue
            print(np.round(np.real(sorted_eig[i][0]) / (2 * np.pi), 5))
            dictionary_eigenstate_initial = {}
            for j in range(len(basis_states)):
                print(basis_states[j] + ': ', np.round(eigvect[j, sorted_eig[i][1]] , 10))
                dictionary_eigenstate_initial[basis_states[j]] = np.round(eigvect[j, sorted_eig[i][1]] , 10)
            print(dictionary_eigenstate_initial)

    if print_populations:
        state = eigvect[:, sorted_eig[eigenstate_population][1]]
        populations = Expectation_values_basis_labels_single_state(state, values_to_see, basis_states)
        for i, label in enumerate(values_to_see):
            print(label + ': ' + str(np.round(populations[i], 3)))