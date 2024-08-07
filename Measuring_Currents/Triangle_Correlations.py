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
basis_labels = ['1a', '1b', '2a', '2b', '3a', '3b', '4a', '4b']
# basis_labels = ['1a', '1b', '2a', '2b'] #, '4a', '4b']


g = -5 * 2 * np.pi    #In MHz and time in us
U= -150 * 2 * np.pi
e = 0

coupling_dictionary_NN = {'aa': 2 * g, 'ba': g, 'bb': 2 * g}     # Coupling to nearest unit cell
coupling_dictionary_same = {'ab' : g}  #Coupling within a unit cell
coupling_periodic = False #{'aa': -3 * g, 'ba': g, 'bb': -3 * g}


#one photon basis states
basis_states = RecursiveBasisState(4, basis_labels, state = '')
# initial_state = state_generation(basis_states, {'1a1a1a': (0.013690768-0j), '1a1a1b': (0.0755303879-0j), '1a1a2a': (-0.0484900235-0j), '1a1a2b': (-0.0001712278-0j), '1a1a3a': (-0.0483280617-0j), '1a1a3b': (-0.0753999194-0j), '1a1b1b': (0.0348905793-0j), '1a1b2a': (-0.0005168125-0j), '1a1b2b': (0.1487884197-0j), '1a1b3a': (-0.2808469975-0j), '1a1b3b': (-0.2752951269-0j), '1a2a2a': (-0.0483280617-0j), '1a2a2b': (-0.2808469975-0j), '1a2a3a': (0.2692711695-0j), '1a2a3b': (0.2814076211-0j), '1a2b2b': (-0.0912094311-0j), '1a2b3a': (0.2814076211-0j), '1a2b3b': (0.1498439421-0j), '1a3a3a': (-0.0484900235-0j), '1a3a3b': (-0.0005168125-0j), '1a3b3b': (0.0346072373-0j), '1b1b1b': (2.57017e-05-0j), '1b1b2a': (0.0346072373-0j), '1b1b2b': (0.04304595-0j), '1b1b3a': (-0.0912094311-0j), '1b1b3b': (-0.0432277069-0j), '1b2a2a': (-0.0753999194-0j), '1b2a2b': (-0.2752951269-0j), '1b2a3a': (0.2814076211+0j), '1b2a3b': (0.1498439421-0j), '1b2b2b': (-0.0432277069-0j), '1b2b3a': (0.1498439421-0j), '1b2b3b': (0.0005055037-0j), '1b3a3a': (-0.0001712278-0j), '1b3a3b': (0.1487884197-0j), '1b3b3b': (0.04304595-0j), '2a2a2a': (0.013690768-0j), '2a2a2b': (0.0755303879-0j), '2a2a3a': (-0.0484900235-0j), '2a2a3b': (-0.0001712278-0j), '2a2b2b': (0.0348905793-0j), '2a2b3a': (-0.0005168125-0j), '2a2b3b': (0.1487884197-0j), '2a3a3a': (-0.0483280617-0j), '2a3a3b': (-0.2808469975-0j), '2a3b3b': (-0.0912094311-0j), '2b2b2b': (2.57017e-05-0j), '2b2b3a': (0.0346072373-0j), '2b2b3b': (0.04304595-0j), '2b3a3a': (-0.0753999194-0j), '2b3a3b': (-0.2752951269-0j), '2b3b3b': (-0.0432277069-0j), '3a3a3a': (0.013690768-0j), '3a3a3b': (0.0755303879-0j), '3a3b3b': (0.0348905793-0j), '3b3b3b': (2.57017e-05-0j)}
# )
initial_state = -1
print(len(basis_states))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

eigenstates = False
max_number_of_states = 0

analytic_correlation = True

current_correlations = [[['1a', '1b'], ['2a', '2b']],
                        [['1a', '1b'], ['3a', '3b']],
                        [['1a', '1b'], ['4a', '4b']],
                        ]

# current_correlations = [
#                         [['1a', '1b'], ['3a', '2b']],
#                         [['1a', '1b'], ['4a', '3b']],
#                         [['1a', '1b'], ['5a', '4b']]
#                         ]
# current_correlations = [['1a', '2a'], ['1b', '2b']]

# correlation_values = {'1a1b': 1, '1b2a': -1, '1a2b': -1, '2a2b': 1}
energy_map = {}
for key in basis_labels:
    if key not in energy_map.keys():
        energy_map[key] = e + 0 * 3 * np.random.rand(1)

energy_map.update((key, value * 2 * np.pi) for key, value in energy_map.items())#

H = Hamiltonian(basis_states, U, energy_map, coupling_dictionary_same,
                         coupling_dictionary_NN, coupling_periodic)
Evolution_H = copy.copy(H.H)
print('Done Making Matrix')

if analytic_correlation:
    Current_obj = Current_Operator(basis_states, basis_labels, ['1a', '1b'], ['1c', '2a'])

    if type(initial_state) == int or type(initial_state) == float:
        eigval, eigvect = np.linalg.eig(H.H)
        print('Done Eigenvalues')

        sorted_eig = []
        for k, val in enumerate(eigval):
            sorted_eig.append((val, k))
        sorted_eig.sort()
        initial_state_vector = eigvect[:, sorted_eig[initial_state][1]]
        print(np.array(sorted_eig[-3:]) / 2/ np.pi)
        initial_state = normalize_eigenstate(initial_state_vector)
    print('Done Sorting Eigenvalues')
    correlations = []
    for i in range(len(current_correlations)):
        initial_state = copy.copy(initial_state)
        op1 = Current_obj.operator_measurement(initial_state, [current_correlations[i][0][0], current_correlations[i][1][0]],
                                               [current_correlations[i][0][1], current_correlations[i][1][1]])
        op2 = Current_obj.operator_measurement(initial_state, [current_correlations[i][0][0], current_correlations[i][1][1]],
                                               [current_correlations[i][0][1], current_correlations[i][1][0]])
        op3 = Current_obj.operator_measurement(initial_state, [current_correlations[i][0][1], current_correlations[i][1][0]],
                                               [current_correlations[i][0][0], current_correlations[i][1][1]])
        op4 = Current_obj.operator_measurement(initial_state, [current_correlations[i][0][1], current_correlations[i][1][1]],
                                               [current_correlations[i][0][0], current_correlations[i][1][0]])

        correlations.append(-1 * (op1 - op2 - op3 + op4))
        print(correlations[i])

    correlations = np.array(correlations)
    plt.figure(figsize=([15, 8]))
    plt.plot(range(1, len(correlations) + 1), np.array(correlations).real, 'o-')
    # plt.plot(updated_times[:None], Time_correlations.imag[:None], label = 'Imag')
    plt.xlabel('Rung Steps')
    plt.ylabel('Current Correlation')
    plt.title('Analytic')
    plt.ylim(min(correlations.real) - 0.1, max(correlations.real) + 0.1)
    plt.show()

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

    print(np.round(np.array(np.real(sorted_eig))[:, 0] / (2 * np.pi), 5))
    for i in range(len(sorted_eig)):
        if i > max_number_of_states:
            continue
        print(np.round(np.real(sorted_eig[i][0]) / (2 * np.pi), 5))
        dictionary_eigenstate_initial = {}
        for j in range(len(basis_states)):
            print(basis_states[j] + ': ', np.round(eigvect[j, sorted_eig[i][1]] , 10))
            dictionary_eigenstate_initial[basis_states[j]] = np.round(eigvect[j, sorted_eig[i][1]] , 10)
        print(dictionary_eigenstate_initial)