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

g = 10 * 2 * np.pi    #In MHz and time in us
U= 150 * 2 * np.pi
e = 0

coupling_dictionary_NN = {'aa': -3 * g, 'ba': g, 'bb': -3 * g}     # Coupling to nearest unit cell
coupling_dictionary_same = {'ab' : 1*g}  #Coupling within a unit cell
coupling_periodic = {'aa': 3 * g, 'ba': g, 'bb': 3 * g}

coupling_dictionary_NN = {'aa': -3 * g, 'ba': g, 'bb': -3 * g}     # Coupling to nearest unit cell
coupling_dictionary_same = {'ab' : g}  #Coupling within a unit cell
coupling_periodic = False #{'aa': 3 * g, 'ba': g, 'bb': 3 * g}


#one photon basis states
basis_states = RecursiveBasisState(4, basis_labels, state = '')
values_to_see = basis_states #plot all of the photon fock states
# initial_state = state_generation(basis_states, {'1a1a1a': (0.013690768-0j), '1a1a1b': (0.0755303879-0j), '1a1a2a': (-0.0484900235-0j), '1a1a2b': (-0.0001712278-0j), '1a1a3a': (-0.0483280617-0j), '1a1a3b': (-0.0753999194-0j), '1a1b1b': (0.0348905793-0j), '1a1b2a': (-0.0005168125-0j), '1a1b2b': (0.1487884197-0j), '1a1b3a': (-0.2808469975-0j), '1a1b3b': (-0.2752951269-0j), '1a2a2a': (-0.0483280617-0j), '1a2a2b': (-0.2808469975-0j), '1a2a3a': (0.2692711695-0j), '1a2a3b': (0.2814076211-0j), '1a2b2b': (-0.0912094311-0j), '1a2b3a': (0.2814076211-0j), '1a2b3b': (0.1498439421-0j), '1a3a3a': (-0.0484900235-0j), '1a3a3b': (-0.0005168125-0j), '1a3b3b': (0.0346072373-0j), '1b1b1b': (2.57017e-05-0j), '1b1b2a': (0.0346072373-0j), '1b1b2b': (0.04304595-0j), '1b1b3a': (-0.0912094311-0j), '1b1b3b': (-0.0432277069-0j), '1b2a2a': (-0.0753999194-0j), '1b2a2b': (-0.2752951269-0j), '1b2a3a': (0.2814076211+0j), '1b2a3b': (0.1498439421-0j), '1b2b2b': (-0.0432277069-0j), '1b2b3a': (0.1498439421-0j), '1b2b3b': (0.0005055037-0j), '1b3a3a': (-0.0001712278-0j), '1b3a3b': (0.1487884197-0j), '1b3b3b': (0.04304595-0j), '2a2a2a': (0.013690768-0j), '2a2a2b': (0.0755303879-0j), '2a2a3a': (-0.0484900235-0j), '2a2a3b': (-0.0001712278-0j), '2a2b2b': (0.0348905793-0j), '2a2b3a': (-0.0005168125-0j), '2a2b3b': (0.1487884197-0j), '2a3a3a': (-0.0483280617-0j), '2a3a3b': (-0.2808469975-0j), '2a3b3b': (-0.0912094311-0j), '2b2b2b': (2.57017e-05-0j), '2b2b3a': (0.0346072373-0j), '2b2b3b': (0.04304595-0j), '2b3a3a': (-0.0753999194-0j), '2b3a3b': (-0.2752951269-0j), '2b3b3b': (-0.0432277069-0j), '3a3a3a': (0.013690768-0j), '3a3a3b': (0.0755303879-0j), '3a3b3b': (0.0348905793-0j), '3b3b3b': (2.57017e-05-0j)}
# )

initial_state = 4


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(initial_state)
print(basis_states)
print(len(basis_states))


eigenstates = False
max_number_of_states = 0
time_evolution = True
Fock_Space_Measure = False  #If false, prints occupation in the basis_label spatial occupation
t = np.linspace(0,0.5, 500) #in us


detuning_measurement = 100 * g

Measure_currents = False
analytic_currents = False

current_measurements = [['1a', '1b'], ['2a', '2b'], ['3a', '3b']]
# current_measurements = [['1a', '2a'], ['1b', '2b']]

skip_number = 5  #if 1, then do all of them. if 10, then every 10 times
Measure_correlations = False
analytic_correlations = True

analytic_correlation_sonly = True


current_correlations = [['1a', '1b'], ['3a', '3b']]
# current_correlations = [['1a', '2a'], ['1b', '2b']]

# correlation_values = {'1a1b': 1, '1b2a': -1, '1a2b': -1, '2a2b': 1}
time_beam = 2 * np.pi / (8 * g)

energy_map = {}
for key in basis_labels:
    if key not in energy_map.keys():
        energy_map[key] = e

energy_map.update((key, value * 2 * np.pi) for key, value in energy_map.items())#

H = Hamiltonian(basis_states, U, energy_map, coupling_dictionary_same,
                         coupling_dictionary_NN, coupling_periodic)
Evolution_H = copy.copy(H.H)

U = -150 *0
current_Hamiltonians = []
for c_m in current_measurements:
    H2 = Hamiltonian(basis_states, U, energy_map, coupling_dictionary_same,
                    coupling_dictionary_NN, coupling_periodic)
    Hamm = H_beamsplitter(c_m, detuning_measurement, H2, energy_map)
    current_Hamiltonians.append(copy.copy(Hamm))

H3 = Hamiltonian(basis_states, U, energy_map, coupling_dictionary_same,
                coupling_dictionary_NN, coupling_periodic)
Hamm = H_Beamsplitter_Correlation(current_correlations, detuning_measurement, H3, energy_map)
correlation_Hamiltonian = copy.copy(Hamm)
print(correlation_Hamiltonian / (2*np.pi))

if time_evolution:
    if type(initial_state) == int or type(initial_state) == float:
        eigval, eigvect = np.linalg.eig(H.H)
        sorted_eig = []
        for k, val in enumerate(eigval):
            sorted_eig.append((val, k))
        sorted_eig.sort()
        initial_state_vector = eigvect[:, sorted_eig[initial_state][1]]
        initial_state = normalize_eigenstate(initial_state_vector)


    sol = odeintw(function_vector, initial_state, t, args=(Evolution_H,), full_output = 1)[0]
    sol_copy = copy.copy(sol)
    if Fock_Space_Measure:
        all_exp_values = Expectation_values(sol, values_to_see, basis_states).real
        vmax = 1
    else:
        values_to_see = basis_labels
        all_exp_values = Expectation_values_basis_labels(sol, values_to_see, basis_states).real
        vmax = len(basis_states[0]) / len(basis_labels[0])

    # current = all_exp_values[1] - all_exp_values[0]

    plt.figure(figsize=([15, 8]))
    for i, exp in enumerate(all_exp_values):
        plt.plot(t[:None], exp[:None], label = str(values_to_see[i]))
    total_sum = np.array(all_exp_values[0])
    for i in range(len(all_exp_values[1:])):
        total_sum += np.array(all_exp_values[i + 1])
    # plt.plot(t[:None], total_sum[:None], label = 'Sum')
    plt.xlabel('Time (us)')
    plt.ylabel('Population')
    plt.legend(loc = 'upper right')
    plt.show()

    plt.figure(figsize = ([13, 15]))
    step = t[1] - t[0]
    plt.imshow(np.array(all_exp_values).T, vmin=0, vmax=vmax, aspect = 'auto',
               origin  = 'lower', interpolation = 'none', extent = (0.5, len(all_exp_values) + 0.5, t[0] - step/2, t[-1] + step/2),
               cmap = 'magma')
    plt.locator_params(axis='x', nbins=4)
    plt.ylabel('Time (us)')
    plt.xlabel('Qubit Number')
    plt.colorbar(label = 'Qubit Population')
    plt.show()

    if analytic_currents:
        Current_obj = Current_Operator(basis_states, basis_labels, ['1a'], ['1c'])
        updated_times = []
        Time_currents = [[] for i in range(len(current_measurements))]
        time_skips = skip_number

        for i, t_step in enumerate(t):
            if i % time_skips != 0:
                continue
            updated_times.append(t[i])
            initial_state = sol_copy[i, :]
            for j, c_meas in enumerate(current_measurements):
                op1 = Current_obj.operator_measurement(initial_state,
                                                       [c_meas[0]],
                                                       [c_meas[1]])
                op2 = Current_obj.operator_measurement(initial_state,
                                                       [c_meas[1]],
                                                       [c_meas[0]])
                Time_currents[j].append(1j * (op1 - op2))

        Time_currents = np.array(Time_currents)

        plt.figure(figsize=([15, 8]))
        for i, exp in enumerate(Time_currents):
            plt.plot(updated_times[:None], exp.real[:None], label=str(current_measurements[i]))
            # plt.plot(updated_times[:None], exp.imag[:None], label=str(current_measurements[i]) + 'imag')
        plt.legend(loc='upper right')
        plt.xlabel('Time (us)')
        plt.ylabel('Current')
        plt.title('Analytic')
        plt.show()

    if Measure_currents:
        t_beam = np.linspace(0, time_beam, 100)
        time_skips = skip_number
        if time_skips < 1:
            time_skips == 1
        occupation_numbers = [[] for i in range(len(current_measurements))]
        updated_times = []
        for i, t_step in enumerate(t):
            if i % time_skips != 0:
                continue
            updated_times.append(t[i])
            initial_state = sol_copy[i, :]
            for j, Ham_c in enumerate(current_Hamiltonians):
                sol_j = odeintw(function_vector, initial_state, t_beam, args=(Ham_c,), full_output=1)[0]
                values_to_see = current_measurements[j]
                all_exp_values = Expectation_values_basis_labels(sol_j, values_to_see, basis_states).real
                vmax = len(basis_states[0]) / len(basis_labels[0])
                final_values = all_exp_values[:, -1]
                occupation_numbers[j].append(final_values[1] - final_values[0])

        plt.figure(figsize=([15, 8]))
        for i, exp in enumerate(occupation_numbers):
            plt.plot(updated_times[:None], exp[:None], label=str(current_measurements[i]))

        plt.legend(loc='upper right')
        plt.xlabel('Time (us)')
        plt.ylabel('Current')
        plt.show()

    if analytic_correlations:
        Current_obj = Current_Operator(basis_states, basis_labels, ['1a', '1b'], ['1c', '2a'])

        updated_times = []
        Time_correlations = []
        time_skips = skip_number

        for i, t_step in enumerate(t):
            if i % time_skips != 0:
                continue
            updated_times.append(t[i])
            initial_state = sol_copy[i, :]
            op1 = Current_obj.operator_measurement(initial_state, [current_correlations[0][0], current_correlations[1][0]],
                                                   [current_correlations[0][1], current_correlations[1][1]])
            op2 = Current_obj.operator_measurement(initial_state, [current_correlations[0][0], current_correlations[1][1]],
                                                   [current_correlations[0][1], current_correlations[1][0]])
            op3 = Current_obj.operator_measurement(initial_state, [current_correlations[0][1], current_correlations[1][0]],
                                                   [current_correlations[0][0], current_correlations[1][1]])
            op4 = Current_obj.operator_measurement(initial_state, [current_correlations[0][1], current_correlations[1][1]],
                                                   [current_correlations[0][0], current_correlations[1][0]])

            Time_correlations.append(-1 * (op1 - op2 - op3 + op4))

        Time_correlations = np.array(Time_correlations)

        plt.figure(figsize=([15, 8]))
        plt.plot(updated_times[:None], Time_correlations.real[:None], label = 'Real')
        # plt.plot(updated_times[:None], Time_correlations.imag[:None], label = 'Imag')
        plt.xlabel('Time (us)')
        plt.ylabel('Current Correlation')
        plt.title('Analytic')
        plt.ylim(min(Time_correlations) - 0.1, max(Time_correlations) + 0.1)
        plt.show()

    if Measure_correlations:
        values_to_see = np.concatenate(current_correlations)
        values_to_see = [[current_correlations[0][0], current_correlations[1][0]],
                         [current_correlations[0][0], current_correlations[1][1]],
                         [current_correlations[0][1], current_correlations[1][0]],
                         [current_correlations[0][1], current_correlations[1][1]]]

        t_beam = np.linspace(0, time_beam, 100)
        time_skips = skip_number
        if time_skips < 1:
            time_skips == 1
        occupation_numbers = []
        occupation_numbers_individual = [[] for i in list(values_to_see)]
        updated_times = []
        for i, t_step in enumerate(t):
            if i % time_skips != 0:
                continue
            updated_times.append(t[i])
            initial_state = sol_copy[i, :]
            sol_j = odeintw(function_vector, initial_state, t_beam, args=(correlation_Hamiltonian,), full_output=1)[0]
            all_exp_values = Expectation_values_basis_labels(sol_j, values_to_see, basis_states).real
            final_values = all_exp_values[:, -1]
            # sum_final_values = (final_values[1] - final_values[0]) * (final_values[3] - final_values[2])
            sum_final_values = (final_values[0] + final_values[3] - final_values[1] - final_values[2])
            for k, val in enumerate(values_to_see):
                occupation_numbers_individual[k].append(final_values[k])
            occupation_numbers.append(sum_final_values)

        # plt.figure(figsize=([15, 8]))
        # for i, exp in enumerate(occupation_numbers_individual):
        #     plt.plot(updated_times[:None], exp[:None], label=str(values_to_see[i]))
        #
        # plt.legend(loc='upper right')
        # plt.xlabel('Time (us)')
        # plt.ylabel('Current')
        # plt.show()

        occupation_numbers = np.array(occupation_numbers)
        plt.figure(figsize=([15, 8]))
        plt.plot(updated_times[:None], occupation_numbers[:None])

        # plt.legend(loc='upper right')
        plt.xlabel('Time (us)')
        plt.ylabel('Current Correlation')
        plt.ylim(min(occupation_numbers) - 0.1, max(occupation_numbers) + 0.1)
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