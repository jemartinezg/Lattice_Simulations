from Hamiltonian_Generation_Codebase._BaseCodeHam import *

from Hamiltonian_Generation_Codebase.state_initialization import *
from Hamiltonian_Generation_Codebase.time_evolution import *
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 25})
plotsize = (10, 6)
legend_size = 12
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = 'tight'

basis_labels = ['1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b']
# basis_labels = ['1a', '2b', '3a', '4a']

basis_labels = ['1a', '2a', '3a', '4a']
basis_labels = ['1a', '2a', '3a']
basis_labels = ['1a', '2a', '3a', '4a', '5a', '6a']


g = -10 * 2 * np.pi    #In MHz and time in us
U= -250 * 2 * np.pi
e = 0

coupling_dictionary_NN = {'aa': g, 'ba': g, 'bb': g}     # Coupling to nearest unit cell
coupling_dictionary_same = {'ab' : -1j*g}  #Coupling within a unit cell
coupling_periodic = False

#one photon basis states
basis_states = RecursiveBasisState(1, basis_labels, state = '')
values_to_see = basis_states #plot all of the photon fock states
initial_state = state_generation(basis_states, {'1a': 1})
# initial_state = state_generation(basis_states, {'1a1a1a': -0.5})

print(initial_state)
print(basis_states)

eigenstates = False
time_evolution = True
Fock_Space_Measure = False  #If false, prints occupation in the basis_label spatial occupation
t = np.linspace(0,0.025/2, 500 * 3) #in us
t = np.linspace(0,0.4, 1000) #in us


detuning_measurement = 100 * g

Measure_currents = True
current_measurements = [['1a', '2a']]


skip_number = 5
Measure_correlations = False
current_correlations = [['1a', '2a'], ['3a', '4a']]


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

U = 5000 * 2 * np.pi * 0
current_Hamiltonians = []
for c_m in current_measurements:
    H2 = Hamiltonian(basis_states, U, energy_map, coupling_dictionary_same,
                    coupling_dictionary_NN, coupling_periodic)
    Hamm = H_beamsplitter(c_m, detuning_measurement, H2, energy_map)
    current_Hamiltonians.append(copy.copy(Hamm))
    print('Correlation Ham:')
    print(np.round(Hamm.real / (2 * np.pi)), 4)


H3 = Hamiltonian(basis_states, U, energy_map, coupling_dictionary_same,
                coupling_dictionary_NN, coupling_periodic)
Hamm = H_Beamsplitter_Correlation(current_correlations, detuning_measurement, H3, energy_map)
correlation_Hamiltonian = copy.copy(Hamm)
print(correlation_Hamiltonian / (2*np.pi))

if time_evolution:
    print(Evolution_H)
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
        print(max(exp[:None]))
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

    if Measure_currents:
        t_beam = np.linspace(0, time_beam, 100)
        time_skips = time_beam // step // skip_number
        if time_skips < 1:
            time_skips == 1
        print(time_skips)
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
            print(max(exp[:None]))
            plt.plot(updated_times[:None], exp[:None], label=str(current_measurements[i]))

        plt.legend(loc='upper right')
        plt.xlabel('Time (us)')
        plt.ylabel('Current')
        plt.show()
        print(len(updated_times))
    if Measure_correlations:
        values_to_see = np.concatenate(current_correlations)
        values_to_see = [[current_correlations[0][0], current_correlations[1][0]],
                         [current_correlations[0][0], current_correlations[1][1]],
                         [current_correlations[0][1], current_correlations[1][0]],
                         [current_correlations[0][1], current_correlations[1][1]]]

        t_beam = np.linspace(0, time_beam, 100)
        time_skips = time_beam // step // skip_number
        if time_skips < 1:
            time_skips == 1
        print(time_skips)
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
        print('All ', final_values)
        print('01 ', final_values[1] - final_values[0])
        print('23 ', final_values[3] - final_values[2])



        plt.figure(figsize=([15, 8]))
        for i, exp in enumerate(occupation_numbers_individual):
            print(max(exp[:None]))
            plt.plot(updated_times[:None], exp[:None], label=str(values_to_see[i]))

        plt.legend(loc='upper right')
        plt.xlabel('Time (us)')
        plt.ylabel('Current')
        plt.show()
        print(len(updated_times))

        plt.figure(figsize=([15, 8]))
        plt.plot(updated_times[:None], occupation_numbers[:None])

        plt.legend(loc='upper right')
        plt.xlabel('Time (us)')
        plt.ylabel('Current Correlation')
        plt.ylim(min(occupation_numbers) - 0.1, max(occupation_numbers) + 0.1)
        plt.show()
        print(len(updated_times))
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
        print(np.round(np.real(sorted_eig[i][0]) / (2 * np.pi), 5))
        dictionary_eigenstate_initial = {}
        for j in range(len(basis_states)):
            print(basis_states[j] + ': ', np.round(eigvect[j, sorted_eig[i][1]] , 10))
            dictionary_eigenstate_initial[basis_states[j]] = np.round(eigvect[j, sorted_eig[i][1]] , 10)
        print(dictionary_eigenstate_initial)