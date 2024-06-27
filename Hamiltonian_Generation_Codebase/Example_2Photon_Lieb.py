from _BaseCodeHam import *
from state_initialization import *
from time_evolution import *
import itertools
import matplotlib.pyplot as plt


# rcParams['figure.figsize'] = 18, 8
plt.rcParams.update({'font.size': 22})
plotsize = (10, 6)
legend_size = 12
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = 'tight'

basis_states = ['1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b']
# basis_states = ['1a', '1b', '1c', '2a', '2b']

basis_states = RecursiveBasisState(2, basis_states, state = '')
g = 0.05 * 2 * np.pi

energy = 0
energy_map = {'a': energy, 'b': energy, 'c': energy}
coupling_dictionary_NN = {'cb': g}

coupling_dictionary_same = {'ab' : g, 'bc': g}

# H_Lieb = generate_hamiltonian(basis_states, 0, energy_map, coupling_dictionary_same,
#                          coupling_dictionary_NN, 2, False).astype('csingle')

U = 0.01 * 2 * np.pi

H_Lieb = Hamiltonian(basis_states, U, energy_map, coupling_dictionary_same, coupling_dictionary_NN, False)


# initial_state_lieb = state_generation(basis_states, {'1b': 1, '2a': -1, '2b': 1}).astype('csingle')
tensored = tensor_product({'1a': 1 / np.sqrt(3), '1c': -1 / np.sqrt(3), '2a': 1 / np.sqrt(3)},
                          {'1a': 1 / np.sqrt(3), '1c': -1 / np.sqrt(3), '2a': 1 / np.sqrt(3)})
# tensored = tensor_product({'1a': 1 / np.sqrt(3), '1c': -1 / np.sqrt(3), '2a': 1 / np.sqrt(3)},
#                           {'2a': 1 / np.sqrt(3), '2c': -1 / np.sqrt(3), '3a': 1 / np.sqrt(3)})
# initial_state_lieb = state_generation(basis_states, {'1a1a': 1, '2a2a': 1, '1c1c': 1, '1a2a': np.sqrt(2),
#                                                     '1a1c': -np.sqrt(2), '1c2a': -np.sqrt(2)}).astype('csingle')

initial_state_lieb_tens = state_generation(basis_states, tensored).astype('csingle')

print(tensored)

print(initial_state_lieb_tens)

final_state = np.round(H_Lieb.H.dot(initial_state_lieb_tens).real, 5)
# for i in range(len(final_state)):
#     print(basis_states[i], final_state[i])
# for i in range(len(final_state)):
#     print(H_Lieb.H[i, i])
# print('unsqared', np.round(H_Lieb.H.dot(initial_state_lieb_tens).real, 5))


t = np.linspace(0, 40, 5000)
# t = np.linspace(0, 300, 5000)


sol = States_time_evolution(H_Lieb.H, t, initial_state_lieb_tens)
sol_total = (sol.conj() * sol).real
print(len(sol_total))
print(len(sol_total[0]))
print(np.round(sol_total[-1, :], 5))
print(sum(sol_total[-1, :]**2))
print(sum(sol_total[-5, :]**2))
print(sum(sol_total[-10, :]**2))
print(sum(sol_total[-15, :]**2))
print(sum(sol_total[0, :]**2))



# print(sum(sol_total[-1, :]))

# print(np.sum(np.round(sol_total[-1, :].real), 5))


# all_exp_values_low = expectation_values_vector(sol, list(range(len(basis_states))) + [0]).real


#
# # plot_data(all_exp_values, t, '1 MHz Disorder', ylim = [-0.04, 1.04])
# plt.show()
#
# # plot_data(all_exp_values, t, '1000 MHz Disorder', ylim = [-0.04, 1.04])
# plt.figure(figsize = ([10, 12]))
# ax = fig.add_subplot(111)
# axp = ax.imshow(np.array(sol[:-1]), vmin=0, vmax = 1, aspect = 'auto',
#            origin  = 'lower', cmap = 'magma', interpolation = 'none')
# plt.ylabel('Time (ns)')
# plt.xlabel('Eigenstate Label')
#
# cb = plt.colorbar(axp ,ax = [ax], location = 'top', label= r'$\leftangle \hat{n}_i \rangle$')


fig = plt.figure(figsize = ([10, 16]))
ax = fig.add_subplot(111)

axp = ax.imshow(np.array(sol_total[:-1]), vmin=0, vmax=0.3, aspect = 'auto',  #0.002,
           origin  = 'lower', cmap = 'magma', interpolation = 'none', extent = (0.5, len(basis_states) + 5,
                                                                                0, t[-1]))#, extent = (0.5, 7.5))
plt.xlabel('Qubit Index Label')
plt.ylabel('Time (1/J)')
plt.locator_params(axis='x', nbins=7)
cb = plt.colorbar(axp ,ax = [ax], location = 'top', label= r'$\leftangle \hat{n}_i \rangle$')

plt.show()
#
# print('squared', np.linalg.eig(H_Lieb.H**2)[0])
