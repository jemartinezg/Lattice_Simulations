import numpy as np


def Eigenstates(Hamiltonian, basis_states, Eigenstate_Dictionary = True,
                round_Hamiltonian = 3, round_eigenstates = 5):
    print('Hamiltonian real:')
    print(np.round(Hamiltonian.real / (2 * np.pi), round_Hamiltonian))
    print('Hamiltonian imaginary:')
    print(np.round(Hamiltonian.imag / (2 * np.pi), round_Hamiltonian))

    eigval, eigvect = np.linalg.eig(Hamiltonian)
    sorted_eig = []
    for k, val in enumerate(eigval):
        sorted_eig.append((val, k))
    sorted_eig.sort()

    print(np.round(np.array(np.real(sorted_eig))[:, 0] / (2 * np.pi), round_eigenstates))
    for i in range(len(sorted_eig)):
        print(np.round(np.real(sorted_eig[i][0]) / (2 * np.pi), round_eigenstates))
        dictionary_eigenstate_initial = {}
        for j in range(len(basis_states)):
            print(basis_states[j] + ': ', np.round(eigvect[j, sorted_eig[i][1]], round_eigenstates))
            dictionary_eigenstate_initial[basis_states[j]] = np.round(eigvect[j, sorted_eig[i][1]], round_eigenstates)
        if Eigenstate_Dictionary:
            print(dictionary_eigenstate_initial)
