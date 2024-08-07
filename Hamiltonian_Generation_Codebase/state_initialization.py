import numpy as np
import itertools



def matrix_element(H, state1, state2):
    final_state = H.dot(state1)
    overlap = state2.conj().dot(final_state)
    return (overlap)

def separated_list(basis, n=2):
    '''
    splits the basis into a list of individual states

    '1u1d' -> ['1u', '1d']

    n corresponds to the length that each state is. Typically just 2 unless you make a Hamiltoinan with >10
    lattice sites where youll need to use 3 values to distinguish each state
    '''
    return ([basis[i:i + n] for i in range(0, len(basis), n)])


def state_generation(basis_states, dictionary_excitation):
    state = np.zeros(len(basis_states), dtype=np.csingle)
    for i, basis in enumerate(basis_states):
        for key in dictionary_excitation.keys():
            permutations_state = permutations([key])

            for perm in permutations_state:
                if basis == perm:
                    state[i] = dictionary_excitation[key]
                    continue
    return(normalize_eigenstate(state))


def normalize_eigenstate(state):
    norm_factor = np.dot(state.conj(), state)
    if np.round(norm_factor, 7) == 0:
        print('State is zero. Check basis/dictionary')
        return (0)
    return (state / np.sqrt(norm_factor))


def tensor_product(dict1, dict2, factorials = True, n = 2):
    '''
    Need to add the sqrt stuff!
    :param dict1:
    :param dict2:
    :return:
    '''
    tensor_state = {}
    for key1, val1 in dict1.items():
        for key2, val2 in dict2.items():
            combined = key1 + key2
            separated = separated_list(combined, n = n)
            set_separated = set(separated)
            mult_factor = 1
            for val in set_separated:
                mult_factor *= np.sqrt(factorial(separated.count(val)))
            if not factorials:
                mult_factor = 1
            if key2 + key1 in tensor_state:
                tensor_state[key2 + key1] += mult_factor * val1 * val2 #Because Bosons commute except for raising and lowering
                continue                                 #on same site
            tensor_state[key1 + key2] = mult_factor * val1 * val2

    return(tensor_state)

def factorial(n):
    if n == 0:
        return(1)
    return(n * factorial(n - 1))


def permutations(state):
    '''
    Input state as a list. If single state '1ab1', input as ['1a1b'].
    You can also input as already separated list ['1a', '1b']
    '''
    if len(state) == 1:
        state = separated_list(state[0])
    perms = [''.join(p) for p in itertools.permutations(state)]
    return (perms)