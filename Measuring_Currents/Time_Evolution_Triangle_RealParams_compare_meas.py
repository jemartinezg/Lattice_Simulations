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

g = -5 * 2 * np.pi    #In MHz and time in us
U= -150 * 2 * np.pi
e = 0

coupling_dictionary_NN = {'aa': -2 * g, 'ba': g, 'bb': -2 * g}     # Coupling to nearest unit cell
coupling_dictionary_same = {'ab' : g}  #Coupling within a unit cell
coupling_periodic = False #{'aa': -3 * g, 'ba': g, 'bb': -3 * g}


#one photon basis states
basis_states = RecursiveBasisState(4, basis_labels, state = '')
values_to_see = basis_states #plot all of the photon fock states

#0 flux
initial_state = state_generation(basis_states, {'1a1a1a1a': (1.458e-06-0j), '1a1a1a1b': (-5.19908e-05-0j), '1a1a1a2a': (4.44486e-05-0j), '1a1a1a2b': (-0.0001847235-0j), '1a1a1a3a': (0.0002140174-0j), '1a1a1a3b': (-0.0003520384-0j), '1a1a1a4a': (0.0002893313-0j), '1a1a1a4b': (-0.000294243-0j), '1a1a1b1b': (0.0003154377-0j), '1a1a1b2a': (-0.0012064019-0j), '1a1a1b2b': (0.0036389463-0j), '1a1a1b3a': (-0.0051120257-0j), '1a1a1b3b': (0.0075967508-0j), '1a1a1b4a': (-0.0065260382-0j), '1a1a1b4b': (0.0064109778-0j), '1a1a2a2a': (0.000373473-0j), '1a1a2a2b': (-0.0033789972-0j), '1a1a2a3a': (0.0034758912-0j), '1a1a2a3b': (-0.006273564-0j), '1a1a2a4a': (0.0050514148-0j), '1a1a2a4b': (-0.0052787136-0j), '1a1a2b2b': (0.0017460195-0j), '1a1a2b3a': (-0.0108733745-0j), '1a1a2b3b': (0.0128152965-0j), '1a1a2b4a': (-0.0135379485-0j), '1a1a2b4b': (0.0124428928-0j), '1a1a3a3a': (0.002049717-0j), '1a1a3a3b': (-0.0158159942-0j), '1a1a3a4a': (0.0107346549-0j), '1a1a3a4b': (-0.013445391-0j), '1a1a3b3b': (0.0030420804-0j), '1a1a3b4a': (-0.0157116474-0j), '1a1a3b4b': (0.0114818052-0j), '1a1a4a4a': (0.0018507834-0j), '1a1a4a4b': (-0.0112217829-0j), '1a1a4b4b': (0.0013198311-0j), '1a1b1b1b': (-0.0001983924-0j), '1a1b1b2a': (0.003711274-0j), '1a1b1b2b': (-0.0036697432-0j), '1a1b1b3a': (0.0093295715-0j), '1a1b1b3b': (-0.0107324306-0j), '1a1b1b4a': (0.0108340238-0j), '1a1b1b4b': (-0.0097624183-0j), '1a1b2a2a': (-0.0057285273-0j), '1a1b2a2b': (0.0420869789-0j), '1a1b2a3a': (-0.0531013993-0j), '1a1b2a3b': (0.0858848855-0j), '1a1b2a4a': (-0.0720501025-0j), '1a1b2a4b': (0.0726271861-0j), '1a1b2b2b': (-0.0116424724-0j), '1a1b2b3a': (0.0890599082-0j), '1a1b2b3b': (-0.0882584102-0j), '1a1b2b4a': (0.1003962999-0j), '1a1b2b4b': (-0.086744435-0j), '1a1b3a3a': (-0.0208938903-0j), '1a1b3a3b': (0.1499082332-0j), '1a1b3a4a': (-0.1062533661-0j), '1a1b3a4b': (0.1300642891-0j), '1a1b3b3b': (-0.0247692888-0j), '1a1b3b4a': (0.1370397451-0j), '1a1b3b4b': (-0.0948934027-0j), '1a1b4a4a': (-0.017240557-0j), '1a1b4a4b': (0.1009578837-0j), '1a1b4b4b': (-0.0112217829-0j), '1a2a2a2a': (0.0005563402-0j), '1a2a2a2b': (-0.0097991945-0j), '1a2a2a3a': (0.0079836311-0j), '1a2a2a3b': (-0.0177072161-0j), '1a2a2a4a': (0.0138976494-0j), '1a2a2a4b': (-0.0153211446-0j), '1a2a2b2b': (0.0158225797-0j), '1a2a2b3a': (-0.0873573483-0j), '1a2a2b3b': (0.1164785774-0j), '1a2a2b4a': (-0.118936825-0j), '1a2a2b4b': (0.1138183104-0j), '1a2a3a3a': (0.0150882406-0j), '1a2a3a3b': (-0.1233334408-0j), '1a2a3a4a': (0.0828978992-0j), '1a2a3a4b': (-0.1055064382-0j), '1a2a3b3b': (0.0263483901-0j), '1a2a3b4a': (-0.1327931412-0j), '1a2a3b4b': (0.1007280587-0j), '1a2a4a4a': (0.01511612-0j), '1a2a4a4b': (-0.0948934027-0j), '1a2a4b4b': (0.0114818052-0j), '1a2b2b2b': (-0.0018103325-0j), '1a2b2b3a': (0.0245090359-0j), '1a2b2b3b': (-0.0208607804-0j), '1a2b2b4a': (0.0289880261-0j), '1a2b2b4b': (-0.0249440137-0j), '1a2b3a3a': (-0.0252476479-0j), '1a2b3a3b': (0.1623861072-0j), '1a2b3a4a': (-0.1376826946-0j), '1a2b3a4b': (0.1645548395-0j), '1a2b3b3b': (-0.0232991474-0j), '1a2b3b4a': (0.1526872586-0j), '1a2b3b4b': (-0.1055064382-0j), '1a2b4a4a': (-0.0217618427-0j), '1a2b4a4b': (0.1300642891-0j), '1a2b4b4b': (-0.013445391-0j), '1a3a3a3a': (0.0020065093-0j), '1a3a3a3b': (-0.0268573956-0j), '1a3a3a4a': (0.017208235-0j), '1a3a3a4b': (-0.0249440137-0j), '1a3a3b3b': (0.0279357157-0j), '1a3a3b4a': (-0.1181199936-0j), '1a3a3b4b': (0.1138183104-0j), '1a3a4a4a': (0.0106238093-0j), '1a3a4a4b': (-0.086744435-0j), '1a3a4b4b': (0.0124428928-0j), '1a3b3b3b': (-0.0022093701-0j), '1a3b3b4a': (0.0226220425-0j), '1a3b3b4b': (-0.0153211446-0j), '1a3b4a4a': (-0.0138975438-0j), '1a3b4a4b': (0.0726271861-0j), '1a3b4b4b': (-0.0052787136-0j), '1a4a4a4a': (0.0007771789-0j), '1a4a4a4b': (-0.0097624183-0j), '1a4a4b4b': (0.0064109778-0j), '1a4b4b4b': (-0.000294243-0j), '1b1b1b1b': (9.6578e-06-0j), '1b1b1b2a': (-0.0002767774-0j), '1b1b1b2b': (0.0002290325-0j), '1b1b1b3a': (-0.0007128261-0j), '1b1b1b3b': (0.0008035279-0j), '1b1b1b4a': (-0.0008622683-0j), '1b1b1b4b': (0.0007771789-0j), '1b1b2a2a': (0.0012713747-0j), '1b1b2a2b': (-0.004897931-0j), '1b1b2a3a': (0.0108250408-0j), '1b1b2a3b': (-0.0147901316-0j), '1b1b2a4a': (0.0145615018-0j), '1b1b2a4b': (-0.0138975438-0j), '1b1b2b2b': (0.0012604609-0j), '1b1b2b3a': (-0.010215346-0j), '1b1b2b3b': (0.0101551396-0j), '1b1b2b4a': (-0.0121373955-0j), '1b1b2b4b': (0.0106238093-0j), '1b1b3a3a': (0.0034427105-0j), '1b1b3a3b': (-0.0230060255-0j), '1b1b3a4a': (0.0180936816-0j), '1b1b3a4b': (-0.0217618427-0j), '1b1b3b3b': (0.0036385162-0j), '1b1b3b4a': (-0.0216799416-0j), '1b1b3b4b': (0.01511612-0j), '1b1b4a4a': (0.0028997675-0j), '1b1b4a4b': (-0.017240557-0j), '1b1b4b4b': (0.0018507834-0j), '1b2a2a2a': (-0.0008563962-0j), '1b2a2a2b': (0.0120994049-0j), '1b2a2a3a': (-0.0119546453-0j), '1b2a2a3b': (0.0253606916-0j), '1b2a2a4a': (-0.0207625195-0j), '1b2a2a4b': (0.0226220425-0j), '1b2a2b2b': (-0.0137636082-0j), '1b2a2b3a': (0.0979456543-0j), '1b2a2b3b': (-0.1151309792-0j), '1b2a2b4a': (0.1289366683-0j), '1b2a2b4b': (-0.1181199936-0j), '1b2a3a3a': (-0.0209222055-0j), '1b2a3a3b': (0.1707977418-0j), '1b2a3a4a': (-0.1202614846-0j), '1b2a3a4b': (0.1526872586-0j), '1b2a3b3b': (-0.0326444608-0j), '1b2a3b4a': (0.1815120203+0j), '1b2a3b4b': (-0.1327931412-0j), '1b2a4a4a': (-0.0216799416-0j), '1b2a4a4b': (0.1370397451-0j), '1b2a4b4b': (-0.0157116474-0j), '1b2b2b2b': (0.0010600794-0j), '1b2b2b3a': (-0.0168167759-0j), '1b2b2b3b': (0.0134098492-0j), '1b2b2b4a': (-0.0201317128-0j), '1b2b2b4b': (0.017208235-0j), '1b2b3a3a': (0.0227333567-0j), '1b2b3a3b': (-0.129590528-0j), '1b2b3a4a': (0.1190576024-0j), '1b2b3a4b': (-0.1376826946-0j), '1b2b3b3b': (0.0179213237-0j), '1b2b3b4a': (-0.1202614846-0j), '1b2b3b4b': (0.0828978992-0j), '1b2b4a4a': (0.0180936816-0j), '1b2b4a4b': (-0.1062533661-0j), '1b2b4b4b': (0.0107346549-0j), '1b3a3a3a': (-0.0023322398-0j), '1b3a3a3b': (0.030162838-0j), '1b3a3a4a': (-0.0201317128-0j), '1b3a3a4b': (0.0289880261-0j), '1b3a3b3b': (-0.0268252717-0j), '1b3a3b4a': (0.1289366683-0j), '1b3a3b4b': (-0.118936825-0j), '1b3a4a4a': (-0.0121373955-0j), '1b3a4a4b': (0.1003962999-0j), '1b3a4b4b': (-0.0135379485-0j), '1b3b3b3b': (0.0019289961-0j), '1b3b3b4a': (-0.0207625195-0j), '1b3b3b4b': (0.0138976494-0j), '1b3b4a4a': (0.0145615018-0j), '1b3b4a4b': (-0.0720501025-0j), '1b3b4b4b': (0.0050514148-0j), '1b4a4a4a': (-0.0008622683-0j), '1b4a4a4b': (0.0108340238-0j), '1b4a4b4b': (-0.0065260382-0j), '1b4b4b4b': (0.0002893313-0j), '2a2a2a2a': (5.2694e-05-0j), '2a2a2a2b': (-0.0012212381-0j), '2a2a2a3a': (0.0009507558-0j), '2a2a2a3b': (-0.0024287244-0j), '2a2a2a4a': (0.0019289961-0j), '2a2a2a4b': (-0.0022093701-0j), '2a2a2b2b': (0.0036613652-0j), '2a2a2b3a': (-0.0157639869-0j), '2a2a2b3b': (0.0273113392-0j), '2a2a2b4a': (-0.0268252717-0j), '2a2a2b4b': (0.0279357157-0j), '2a2a3a3a': (0.0030190318-0j), '2a2a3a3b': (-0.0255416877-0j), '2a2a3a4a': (0.0179213237-0j), '2a2a3a4b': (-0.0232991474-0j), '2a2a3b3b': (0.0063898475-0j), '2a2a3b4a': (-0.0326444608-0j), '2a2a3b4b': (0.0263483901-0j), '2a2a4a4a': (0.0036385162-0j), '2a2a4a4b': (-0.0247692888-0j), '2a2a4b4b': (0.0030420804-0j), '2a2b2b2b': (-0.0017119843-0j), '2a2b2b3a': (0.0222968962-0j), '2a2b2b3b': (-0.0210844995-0j), '2a2b2b4a': (0.030162838-0j), '2a2b2b4b': (-0.0268573956-0j), '2a2b3a3a': (-0.0201194497-0j), '2a2b3a3b': (0.1554540263-0j), '2a2b3a4a': (-0.129590528-0j), '2a2b3a4b': (0.1623861072-0j), '2a2b3b3b': (-0.0255416877-0j), '2a2b3b4a': (0.1707977418-0j), '2a2b3b4b': (-0.1233334408-0j), '2a2b4a4a': (-0.0230060255-0j), '2a2b4a4b': (0.1499082332-0j), '2a2b4b4b': (-0.0158159942-0j), '2a3a3a3a': (0.0013665758-0j), '2a3a3a3b': (-0.0210844995-0j), '2a3a3a4a': (0.0134098492-0j), '2a3a3a4b': (-0.0208607804-0j), '2a3a3b3b': (0.0273113392-0j), '2a3a3b4a': (-0.1151309792-0j), '2a3a3b4b': (0.1164785774-0j), '2a3a4a4a': (0.0101551396-0j), '2a3a4a4b': (-0.0882584102-0j), '2a3a4b4b': (0.0128152965-0j), '2a3b3b3b': (-0.0024287244-0j), '2a3b3b4a': (0.0253606916-0j), '2a3b3b4b': (-0.0177072161-0j), '2a3b4a4a': (-0.0147901316-0j), '2a3b4a4b': (0.0858848855-0j), '2a3b4b4b': (-0.006273564-0j), '2a4a4a4a': (0.0008035279-0j), '2a4a4a4b': (-0.0107324306-0j), '2a4a4b4b': (0.0075967508-0j), '2a4b4b4b': (-0.0003520384-0j), '2b2b2b2b': (8.66752e-05-0j), '2b2b2b3a': (-0.0018101365-0j), '2b2b2b3b': (0.0013665758-0j), '2b2b2b4a': (-0.0023322398-0j), '2b2b2b4b': (0.0020065093-0j), '2b2b3a3a': (0.0043748782-0j), '2b2b3a3b': (-0.0201194497-0j), '2b2b3a4a': (0.0227333567-0j), '2b2b3a4b': (-0.0252476479-0j), '2b2b3b3b': (0.0030190318-0j), '2b2b3b4a': (-0.0209222055-0j), '2b2b3b4b': (0.0150882406-0j), '2b2b4a4a': (0.0034427105-0j), '2b2b4a4b': (-0.0208938903-0j), '2b2b4b4b': (0.002049717-0j), '2b3a3a3a': (-0.0018101365-0j), '2b3a3a3b': (0.0222968962-0j), '2b3a3a4a': (-0.0168167759-0j), '2b3a3a4b': (0.0245090359-0j), '2b3a3b3b': (-0.0157639869-0j), '2b3a3b4a': (0.0979456543-0j), '2b3a3b4b': (-0.0873573483-0j), '2b3a4a4a': (-0.010215346-0j), '2b3a4a4b': (0.0890599082-0j), '2b3a4b4b': (-0.0108733745-0j), '2b3b3b3b': (0.0009507558-0j), '2b3b3b4a': (-0.0119546453-0j), '2b3b3b4b': (0.0079836311-0j), '2b3b4a4a': (0.0108250408-0j), '2b3b4a4b': (-0.0531013993-0j), '2b3b4b4b': (0.0034758912-0j), '2b4a4a4a': (-0.0007128261-0j), '2b4a4a4b': (0.0093295715-0j), '2b4a4b4b': (-0.0051120257-0j), '2b4b4b4b': (0.0002140174-0j), '3a3a3a3a': (8.66752e-05-0j), '3a3a3a3b': (-0.0017119843-0j), '3a3a3a4a': (0.0010600794-0j), '3a3a3a4b': (-0.0018103325-0j), '3a3a3b3b': (0.0036613652-0j), '3a3a3b4a': (-0.0137636082-0j), '3a3a3b4b': (0.0158225797-0j), '3a3a4a4a': (0.0012604609-0j), '3a3a4a4b': (-0.0116424724-0j), '3a3a4b4b': (0.0017460195-0j), '3a3b3b3b': (-0.0012212381-0j), '3a3b3b4a': (0.0120994049-0j), '3a3b3b4b': (-0.0097991945-0j), '3a3b4a4a': (-0.004897931-0j), '3a3b4a4b': (0.0420869789-0j), '3a3b4b4b': (-0.0033789972-0j), '3a4a4a4a': (0.0002290325-0j), '3a4a4a4b': (-0.0036697432-0j), '3a4a4b4b': (0.0036389463-0j), '3a4b4b4b': (-0.0001847235-0j), '3b3b3b3b': (5.2694e-05-0j), '3b3b3b4a': (-0.0008563962-0j), '3b3b3b4b': (0.0005563402-0j), '3b3b4a4a': (0.0012713747-0j), '3b3b4a4b': (-0.0057285273-0j), '3b3b4b4b': (0.000373473-0j), '3b4a4a4a': (-0.0002767774-0j), '3b4a4a4b': (0.003711274-0j), '3b4a4b4b': (-0.0012064019-0j), '3b4b4b4b': (4.44486e-05-0j), '4a4a4a4a': (9.6578e-06-0j), '4a4a4a4b': (-0.0001983924-0j), '4a4a4b4b': (0.0003154377-0j), '4a4b4b4b': (-5.19908e-05-0j), '4b4b4b4b': (1.458e-06-0j)}
)

#Pi Flux
# initial_state = state_generation(basis_states, {'1a1a1a1a': (1.2768e-06+0j), '1a1a1a1b': (-4.89426e-05+0j), '1a1a1a2a': (-3.61046e-05+0j), '1a1a1a2b': (0.0001614154+0j), '1a1a1a3a': (0.0001633947+0j), '1a1a1a3b': (-0.0003296467+0j), '1a1a1a4a': (-0.0001616404+0j), '1a1a1a4b': (0.0003293954+0j), '1a1a1b1b': (0.0002475508+0j), '1a1a1b2a': (0.0011512026+0j), '1a1a1b2b': (-0.0033858108+0j), '1a1a1b3a': (-0.0046156969+0j), '1a1a1b3b': (0.0077588842+0j), '1a1a1b4a': (0.0044469317+0j), '1a1a1b4b': (-0.0079917889+0j), '1a1a2a2a': (0.0002376493+0j), '1a1a2a2b': (-0.002774715+0j), '1a1a2a3a': (-0.0022358831+0j), '1a1a2a3b': (0.0053319143+0j), '1a1a2a4a': (0.0023382036+0j), '1a1a2a4b': (-0.005255117+0j), '1a1a2b2b': (0.0006190402+0j), '1a1a2b3a': (0.0097541484+0j), '1a1a2b3b': (-0.0073650275+0j), '1a1a2b4a': (-0.009694694+0j), '1a1a2b4b': (0.0103625091+0j), '1a1a3a3a': (0.0005728005+0j), '1a1a3a3b': (-0.013651745+0j), '1a1a3a4a': (-0.0026638056+0j), '1a1a3a4b': (0.0119654087+0j), '1a1a3b3b': (0.0012703484+0j), '1a1a3b4a': (0.0145624089+0j), '1a1a3b4b': (-0.0087729502+0j), '1a1a4a4a': (3.82033e-05+0j), '1a1a4a4b': (-0.0097937082+0j), '1a1a4b4b': (0.0011081554+0j), '1a1b1b1b': (0.000126538+0j), '1a1b1b2a': (-0.0033643845+0j), '1a1b1b2b': (-0.0021590464+0j), '1a1b1b3a': (0.0065055375+0j), '1a1b1b3b': (0.0048642307+0j), '1a1b1b4a': (-0.0057495274+0j), '1a1b1b4b': (-0.0029292188+0j), '1a1b2a2a': (-0.0053386462+0j), '1a1b2a2b': (0.0449011049+0j), '1a1b2a3a': (0.0486067987+0j), '1a1b2a3b': (-0.1009362666+0j), '1a1b2a4a': (-0.0490054999+0j), '1a1b2a4b': (0.101743456+0j), '1a1b2b2b': (0.0041573737+0j), '1a1b2b3a': (-0.0994093266+0j), '1a1b2b3b': (-0.021808607+0j), '1a1b2b4a': (0.0888805396+0j), '1a1b2b4b': (-0.0014129886+0j), '1a1b3a3a': (-0.0120362431+0j), '1a1b3a3b': (0.2028856181+0j), '1a1b3a4a': (0.0486781634+0j), '1a1b3a4b': (-0.191027343+0j), '1a1b3b3b': (-0.0027632519+0j), '1a1b3b4a': (-0.1832673817+0j), '1a1b3b4b': (0.048109079+0j), '1a1b4a4a': (-0.0025840636+0j), '1a1b4a4b': (0.146813561+0j), '1a1b4b4b': (-0.0097937082+0j), '1a2a2a2a': (-4.82755e-05+0j), '1a2a2a2b': (0.0045643498+0j), '1a2a2a3a': (0.0012436753+0j), '1a2a2a3b': (-0.0063385161+0j), '1a2a2a4a': (-0.0020453363+0j), '1a2a2a4b': (0.0061804342+0j), '1a2a2b2b': (-0.0100900773+0j), '1a2a2b3a': (-0.0629515227+0j), '1a2a2b3b': (0.1027083633+0j), '1a2a2b4a': (0.0706289499+0j), '1a2a2b4b': (-0.1269213717+0j), '1a2a3a3a': (-0.0012651105+0j), '1a2a3a3b': (0.0637632573+0j), '1a2a3a4a': (0.0097373296+0j), '1a2a3a4b': (-0.0521017533+0j), '1a2a3b3b': (-0.0142768457+0j), '1a2a3b4a': (-0.0863590209+0j), '1a2a3b4b': (0.0837118564+0j), '1a2a4a4a': (0.0006289621+0j), '1a2a4a4b': (0.048109079+0j), '1a2a4b4b': (-0.0087729502+0j), '1a2b2b2b': (-0.0006459637+0j), '1a2b2b3a': (0.0269579633+0j), '1a2b2b3b': (0.0055642647+0j), '1a2b2b4a': (-0.0233491222+0j), '1a2b2b4b': (-3.10186e-05+0j), '1a2b3a3a': (0.018302975+0j), '1a2b3a3b': (-0.2185418905+0j), '1a2b3a4a': (-0.0795296764+0j), '1a2b3a4b': (0.2463280799+0j), '1a2b3b3b': (0.0037912005+0j), '1a2b3b4a': (0.2020697907+0j), '1a2b3b4b': (-0.0521017533+0j), '1a2b4a4a': (0.0060431295+0j), '1a2b4a4b': (-0.191027343+0j), '1a2b4b4b': (0.0119654087+0j), '1a3a3a3a': (-0.000260486+0j), '1a3a3a3b': (-0.0024457578+0j), '1a3a3a4a': (0.000842018+0j), '1a3a3a4b': (-3.10186e-05+0j), '1a3a3b3b': (0.0265143256+0j), '1a3a3b4a': (0.0423088881+0j), '1a3a3b4b': (-0.1269213717+0j), '1a3a4a4a': (-0.0016657109+0j), '1a3a4a4b': (-0.0014129886+0j), '1a3a4b4b': (0.0103625091+0j), '1a3b3b3b': (-0.0003636562+0j), '1a3b3b4a': (-0.0258877612+0j), '1a3b3b4b': (0.0061804342+0j), '1a3b4a4a': (-0.0058657677+0j), '1a3b4a4b': (0.101743456+0j), '1a3b4b4b': (-0.005255117+0j), '1a4a4a4a': (0.000215197+0j), '1a4a4a4b': (-0.0029292188+0j), '1a4a4b4b': (-0.0079917889+0j), '1a4b4b4b': (0.0003293954+0j), '1b1b1b1b': (-2.8406e-06+0j), '1b1b1b2a': (3.98361e-05+0j), '1b1b1b2b': (5.15839e-05+0j), '1b1b1b3a': (6.60948e-05+0j), '1b1b1b3b': (-0.0001719612+0j), '1b1b1b4a': (-0.0001250735+0j), '1b1b1b4b': (0.000215197+0j), '1b1b2a2a': (0.0011926367+0j), '1b1b2a2b': (-0.0004330469+0j), '1b1b2a3a': (-0.0108625058+0j), '1b1b2a3b': (0.002563281+0j), '1b1b2a4a': (0.0115051172+0j), '1b1b2a4b': (-0.0058657677+0j), '1b1b2b2b': (-8.33017e-05+0j), '1b1b2b3a': (0.0003586332+0j), '1b1b2b3b': (0.0011024606+0j), '1b1b2b4a': (0.0006742159+0j), '1b1b2b4b': (-0.0016657109+0j), '1b1b3a3a': (0.0015377199+0j), '1b1b3a3b': (-0.0032247109+0j), '1b1b3a4a': (-0.0066878503+0j), '1b1b3a4b': (0.0060431295+0j), '1b1b3b3b': (-8.47515e-05+0j), '1b1b3b4a': (0.0015355894+0j), '1b1b3b4b': (0.0006289621+0j), '1b1b4a4a': (0.0005846546+0j), '1b1b4a4b': (-0.0025840636+0j), '1b1b4b4b': (3.82033e-05+0j), '1b2a2a2a': (0.0007077775+0j), '1b2a2a2b': (-0.0129585866+0j), '1b2a2a3a': (-0.009297248+0j), '1b2a2a3b': (0.0261462921+0j), '1b2a2a4a': (0.0115510295+0j), '1b2a2a4b': (-0.0258877612+0j), '1b2a2b2b': (-0.0032037463+0j), '1b2a2b3a': (0.1186621373+0j), '1b2a2b3b': (-0.001288856+0j), '1b2a2b4a': (-0.1279994759+0j), '1b2a2b4b': (0.0423088881+0j), '1b2a3a3a': (0.0094930748+0j), '1b2a3a3b': (-0.2219390007+0j), '1b2a3a4a': (-0.0458387405+0j), '1b2a3a4b': (0.2020697907+0j), '1b2a3b3b': (0.0064045455+0j), '1b2a3b4a': (0.2427954692+0j), '1b2a3b4b': (-0.0863590209+0j), '1b2a4a4a': (0.0015355894+0j), '1b2a4a4b': (-0.1832673817+0j), '1b2a4b4b': (0.0145624089+0j), '1b2b2b2b': (0.0002446492+0j), '1b2b2b3a': (-0.0070132571+0j), '1b2b2b3b': (-0.0017902805+0j), '1b2b2b4a': (0.0052231034+0j), '1b2b2b4b': (0.000842018+0j), '1b2b3a3a': (-0.0176825356+0j), '1b2b3a3b': (0.0487043854+0j), '1b2b3a4a': (0.080607269+0j), '1b2b3a4b': (-0.0795296764+0j), '1b2b3b3b': (-0.0006165815+0j), '1b2b3b4a': (-0.0458387405+0j), '1b2b3b4b': (0.0097373296+0j), '1b2b4a4a': (-0.0066878503+0j), '1b2b4a4b': (0.0486781634+0j), '1b2b4b4b': (-0.0026638056+0j), '1b3a3a3a': (-0.0006913582+0j), '1b3a3a3b': (0.0269163987+0j), '1b3a3a4a': (0.0052231034+0j), '1b3a3a4b': (-0.0233491222+0j), '1b3a3b3b': (-0.0064330776+0j), '1b3a3b4a': (-0.1279994759+0j), '1b3a3b4b': (0.0706289499+0j), '1b3a4a4a': (0.0006742159+0j), '1b3a4a4b': (0.0888805396+0j), '1b3a4b4b': (-0.009694694+0j), '1b3b3b3b': (3.0964e-06+0j), '1b3b3b4a': (0.0115510295+0j), '1b3b3b4b': (-0.0020453363+0j), '1b3b4a4a': (0.0115051172+0j), '1b3b4a4b': (-0.0490054999+0j), '1b3b4b4b': (0.0023382036+0j), '1b4a4a4a': (-0.0001250735+0j), '1b4a4a4b': (-0.0057495274+0j), '1b4a4b4b': (0.0044469317+0j), '1b4b4b4b': (-0.0001616404+0j), '2a2a2a2a': (-5.5361e-06+0j), '2a2a2a2b': (-0.0001697652+0j), '2a2a2a3a': (4.19329e-05+0j), '2a2a2a3b': (0.0003274266+0j), '2a2a2a4a': (3.0964e-06+0j), '2a2a2a4b': (-0.0003636562+0j), '2a2a2b2b': (0.0029296918+0j), '2a2a2b3a': (0.0041267154+0j), '2a2a2b3b': (-0.0244938441+0j), '2a2a2b4a': (-0.0064330776+0j), '2a2a2b4b': (0.0265143256+0j), '2a2a3a3a': (6.56449e-05+0j), '2a2a3a3b': (-0.0036240204+0j), '2a2a3a4a': (-0.0006165815+0j), '2a2a3a4b': (0.0037912005+0j), '2a2a3b3b': (0.003030279+0j), '2a2a3b4a': (0.0064045455+0j), '2a2a3b4b': (-0.0142768457+0j), '2a2a4a4a': (-8.47515e-05+0j), '2a2a4a4b': (-0.0027632519+0j), '2a2a4b4b': (0.0012703484+0j), '2a2b2b2b': (0.0006587087+0j), '2a2b2b3a': (-0.0256216544+0j), '2a2b2b3b': (-0.0052688796+0j), '2a2b2b4a': (0.0269163987+0j), '2a2b2b4b': (-0.0024457578+0j), '2a2b3a3a': (-0.0107997633+0j), '2a2b3a3b': (0.2048592547+0j), '2a2b3a4a': (0.0487043854+0j), '2a2b3a4b': (-0.2185418905+0j), '2a2b3b3b': (-0.0036240204+0j), '2a2b3b4a': (-0.2219390007+0j), '2a2b3b4b': (0.0637632573+0j), '2a2b4a4a': (-0.0032247109+0j), '2a2b4a4b': (0.2028856181+0j), '2a2b4b4b': (-0.013651745+0j), '2a3a3a3a': (0.0003415114+0j), '2a3a3a3b': (-0.0052688796+0j), '2a3a3a4a': (-0.0017902805+0j), '2a3a3a4b': (0.0055642647+0j), '2a3a3b3b': (-0.0244938441+0j), '2a3a3b4a': (-0.001288856+0j), '2a3a3b4b': (0.1027083633+0j), '2a3a4a4a': (0.0011024606+0j), '2a3a4a4b': (-0.021808607+0j), '2a3a4b4b': (-0.0073650275+0j), '2a3b3b3b': (0.0003274266+0j), '2a3b3b4a': (0.0261462921+0j), '2a3b3b4b': (-0.0063385161+0j), '2a3b4a4a': (0.002563281+0j), '2a3b4a4b': (-0.1009362666+0j), '2a3b4b4b': (0.0053319143+0j), '2a4a4a4a': (-0.0001719612+0j), '2a4a4a4b': (0.0048642307+0j), '2a4a4b4b': (0.0077588842+0j), '2a4b4b4b': (-0.0003296467+0j), '2b2b2b2b': (-2.94736e-05+0j), '2b2b2b3a': (0.0009657334+0j), '2b2b2b3b': (0.0003415114+0j), '2b2b2b4a': (-0.0006913582+0j), '2b2b2b4b': (-0.000260486+0j), '2b2b3a3a': (0.0038804413+0j), '2b2b3a3b': (-0.0107997633+0j), '2b2b3a4a': (-0.0176825356+0j), '2b2b3a4b': (0.018302975+0j), '2b2b3b3b': (6.56449e-05+0j), '2b2b3b4a': (0.0094930748+0j), '2b2b3b4b': (-0.0012651105+0j), '2b2b4a4a': (0.0015377199+0j), '2b2b4a4b': (-0.0120362431+0j), '2b2b4b4b': (0.0005728005+0j), '2b3a3a3a': (0.0009657334+0j), '2b3a3a3b': (-0.0256216544+0j), '2b3a3a4a': (-0.0070132571+0j), '2b3a3a4b': (0.0269579633+0j), '2b3a3b3b': (0.0041267154+0j), '2b3a3b4a': (0.1186621373+0j), '2b3a3b4b': (-0.0629515227+0j), '2b3a4a4a': (0.0003586332+0j), '2b3a4a4b': (-0.0994093266+0j), '2b3a4b4b': (0.0097541484+0j), '2b3b3b3b': (4.19329e-05+0j), '2b3b3b4a': (-0.009297248+0j), '2b3b3b4b': (0.0012436753+0j), '2b3b4a4a': (-0.0108625058+0j), '2b3b4a4b': (0.0486067987+0j), '2b3b4b4b': (-0.0022358831+0j), '2b4a4a4a': (6.60948e-05+0j), '2b4a4a4b': (0.0065055375+0j), '2b4a4b4b': (-0.0046156969+0j), '2b4b4b4b': (0.0001633947+0j), '3a3a3a3a': (-2.94736e-05+0j), '3a3a3a3b': (0.0006587087+0j), '3a3a3a4a': (0.0002446492+0j), '3a3a3a4b': (-0.0006459637+0j), '3a3a3b3b': (0.0029296918+0j), '3a3a3b4a': (-0.0032037463+0j), '3a3a3b4b': (-0.0100900773+0j), '3a3a4a4a': (-8.33017e-05+0j), '3a3a4a4b': (0.0041573737+0j), '3a3a4b4b': (0.0006190402+0j), '3a3b3b3b': (-0.0001697652+0j), '3a3b3b4a': (-0.0129585866+0j), '3a3b3b4b': (0.0045643498+0j), '3a3b4a4a': (-0.0004330469+0j), '3a3b4a4b': (0.0449011049+0j), '3a3b4b4b': (-0.002774715+0j), '3a4a4a4a': (5.15839e-05+0j), '3a4a4a4b': (-0.0021590464+0j), '3a4a4b4b': (-0.0033858108+0j), '3a4b4b4b': (0.0001614154+0j), '3b3b3b3b': (-5.5361e-06+0j), '3b3b3b4a': (0.0007077775+0j), '3b3b3b4b': (-4.82755e-05+0j), '3b3b4a4a': (0.0011926367+0j), '3b3b4a4b': (-0.0053386462+0j), '3b3b4b4b': (0.0002376493+0j), '3b4a4a4a': (3.98361e-05+0j), '3b4a4a4b': (-0.0033643845+0j), '3b4a4b4b': (0.0011512026+0j), '3b4b4b4b': (-3.61046e-05+0j), '4a4a4a4a': (-2.8406e-06+0j), '4a4a4a4b': (0.000126538+0j), '4a4a4b4b': (0.0002475508+0j), '4a4b4b4b': (-4.89426e-05+0j), '4b4b4b4b': (1.2768e-06+0j)}
# )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(initial_state)
print(basis_states)
print(len(basis_states))


eigenstates = False
max_number_of_states = np.inf
time_evolution = True
Fock_Space_Measure = False  #If false, prints occupation in the basis_label spatial occupation
t = np.linspace(0,0.5, 500) #in us


detuning_measurement = np.abs(20 * g)

Measure_currents = False
analytic_currents = False

current_measurements = [['1a', '1b'], ['2a', '2b'], ['3a', '3b']]
# current_measurements = [['1a', '2a'], ['1b', '2b']]

skip_number = 5  #if 1, then do all of them. if 10, then every 10 times
Measure_correlations = True
analytic_correlations = False

analytic_correlation_sonly = False


current_correlations = [['1a', '1b'], ['4a', '4b']]
# current_correlations = [['1a', '2a'], ['1b', '2b']]

# correlation_values = {'1a1b': 1, '1b2a': -1, '1a2b': -1, '2a2b': 1}
time_beam = np.abs(2 * np.pi / (8 * g))

energy_map = {}
for key in basis_labels:
    if key not in energy_map.keys():
        energy_map[key] = e

energy_map.update((key, value * 2 * np.pi) for key, value in energy_map.items())#

H = Hamiltonian(basis_states, U, energy_map, coupling_dictionary_same,
                         coupling_dictionary_NN, coupling_periodic)
Evolution_H = copy.copy(H.H)

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
        print(Time_correlations[0])

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

        print(occupation_numbers[0])

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