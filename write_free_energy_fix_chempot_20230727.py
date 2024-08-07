from scipy.integrate import trapz
import copy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl


K_B = 8.6173e-5

with open('results_energy.txt', 'r') as energy_total :
    data = {}
    lines = energy_total.readlines()
    for i in range(len(lines)) :
        tmp = lines[i].strip()
        tmp = tmp.split()
        if i == 0 :
            continue
        for j in range(len(tmp)) :
            T_tmp = 25 + (j - 1) * 25
            if i == 1 :
                if j == 0 :
                    data['conc'] = []
                    data['conc'].append(float(tmp[j]))
                else : 
                    data[str(T_tmp)] = []
                    data[str(T_tmp)].append(float(tmp[j]))
            else :
                if j == 0 :
                    data['conc'].append(float(tmp[j]))
                else : 
                    data[str(T_tmp)].append(float(tmp[j]))
    #print(data)
    data_free_r = {}
    data_free_total_r = {}
    data_free_r['conc'] = data['conc']
    data_free_total_r['conc'] = data['conc']

myData = []
for idx, (key, value) in enumerate(data.items()):
    if key == "conc":
        tmpConc = np.array(value)
        continue
    tmp = np.array(value)
    myData.append(tmp.copy())
    # plt.plot(tmp - tmpConc*tmp[-1] - (1-tmpConc)*tmp[0], c=cmap(idx/len(data)))
myData = np.array(myData)
myData = myData - np.outer(myData[:, -1],tmpConc) - np.outer(myData[:, 0],(1-tmpConc))

beta = []
for T in range(25, 925, 25) :
    beta.append(1 / ((float(T)) * K_B))

phiTable = np.zeros(myData.shape)
# # holding temperature the same
# for i in range(phiTable.shape[0]):
#     # integrate over mu
#     for j in range(phiTable.shape[1]):
#         pass
        
# # holding mu the same
# for i in range(phiTable.shape[1]):
#     # integrate over temperature
#     for j in range(phiTable.shape[0]):
#         if j == 0:
#             phiTable[-j-1, i] = myData[-j-1, i]
#         else:
#             phiTable[-1-j, i] = ((phiTable[-j-1, i]*beta[-j-1] + (myData[-1-j, i] + myData[-j, i])/2)*(beta[-1-j]-beta[-1-j]))/beta[-1-j]
#             # phiTable[-1-j, i] = (trapz(myData[-1-j:, i], beta[-1-j:]) + phiTable[0, i] / K_B / 900 )*beta[-1-j]

# cmap = mpl.colormaps['plasma']
# for idx, value in enumerate(phiTable):
#     tmp = np.array(value)
#     plt.plot(tmp, c=cmap(idx/len(data)))

# plt.show()
# exit()

    
beta_r = copy.deepcopy(beta)
beta_r.reverse()
    
# for i in range(len(data_free_r['conc'])) :
#     data_tmp = []
#     for T in range(25, 925, 25) :
#         data_tmp.append(data[str(T)][i])
#     data_tmp_r = copy.deepcopy(data_tmp)
#     data_tmp_r.reverse()
#     for j in range(len(data_tmp)) :
#         free_tmp_r = trapz(data_tmp_r[:j + 1], x=beta_r[:j + 1]) 
#         free_tmp_r = free_tmp_r + data['900'][i] * (1 / ((float(900)) * K_B))
#         T_tmp = 25 + j * 25
#         T_tmp_r = 900 - j * 25

#         if i == 0 :
#             data_free_r[str(T_tmp_r)] = []
#             data_free_r[str(T_tmp_r)].append(- free_tmp_r / 1000 * 96 * (float(T_tmp_r)) * K_B)
#         else : 
#             data_free_r[str(T_tmp_r)].append(- free_tmp_r / 1000 * 96 * (float(T_tmp_r)) * K_B)
            
# Calculate the free energy for each concentration and temperature
for i in range(len(data_free_r['conc'])):  # Loop through each concentration value
    print(i)
    data_tmp = []
    for T in range(25, 925, 25):  # Loop through temperatures from 25K to 900K
        data_tmp.append(data[str(T)][i])  # Collect energy data for the current concentration
    data_tmp_r = copy.deepcopy(data_tmp)  # Create a reversed copy of the energy data
    data_tmp_r.reverse()

    for j in range(len(data_tmp)):  # Loop through the energy data to calculate free energy
        T_tmp = 25 + j * 25  # Current temperature
        T_tmp_r = 900 - j * 25  # Corresponding reversed temperature
        print(j)
        # Integrate the reversed energy data with respect to reversed beta values up to the j-th point
        free_tmp_r = trapz(data_tmp_r[:j + 1], x=beta_r[:j + 1]) 
        # Add the energy contribution at the highest temperature (900K)
        free_tmp_r = free_tmp_r + data['900'][i] * (1 / (float(900) * K_B))
        # if j == 0:
        #     free_tmp_r = data['25'][i] * beta[0]
        # else:
        #     free_tmp_r = (data_tmp[j] + data_tmp[j-1])/2*(beta[j]-beta[j-1])
        #     free_tmp_r = phiTable[j-1, i] * beta[j-1] + free_tmp_r
        
        phiTable[j, i] = free_tmp_r  * float(T_tmp_r) * K_B
        # Initialize the free energy list for the current reversed temperature if it's the first concentration value
        # if i == 0:
        #     data_free_r[str(T_tmp_r)] = []
        #     # Calculate the free energy in kJ/mol and append to the list
        #     data_free_r[str(T_tmp_r)].append(-free_tmp_r  * float(T_tmp_r) * K_B)
        # else:
        #     # Calculate the free energy in kJ/mol and append to the existing list
        #     data_free_r[str(T_tmp_r)].append(-free_tmp_r * float(T_tmp_r) * K_B)

# for T in range(25, 925, 25) :
#     free = []
#     free_energy_mix_tmp = []
#     free_energy_mix_tmp_r = []
#     for i in range(len(data_free_r['conc'])) :
#         free_tmp_r = float(data_free_r[str(T)][i]) \
#             - float(data_free_r['conc'][i]) * float(data_free_r[str(T)][len(data_free_r['conc']) - 1]) \
#                 - (1 - float(data_free_r['conc'][i])) * float(data_free_r[str(T)][0])
                
#         free_energy_mix_tmp_r.append(- free_tmp_r)
#     data_free_total_r[str(T)] = free_energy_mix_tmp_r
#     #print(free_energy_mix_tmp)

# cmap = mpl.colormaps['plasma']
# for idx, (key, value) in enumerate(data_free_total_r.items()):
#     if key == "conc":
#         tmpConc = np.array(value)
#         continue
#     tmp = np.array(value)
#     # plt.plot(tmp - tmpConc*tmp[-1] - (1-tmpConc)*tmp[0], c=cmap(idx/len(data)))
#     plt.plot(tmp, c=cmap(idx/len(data)))

# plt.show()
# exit()

phiTable = phiTable - np.outer(phiTable[:, -1],tmpConc) - np.outer(phiTable[:, 0],(1-tmpConc))

cmap = mpl.colormaps['plasma']
for i in range(phiTable.shape[0]):
    # plt.plot(tmp - tmpConc*tmp[-1] - (1-tmpConc)*tmp[0], c=cmap(idx/len(data)))
    plt.plot(phiTable[i], c=cmap(i/len(data)))

plt.show()
exit()


with open('free_energy_low.txt', 'w') as free_energy_file_low :
    free_energy_file_low.write('#{:<20}'.format('T'))
    for T in range(25, 925, 25) :
        free_energy_file_low.write('#{:<20}'.format(T))
    free_energy_file_low.write('\n')
    for j in range(len(data_free_r['conc'])) :
        for i in range(len(data_free_r)) : 
            if i == 0 :
                free_energy_file_low.write('{:<20.2f}'.format(data_free_r['conc'][j]))
            else :
                T_tmp = 25 + (i - 1) * 25
                free_energy_file_low.write('{:<20.6f}'.format(data_free_total_r[str(T_tmp)][j]))
        free_energy_file_low.write('\n')
