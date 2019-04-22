# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:46:02 2019

@author: akdea
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, segmentation, color
from skimage.future import graph
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import skimage
import pickle
#%%
def extFeature_carton(inp_arr):
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    f5 = []
    f6 = []
    f7 = []
    f8 = []
    f9 = []
    f10 = []
    f11 = []
    f12 = []
    f13 = []
    f14 = []
    f15 = []
    f16 = []
    f17 = []
    f18 = []
    f19 = []
    f20 = []
    f21 = []
    f22 = []
    f23 = []
    f24 = []
    f25 = []
    
    for ctr in range(0, len(inp_arr) - 25):
        f1.append(-inp_arr[ctr + 1] + inp_arr[ctr])
        f2.append(-inp_arr[ctr + 2] + inp_arr[ctr])
        f3.append(-inp_arr[ctr + 3] + inp_arr[ctr])
        f4.append(-inp_arr[ctr + 4] + inp_arr[ctr])
        f5.append(-inp_arr[ctr + 5] + inp_arr[ctr])
        f6.append(-inp_arr[ctr + 6] + inp_arr[ctr])
        f7.append(-inp_arr[ctr + 7] + inp_arr[ctr])
        f8.append(-inp_arr[ctr + 8] + inp_arr[ctr])
        f9.append(-inp_arr[ctr + 9] + inp_arr[ctr])
        f10.append(-inp_arr[ctr + 10] + inp_arr[ctr])
        f11.append(-inp_arr[ctr + 11] + inp_arr[ctr])
        f12.append(-inp_arr[ctr + 12] + inp_arr[ctr])
        f13.append(-inp_arr[ctr + 13] + inp_arr[ctr])
        f14.append(-inp_arr[ctr + 14] + inp_arr[ctr])
        f15.append(-inp_arr[ctr + 15] + inp_arr[ctr])
        f16.append(-inp_arr[ctr + 16] + inp_arr[ctr])
        f17.append(-inp_arr[ctr + 17] + inp_arr[ctr])
        f18.append(-inp_arr[ctr + 18] + inp_arr[ctr])
        f19.append(-inp_arr[ctr + 19] + inp_arr[ctr])
        f20.append(-inp_arr[ctr + 20] + inp_arr[ctr])
        f21.append(-inp_arr[ctr + 21] + inp_arr[ctr])
        f22.append(-inp_arr[ctr + 22] + inp_arr[ctr])
        f23.append(-inp_arr[ctr + 23] + inp_arr[ctr])
        f24.append(-inp_arr[ctr + 24] + inp_arr[ctr])
        f25.append(-inp_arr[ctr + 25] + inp_arr[ctr])

    return f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25
#%%
def extFeature_metal_plastic(inp_arr):
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    f5 = []
    f6 = []
    f7 = []
    f8 = []
    f9 = []
    f10 = []
    f11 = []
    f12 = []
    f13 = []
    f14 = []
    f15 = []
    f16 = []
    f17 = []
    f18 = []
    f19 = []
    f20 = []
    f21 = []
    f22 = []
    f23 = []
    f24 = []
    f25 = []
    
    for ctr in range(25, len(inp_arr)):
        f1.append(inp_arr[ctr - 1] - inp_arr[ctr])
        f2.append(inp_arr[ctr - 2] - inp_arr[ctr])
        f3.append(inp_arr[ctr - 3] - inp_arr[ctr])
        f4.append(inp_arr[ctr - 4] - inp_arr[ctr])
        f5.append(inp_arr[ctr - 5] - inp_arr[ctr])
        f6.append(inp_arr[ctr - 6] - inp_arr[ctr])
        f7.append(inp_arr[ctr - 7] - inp_arr[ctr])
        f8.append(inp_arr[ctr - 8] - inp_arr[ctr])
        f9.append(inp_arr[ctr - 9] - inp_arr[ctr])
        f10.append(inp_arr[ctr - 10] - inp_arr[ctr])
        f11.append(inp_arr[ctr - 11] - inp_arr[ctr])
        f12.append(inp_arr[ctr - 12] - inp_arr[ctr])
        f13.append(inp_arr[ctr - 13] - inp_arr[ctr])
        f14.append(inp_arr[ctr - 14] - inp_arr[ctr])
        f15.append(inp_arr[ctr - 15] - inp_arr[ctr])
        f16.append(inp_arr[ctr - 16] - inp_arr[ctr])
        f17.append(inp_arr[ctr - 17] - inp_arr[ctr])
        f18.append(inp_arr[ctr - 18] - inp_arr[ctr])
        f19.append(inp_arr[ctr - 19] - inp_arr[ctr])
        f20.append(inp_arr[ctr - 20] - inp_arr[ctr])
        f21.append(inp_arr[ctr - 21] - inp_arr[ctr])
        f22.append(inp_arr[ctr - 22] - inp_arr[ctr])
        f23.append(inp_arr[ctr - 23] - inp_arr[ctr])
        f24.append(inp_arr[ctr - 24] - inp_arr[ctr])
        f25.append(inp_arr[ctr - 25] - inp_arr[ctr])

    return f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25
#%%
clf_carton = pickle.load( open( "clf_carton.pkl", "rb" ) )
clf_metal = pickle.load( open( "clf_metal.pkl", "rb" ) )
(obj0_temp, obj1_temp) = pickle.load( open( "d168.pkl", "rb" ) ); test_unit = obj1_temp

f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, \
f16, f17, f18, f19, f20, f21, f22, f23, f24 = extFeature_carton(test_unit)
ext_features_dum = np.zeros((len(f1),25))
ext_features_dum[:,0] = f0; ext_features_dum[:,1] = f1; ext_features_dum[:,2] = f2
ext_features_dum[:,3] = f3; ext_features_dum[:,4] = f4; ext_features_dum[:,5] = f5
ext_features_dum[:,6] = f6; ext_features_dum[:,7] = f7; ext_features_dum[:,8] = f8
ext_features_dum[:,9] = f9; ext_features_dum[:,10] = f10; ext_features_dum[:,11] = f11
ext_features_dum[:,12] = f12; ext_features_dum[:,13] = f13; ext_features_dum[:,14] = f14
ext_features_dum[:,15] = f15; ext_features_dum[:,16] = f16; ext_features_dum[:,17] = f17
ext_features_dum[:,18] = f18; ext_features_dum[:,19] = f19; ext_features_dum[:,20] = f20
ext_features_dum[:,21] = f21; ext_features_dum[:,22] = f22; ext_features_dum[:,23] = f23
ext_features_dum[:,24] = f24

res = []
for ctr in range(len(ext_features_dum)):
    res.append(clf_carton.predict([[f0[ctr], f1[ctr], f2[ctr], \
                                    f3[ctr], f4[ctr], f5[ctr], f6[ctr], \
                                    f7[ctr], f8[ctr], f9[ctr], f10[ctr], \
                                    f11[ctr],f12[ctr], f13[ctr], f14[ctr], \
                                    f15[ctr], f16[ctr], f17[ctr],f18[ctr], \
                                    f19[ctr], f20[ctr], f21[ctr], f22[ctr], \
                                    f23[ctr],f24[ctr]]])[0])
res = np.array(res)

#plt.figure()
#fig, ax = plt.subplots()
#ax.plot(res * 100, label="Tahmin")
#ax.plot(test_unit, label="Sıcaklık")
#ax.legend(fontsize="xx-large")
plt.figure()
plt.subplot(1,2,1)
plt.plot(res * 100, label="Tahmin")
plt.plot(test_unit, label="Sıcaklık")
plt.legend(fontsize="xx-large")
plt.title("carton vs others")

a = res[np.array(test_unit).argmax() - 200 : np.array(test_unit).argmax() - 0]
#plt.plot(res[np.array(obj0_temp).argmax() : np.array(obj0_temp).argmax() + 150])
print("nonzero : ", sum(a))
print("zero    : ", len(a) - sum(a))

if sum(a) > 100:
    print("___carton___")
    carton = True
else:
    print("NOT carton")
    carton = False
    
if carton == False:
    
    f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24 = extFeature_metal_plastic(test_unit)
    metal_features_dum = np.zeros((len(f1),25))
    metal_features_dum[:,0] = f0; metal_features_dum[:,1] = f1; metal_features_dum[:,2] = f2
    metal_features_dum[:,3] = f3; metal_features_dum[:,4] = f4; metal_features_dum[:,5] = f5
    metal_features_dum[:,6] = f6; metal_features_dum[:,7] = f7; metal_features_dum[:,8] = f8
    metal_features_dum[:,9] = f9; metal_features_dum[:,10] = f10; metal_features_dum[:,11] = f11
    metal_features_dum[:,12] = f12; metal_features_dum[:,13] = f13; metal_features_dum[:,14] = f14
    metal_features_dum[:,15] = f15; metal_features_dum[:,16] = f16; metal_features_dum[:,17] = f17
    metal_features_dum[:,18] = f18; metal_features_dum[:,19] = f19; metal_features_dum[:,20] = f20
    metal_features_dum[:,21] = f21; metal_features_dum[:,22] = f22; metal_features_dum[:,23] = f23
    metal_features_dum[:,24] = f24
    
    res = []
    for ctr in range(len(metal_features_dum)):
        res.append(clf_metal.predict([[f0[ctr], f1[ctr], f2[ctr], f3[ctr], f4[ctr], f5[ctr], f6[ctr], f7[ctr], f8[ctr], f9[ctr], f10[ctr], f11[ctr],f12[ctr], f13[ctr], f14[ctr], f15[ctr], f16[ctr], f17[ctr],f18[ctr], f19[ctr], f20[ctr], f21[ctr], f22[ctr], f23[ctr],f24[ctr]]])[0])
    res = np.array(res)
    
    plt.subplot(1,2,2)
    plt.plot(res * 100, label="Tahmin")
    plt.plot(test_unit, label="Sıcaklık")
    plt.legend(fontsize="xx-large")
    plt.title("metal vs plastic")
    
    a = res[np.array(test_unit).argmax() : np.array(test_unit).argmax() + 200]
    #plt.plot(res[np.array(obj0_temp).argmax() : np.array(obj0_temp).argmax() + 200])
    print("nonzero : ", sum(a))
    print("zero    : ", len(a) - sum(a))
    if sum(a) > (len(a) - sum(a)):
        print("___metal___")
    
    else:
        print("___plastic___")
        