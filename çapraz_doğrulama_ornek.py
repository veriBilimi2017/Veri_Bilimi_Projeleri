#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 20:51:03 2018

@author: sait
"""
#***** Çapraz Doğrulama Çalışması*****

# İlk önce kullanacağımız kütüphaneleri çalışma alanımıza dahil ediyoruz
#sklearn = Scikit Learn

from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target

#cross_val_predict, her bir girişin çapraz doğrulama ile elde edilen bir tahmin olduğu,
#"y" ile aynı boyutta bir dizi döndürür

tahmin_edilen = cross_val_predict(lr, boston.data, y, cv = 10)

fig, ax = plt.subplots()
ax.scatter(y, tahmin_edilen, edgecolors = (0.01, 0.01, 0.01))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 3)
ax.set_xlabel('Ölçülen')
ax.set_ylabel('Tahmin Edilen')
plt.show()
