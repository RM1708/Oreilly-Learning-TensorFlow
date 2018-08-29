
# coding: utf-8

# In[ ]:


# coding: utf-8

# ### Estimator: contrib.learn using Feature columns
# 1. learn.LinearRegressor Learning TensorFlow: A Guide to Building Deep Learning Systems (Kindle Locations 4081-4082). O'Reilly Media. Kindle Edition. 

# In[1]:


import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In[2]:


# ### Generate example categorical data

# In[ ]:



POPULATION = 10000
#Generate POPULATION random, normally distributed, body weights with mean of 70kgs
#and standard deviation of 5 kgs.
#NOTE: The index in the list is an implicit id for each individual member of the
#population
list_of_individual_body_weights = np.random.randn(POPULATION)*5+70

#There are just 3 three species in the population. We give them ids 0, 1, & 2.
#Randomize the allocation of species to the membeers of the population.
population_species_id_list = np.random.randint(0,3,POPULATION)
species_specific_correction = [0.9,1,1.1]
height_units_per_unit_weight = [0.01,0.01, 0.01] # What if each species had a different value
list_of_individual_heights = np.array([list_of_individual_body_weights[i] *                                        height_units_per_unit_weight[species_id] +                    species_specific_correction[species_id]                    for i, species_id in enumerate(population_species_id_list)])
list_of_species_names = ['Goblin','Human','ManBears']

list_of_species_of_individual_members = [list_of_species_names[s]                                          for s in population_species_id_list]


# In[ ]:


#population_species_id_list[:10]


# In[ ]:


#population_species_id_list[:10] == 0


# In[ ]:


#list_of_individual_heights[:10][population_species_id_list[:10] == 0]


# ### plot and create data frame

# In[ ]:


colors = ['r','b','g']
PLOT_0 = 0
PLOT_1 = 1

list_of_individual_heights = list_of_individual_heights + np.random.randn(POPULATION)*0.015
#It is the implicit identical ordering of the 3 lists that allows putting the data into
#a table in a straight-forward manner.
table_species_weight_height =         pd.DataFrame({'Species':list_of_species_of_individual_members,                   'Weight':list_of_individual_body_weights,                   'Height':list_of_individual_heights})

_ , axarr = plt.subplots(1,2,figsize = [7,3])

hist_plot = axarr[PLOT_0]

hist_plot.set_ylim([0,260])
hist_plot.set_xlim([1.38,2.05])
NO_OF_BINS = 50
COLOR_BRIGHTNESS = 0.5 #0.1 gives really faded colors

#Position the text for the histogram
hist_plot.text(1.42,150,'Goblins')
hist_plot.text(1.63,210,'Humans')
hist_plot.text(1.85,150,'ManBears')

for ii in range(3):
    hist_plot.hist(list_of_individual_heights[\
                                population_species_id_list == ii],\
                   NO_OF_BINS,color=colors[ii],\
                   alpha=COLOR_BRIGHTNESS)
    hist_plot.set_xlabel('Height')
    hist_plot.set_ylabel('Frequency')
    hist_plot.set_title('Heights distribution')
    
hist_plot.legend(['Goblins','Humans','ManBears'],loc=2, shadow=True,prop={'size':6})

x_y_plot = axarr[PLOT_1]
x_y_plot.set_ylim([45,100])
x_y_plot.set_xlim([1.25, 2.1])

x_y_plot.plot(table_species_weight_height['Height'],        table_species_weight_height['Weight'],        'o',alpha=0.3,mfc='w',mec='b')
x_y_plot.set_xlabel('Weight')
x_y_plot.set_ylabel('Height')
x_y_plot.set_title('Heights vs. Weights')
    
plt.tight_layout()
plt.savefig('test.png', bbox_inches='tight', format='png', dpi=300)

plt.show()


# In[ ]:


def input_fn(table_species_weight_height):
    feature_cols = {}
    feature_cols['Weight'] = tf.constant(table_species_weight_height['Weight'].values)
    
    indices=[[i, 0] for i in range(table_species_weight_height['Species'].size)]
    values=table_species_weight_height['Species'].values
    dense_shape=[table_species_weight_height['Species'].size, 1]
            
    feature_cols['Species'] =  tf.SparseTensor(indices, values, dense_shape)
                    
    measured_heights = tf.constant(table_species_weight_height['Height'].values)

    return feature_cols, measured_heights


# In[ ]:


Weight = layers.real_valued_column("Weight")

Species = layers.sparse_column_with_keys(
    column_name="Species", keys=['Goblin','Human','ManBears'])


# In[ ]:


reg = learn.LinearRegressor(feature_columns=[Weight,Species])


# In[ ]:


reg.fit(input_fn=lambda:input_fn(table_species_weight_height), steps=25000)#steps=50000)


# In[ ]:


regressor_var_names = reg.get_variable_names()
#print regressor_var_names to see where the arguments for get_variable_value() come from.

w_w = reg.get_variable_value('linear/Weight/weight')
print('Estimate of linear slope (height_units_per_unit_weight): {}'.format(w_w))

s_w = reg.get_variable_value('linear/Species/weights')
b = reg.get_variable_value('linear/bias_weight')
estimate = (s_w + b)
print('\nEstimate of species specific correction for:\n Goblins: {}; \n Humans: {} \n ManBears: {} \n'.format(estimate[0],                                                                                                  estimate[1],                                                                                                   estimate[2]))


# In[ ]:


reg.get_variable_names()


# In[ ]:


w_w = reg.get_variable_value('linear/Weight/weight/r/Weight/weight/part_0/Ftrl')
print('Estimation for Weight:\n{}'.format(w_w))


# In[ ]:


w_w = reg.get_variable_value('linear/Weight/weight/r/Weight/weight/part_0/Ftrl_1')
print('Estimation for Weight:\n{}'.format(w_w))

