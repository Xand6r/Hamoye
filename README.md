

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
dataset=pd.read_csv('hamoye.csv')
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_id</th>
      <th>utility_id_ferc1</th>
      <th>report_year</th>
      <th>plant_name_ferc1</th>
      <th>fuel_type_code_pudl</th>
      <th>fuel_unit</th>
      <th>fuel_qty_burned</th>
      <th>fuel_mmbtu_per_unit</th>
      <th>fuel_cost_per_unit_burned</th>
      <th>fuel_cost_per_unit_delivered</th>
      <th>fuel_cost_per_mmbtu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f1_fuel_1994_12_1_0_7</td>
      <td>1</td>
      <td>1994</td>
      <td>rockport</td>
      <td>coal</td>
      <td>ton</td>
      <td>5377489.0</td>
      <td>16.590</td>
      <td>18.59</td>
      <td>18.53</td>
      <td>1.121</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f1_fuel_1994_12_1_0_10</td>
      <td>1</td>
      <td>1994</td>
      <td>rockport total plant</td>
      <td>coal</td>
      <td>ton</td>
      <td>10486945.0</td>
      <td>16.592</td>
      <td>18.58</td>
      <td>18.53</td>
      <td>1.120</td>
    </tr>
    <tr>
      <th>2</th>
      <td>f1_fuel_1994_12_2_0_1</td>
      <td>2</td>
      <td>1994</td>
      <td>gorgas</td>
      <td>coal</td>
      <td>ton</td>
      <td>2978683.0</td>
      <td>24.130</td>
      <td>39.72</td>
      <td>38.12</td>
      <td>1.650</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f1_fuel_1994_12_2_0_7</td>
      <td>2</td>
      <td>1994</td>
      <td>barry</td>
      <td>coal</td>
      <td>ton</td>
      <td>3739484.0</td>
      <td>23.950</td>
      <td>47.21</td>
      <td>45.99</td>
      <td>1.970</td>
    </tr>
    <tr>
      <th>4</th>
      <td>f1_fuel_1994_12_2_0_10</td>
      <td>2</td>
      <td>1994</td>
      <td>chickasaw</td>
      <td>gas</td>
      <td>mcf</td>
      <td>40533.0</td>
      <td>1.000</td>
      <td>2.77</td>
      <td>2.77</td>
      <td>2.570</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Question 3
#which fuel type has the lowest averate fuel cost per unit burned
```


```python
dataset.groupby('fuel_type_code_pudl')['fuel_cost_per_unit_burned'].mean().sort_values()
```




    fuel_type_code_pudl
    gas          13.659397
    other        18.253856
    waste        19.518122
    coal         67.421830
    oil         168.877086
    nuclear    4955.157002
    Name: fuel_cost_per_unit_burned, dtype: float64




```python
#Question 4
#std and 75th percentile of the measure of energy per unit
```


```python
dataset['fuel_mmbtu_per_unit'].describe()
```




    count    29523.000000
    mean         8.492111
    std         10.600220
    min          0.000001
    25%          1.024000
    50%          5.762694
    75%         17.006000
    max        341.260000
    Name: fuel_mmbtu_per_unit, dtype: float64




```python
#Question 5
#skew and kurtosis for fuel quantity burned
```


```python
dataset['fuel_qty_burned'].skew(), dataset['fuel_qty_burned'].kurtosis()
```




    (15.851495469109503, 651.3694501337732)




```python
#Question 6
# missing features column name, value, and percentage
```


```python
dataset.isna().sum()
```




    record_id                         0
    utility_id_ferc1                  0
    report_year                       0
    plant_name_ferc1                  0
    fuel_type_code_pudl               0
    fuel_unit                       180
    fuel_qty_burned                   0
    fuel_mmbtu_per_unit               0
    fuel_cost_per_unit_burned         0
    fuel_cost_per_unit_delivered      0
    fuel_cost_per_mmbtu               0
    dtype: int64




```python
(dataset.isna().sum().loc['fuel_unit'] / dataset.shape[0])*100
```




    0.609694136774718




```python
# Question 7
# The feature with missing values falls under which categories
# what imputationtechnique would you use
```


```python
dataset['fuel_unit'].dtype
```




    dtype('O')




```python
# Question 8
# which feature has teh second and third lowest correlation 
# with fuel cost per unit burned
```


```python
dataset.corr()['fuel_cost_per_unit_burned'].sort_values()
```




    utility_id_ferc1               -0.037863
    fuel_qty_burned                -0.018535
    fuel_mmbtu_per_unit            -0.010034
    fuel_cost_per_mmbtu            -0.000437
    fuel_cost_per_unit_delivered    0.011007
    report_year                     0.013599
    fuel_cost_per_unit_burned       1.000000
    Name: fuel_cost_per_unit_burned, dtype: float64




```python
# Question 9
# for fuel type coal what is the percentage 
# in fuel cost per unit burnned in 1998 compared to 1994
```


```python
# filter the dataset to include only 1994 and 1998
year_mask = dataset['report_year'].map(lambda year: year in [1994, 1998])
dataset_1994_and_1998 = dataset[year_mask]
```


```python
# filter the dataset to include only coal
coal_dataset=dataset_1994_and_1998[dataset_1994_and_1998['fuel_type_code_pudl'] == 'coal']
```


```python
coal_dataset.groupby('report_year')['fuel_cost_per_unit_burned'].sum()
```




    report_year
    1994    14984.572
    1998    11902.597
    Name: fuel_cost_per_unit_burned, dtype: float64




```python
 round((11902.597 - 14984.572)/14984.572,2) *100
```




    -21.0




```python
#Question 10
# which year has teh highest fuel cost per unit delivered
```


```python
dataset.groupby('report_year')['fuel_cost_per_unit_delivered'].mean().sort_values()
```




    report_year
    1999       25.551627
    1995       32.735269
    2006       38.657484
    2005       41.438184
    2007       43.325023
    2017       46.196861
    2002       47.594361
    2003       55.663493
    2008       58.588197
    2011       59.774667
    2001       60.050396
    2012       60.994502
    1994       63.636060
    2010       91.862105
    2016      103.901761
    2004      139.524275
    2013      172.307591
    2014      192.737183
    1998      287.154420
    2015      326.535511
    2018      499.269966
    2009      652.694163
    2000      985.362877
    1996     9196.705948
    1997    11140.197239
    Name: fuel_cost_per_unit_delivered, dtype: float64




```python
dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_id</th>
      <th>utility_id_ferc1</th>
      <th>report_year</th>
      <th>plant_name_ferc1</th>
      <th>fuel_type_code_pudl</th>
      <th>fuel_unit</th>
      <th>fuel_qty_burned</th>
      <th>fuel_mmbtu_per_unit</th>
      <th>fuel_cost_per_unit_burned</th>
      <th>fuel_cost_per_unit_delivered</th>
      <th>fuel_cost_per_mmbtu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f1_fuel_1994_12_1_0_7</td>
      <td>1</td>
      <td>1994</td>
      <td>rockport</td>
      <td>coal</td>
      <td>ton</td>
      <td>5377489.0</td>
      <td>16.590</td>
      <td>18.59</td>
      <td>18.53</td>
      <td>1.121</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f1_fuel_1994_12_1_0_10</td>
      <td>1</td>
      <td>1994</td>
      <td>rockport total plant</td>
      <td>coal</td>
      <td>ton</td>
      <td>10486945.0</td>
      <td>16.592</td>
      <td>18.58</td>
      <td>18.53</td>
      <td>1.120</td>
    </tr>
    <tr>
      <th>2</th>
      <td>f1_fuel_1994_12_2_0_1</td>
      <td>2</td>
      <td>1994</td>
      <td>gorgas</td>
      <td>coal</td>
      <td>ton</td>
      <td>2978683.0</td>
      <td>24.130</td>
      <td>39.72</td>
      <td>38.12</td>
      <td>1.650</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f1_fuel_1994_12_2_0_7</td>
      <td>2</td>
      <td>1994</td>
      <td>barry</td>
      <td>coal</td>
      <td>ton</td>
      <td>3739484.0</td>
      <td>23.950</td>
      <td>47.21</td>
      <td>45.99</td>
      <td>1.970</td>
    </tr>
    <tr>
      <th>4</th>
      <td>f1_fuel_1994_12_2_0_10</td>
      <td>2</td>
      <td>1994</td>
      <td>chickasaw</td>
      <td>gas</td>
      <td>mcf</td>
      <td>40533.0</td>
      <td>1.000</td>
      <td>2.77</td>
      <td>2.77</td>
      <td>2.570</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>29518</th>
      <td>f1_fuel_2018_12_12_0_13</td>
      <td>12</td>
      <td>2018</td>
      <td>neil simpson ct #1</td>
      <td>gas</td>
      <td>mcf</td>
      <td>18799.0</td>
      <td>1.059</td>
      <td>4.78</td>
      <td>4.78</td>
      <td>9.030</td>
    </tr>
    <tr>
      <th>29519</th>
      <td>f1_fuel_2018_12_12_1_1</td>
      <td>12</td>
      <td>2018</td>
      <td>cheyenne prairie 58%</td>
      <td>gas</td>
      <td>mcf</td>
      <td>806730.0</td>
      <td>1.050</td>
      <td>3.65</td>
      <td>3.65</td>
      <td>6.950</td>
    </tr>
    <tr>
      <th>29520</th>
      <td>f1_fuel_2018_12_12_1_10</td>
      <td>12</td>
      <td>2018</td>
      <td>lange ct facility</td>
      <td>gas</td>
      <td>mcf</td>
      <td>104554.0</td>
      <td>1.060</td>
      <td>4.77</td>
      <td>4.77</td>
      <td>8.990</td>
    </tr>
    <tr>
      <th>29521</th>
      <td>f1_fuel_2018_12_12_1_13</td>
      <td>12</td>
      <td>2018</td>
      <td>wygen 3 bhp 52%</td>
      <td>coal</td>
      <td>ton</td>
      <td>315945.0</td>
      <td>16.108</td>
      <td>3.06</td>
      <td>14.76</td>
      <td>1.110</td>
    </tr>
    <tr>
      <th>29522</th>
      <td>f1_fuel_2018_12_12_1_14</td>
      <td>12</td>
      <td>2018</td>
      <td>wygen 3 bhp 52%</td>
      <td>gas</td>
      <td>mcf</td>
      <td>17853.0</td>
      <td>1.059</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>11.680</td>
    </tr>
  </tbody>
</table>
<p>29523 rows × 11 columns</p>
</div>


