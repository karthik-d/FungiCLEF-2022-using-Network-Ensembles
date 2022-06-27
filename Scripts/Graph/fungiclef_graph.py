from matplotlib import pyplot as plt
import pandas as pd
from itertools import chain
import json
import numpy as np

threshold_val = 0.3
df=pd.read_csv('/content/drive/MyDrive/Colab/predict_3463.csv')
print(df.dtypes)


#op_file = open('fungipredict_wthresh_ensemble-01.csv', mode='w')
#op_writer = csv.writer(op_file, delimiter=',', quotechar='"')
#op_writer.writerow(['ObservationId', 'ClassId'])  


def process(df):
  conf3 = df.iloc[:,3].tolist()
  obsid = df.iloc[:,0].tolist()
  classid = df.iloc[:,2].tolist()
  #print(conf3[0])
  #print(type(conf3[0]))
  #conf31= list(conf3.split(" "))
  #print(conf31[0])
  #conf3_lst=[]
  #for i in conf3:
  #  conf3_lst.append(i)
  conf1_2d = []
  conf3_2d = []
  class_pre=[]
  df_new = pd.DataFrame(columns = ['ObservationId', 'ClassId'])
  
  k = 50
  for i,l,m in zip(conf3,obsid,classid):
    res = i.strip('][').split(', ')
    resc = m.strip('][').split(', ')
    #print(type(classid[0]))

    #print(type(res))
    #print(type(i))
    for j,z in zip(res,resc):
      #print((k))
      #res1 = res.split()
      #print(type(res1))
      #print(res1)
      
      
      
      #a_string = "1 2 3"
      a_list = j.split()
      c_list = z.split()
      map_object = map(float, a_list)
      map_object1 = map(int,c_list)
      #applies int() to a_list elements
      list_of_integers = list(map_object)
      list_of_int = list(map_object1)
      if( max(list_of_integers))  < threshold_val :
        class_pre.append(-1)
        class_cur=-1
      else:
        class_pre.append(list_of_int[list_of_integers.index(max(list_of_integers))])
        class_cur= list_of_int[list_of_integers.index(max(list_of_integers))]

      #print(max(list_of_integers))
      #print(list_of_integers)
      conf1_2d.append(max(list_of_integers))
      conf3_2d.append(list_of_integers)
      #df_new.append()
      df_new = df_new.append({'ObservationId' : l , 'ClassId' : class_cur}, 
                  ignore_index = True)
    #res1 = res.split()

    #print(res1)
    #lst = json.loads(i)
    #li = list(i.split(" "))

    #print(i)
    #print(res)
    #conf3_list.append(li)
    #k=k-1
    #if(k==0):
    # break


  #print(conf3_list)
  #lst = json.loads(conf3_list)
  #for i in conf3:
  #print(conf1_2d)

  #print(conf3_flat)
  conf1_final = np.array(conf1_2d)
  #plt.hist(conf3_final)
  

  #fig, ax = plt.subplots(figsize =(10, 7))
  #ax.hist(conf3_final, bins = [k for k in range(0,1,0.01)])
  plt.figure(figsize=(10, 6))
  plt.hist(conf1_final.astype('float') , bins=np.arange(0, 1, 0.04)) 
  plt.hist(conf1_final.astype('float') , bins=np.arange(0, 1, 0.0001), cumulative=True, histtype='step') 
  plt.axvline(pd.Series(conf1_2d).quantile(0.05), color='red', linestyle='dashed', linewidth=2)
  plt.xlabel('Highest Confidence Values')
  plt.ylabel('Frequency')
  print(pd.Series(conf1_2d).quantile(0.2))
  # Show plot
  plt.show() 

  #predict = pd.DataFrame(df_new , columns=['ObservationId', 'ClassId'])

  print(df_new)
  new_column_names = ['ObservationId', 'ClassId']
  df_new.to_csv('fungipredict_wthresh_effnetb6_01.csv', index=False, header=new_column_names)


process(pd.read_csv('/content/drive/MyDrive/Colab/pedict_3264.csv'))
process(pd.read_csv('/content/drive/MyDrive/Colab/pedict_3291.csv'))
process(pd.read_csv('/content/drive/MyDrive/Colab/pedict_3329.csv'))
process(pd.read_csv('/content/drive/MyDrive/Colab/predict_3218.csv'))
process(pd.read_csv('/content/drive/MyDrive/Colab/predict_3463.csv'))
process(pd.read_csv('/content/drive/MyDrive/Colab/predict_ensemble.csv'))
