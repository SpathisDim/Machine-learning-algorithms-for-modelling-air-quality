import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import display


warnings.filterwarnings(action='ignore')
csv_file= r'C:\Users\Nhytu\Desktop\data.csv'

data = pd.read_csv(csv_file)
data.drop("Unnamed: 0",axis=1,inplace=True)
print(data.head())


#------------------------ DATA PREPROCESSING ---------------------------------
#print the info of the data
data.info()
print('Are there any duplicated values in our data ? : {}\n'.format(data.duplicated().any()))
print('The total number of null values in each column:')
new_data=data.replace('NoData', np.NaN)                                        #antikathistw tis NoData values me anagnwrisimo NaN gia to dataframe
new_data=new_data.replace('Samp<', np.NaN)
new_data=new_data.replace('OffScan', np.NaN)
#new_data=new_data["WS"].replace(to_replace="NoData", value=np.NaN)
print(new_data.isnull().sum())

#Conversion of degrees to wind directions
df=new_data #copy to df
df=df.astype({'WD': 'float64'})
df=df.astype({'WS': 'float64'})
df=df.astype({'PM10': 'float64'})

df.loc[(df['WD'] >337.5) | (df['WD']<=22.5),'WD1']= 'N'

df.loc[(df['WD'] >22.5) & (df['WD']<=67.5),'WD2']= 'NE'

df.loc[(df['WD'] >67.5) & (df['WD']<=112.5),'WD3']= 'Ε'

df.loc[(df['WD'] >112.5) & (df['WD']<=157.5),'WD4']= 'SE'

df.loc[(df['WD'] >157.5) & (df['WD']<=202.5),'WD5']= 'S'

df.loc[(df['WD'] >202.5) & (df['WD']<=247.5),'WD6']= 'SW'

df.loc[(df['WD'] >247.5) & (df['WD']<=292.5),'WD7']= 'W'

df.loc[(df['WD'] >292.5) & (df['WD']<=337.5),'WD8']= 'NW'

#kanw x.dropna gia na xefortwthw tis NaN times
df["WindDirection"]=df[['WD1','WD2','WD3','WD4','WD5','WD6','WD7','WD8']].apply(lambda x: ''.join(x.dropna().astype(str)),axis=1)
df=df.replace('24:00', '00:00')                                                #diorthwnw tin lanthasmeni wra

df['months']=df['Date'].str.split('/').str[1]
df['days']=df['Date'].str.split('/').str[0]
#df['years']=df['Date'].str.split().str[0]
df['hours']=df['Hours'].str.split(':').str[0]
df['nodays']=df['Date'].str.split('').str[0]

#ftiaxnw to datetime enwnw thn hmeromhnia me tis wres 
full_date=[]
for i in range(df.shape[0]):
       full_date.append(str(df['months'].values[i]+'-'+str(df['days'].values[i])+'-15'+' '+str(df['Hours'].values[i])))#default seira M D Y vazw 15 gt to kanei aytomata 2015
       #date_time= str(df['Date'].values[i])+'-'+str(df['Hours'].values[i])           
       #full_date.append(date_time)             
dates=pd.to_datetime(full_date)
dates=pd.DataFrame(dates,columns=['datetime'])
df=pd.concat([dates,df],axis=1)

exportdf=df[['hours','days','months','WindDirection','WS','PM10']]
exportdf.to_excel(r'C:\Users\Nhytu\Desktop\git thesis\2015\Καλοκαιρι\data\katw_Komi_Data.xls')

# Find the most appearing wind direction value and the max values 
print('\nMost appearing for Wind speed',df.WS.mode())
print('Most appearing for direction ',df.WindDirection.mode())
print('Most appearing for PM10 ', df.PM10.mode())
print('\nMax values \n', 'PM10:',df.PM10.max(),'\n WindSpeed:',df.WS.max() )

plt.figure(figsize=(12,5))
sns.distplot(df['PM10'],bins=50)
plt.title('Distribution of the hourly recorded PM10 concetration in the air (Katw-Komi)',fontsize=16)
plt.show()

# 1. Find the daily average of PM10 contained in the air in any given hour
daily_data = df[['datetime','PM10']]
#la=dt.datetime.strptime("daily_data['datetime']", "%Y-%d-%m %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
#daily_data['datetime']= daily_data['datetime'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
daily_data = daily_data.set_index('datetime')
#daily_data = daily_data.resample('D').median()
print('\n',daily_data.dtypes)

daily_data.plot(kind='line',figsize=(25,9), color='mediumseagreen',linewidth=1,marker='h',
                markerfacecolor='k',markeredgewidth=2,markersize=1, markevery=24)
plt.xlabel('Date',fontsize=14)
plt.ylabel('PM10 concentration',fontsize=14)
plt.title('Daily trend in the hourly recorded PM10 concentration in\nthe air in Katw-Komi',fontsize=16)
plt.show()
#xreiazomai polla dedomena gia na kanw to seasonal 
#decomposition = seasonal_decompose(daily_data,model='addictive',filt=None, freq=None)

# plot the data

# 2. In which month does the amount of PM10 contained in the air rises 
#diorthwnw to format twn mhnwn px. apo 08 se 8 gia na antistoixisw to fullname enumerate stous hdh yparxontes mhnes
df['months']=df['months'].replace('08',8)
df['months']=df['months'].replace('07',7)
df['months']=df['months'].replace('06',6)
df=df.astype({'months': 'int64'})

monthly_data=df[['months','PM10']]
months2 = ['January','February','March','April','May','June','July',
           'August','September','October','November','December']
monthdf = pd.DataFrame(months2,columns=['months'])
monthFullname={}
for i,j in enumerate(months2):
    monthFullname.setdefault(i+1,j)                                            #i+1 giati thelw na xekinaei apo 1(Ianouarios) kai oxi apo 0

#antistoixisi sto dataframe twn mhnwn -> olografws tous mhnes
monthly_data.months = monthly_data.months.map(monthFullname)
monthly_avg=monthly_data.groupby('months').median()                            #Ton meso oro PM10 ana mina
monthly_avg = pd.merge(monthdf,monthly_avg,left_on='months',right_index=True)
#strogilopoihsh ton meso oro se periptwsh pou xreiastei
monthly_avg = np.round(monthly_avg,1)
monthly_avg=monthly_avg.set_index('months')

with plt.style.context('ggplot'):
    monthly_avg.plot(figsize=(12,7),kind='bar',color='mediumseagreen',linewidth=1)
    plt.xlabel('Months',fontsize=12)
    plt.ylabel('PM10 concentration ',fontsize=12)
    plt.title('Monthly average of the hourly recorded PM10 concentration in\nthe air in Katw-Komi',fontsize=14)
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()

   
#diorthwnw to format ths wras apo 01 02 se 1,2 klp.
df['hours']=df['hours'].replace({'00':0, '01':1})    
  
hours=['12 AM','1 AM','2 AM','3 AM','4 AM','5 AM','6 AM','7 AM','8 AM','9 AM','10 AM',
      '11 AM','12 PM','1 PM','2 PM','3 PM','4 PM','5 PM','6 PM','7 PM','8 PM','9 PM','10 PM','11 PM']
hours_matching={}
for i,j in enumerate(hours):
    hours_matching.setdefault(i,j)

hourly_avg=df[['hours','PM10']].groupby('hours').median().reset_index()
hourly_avg.hours=hourly_avg.hours.map(hours_matching)
hourly_avg=hourly_avg.set_index('hours')

with plt.style.context('ggplot'):
    hourly_avg.plot(figsize=(12,8),color='mediumseagreen',kind='barh',linewidth=1)
    plt.ylabel('Hours',fontsize=12)
    plt.xlabel('PM10 concentration',fontsize=12)
    plt.title('Average recorded PM10 concentration in the air by the hour of the day',fontsize=16)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

wind_data=df[['WindDirection','PM10']]
wind_data=wind_data.groupby('WindDirection').median()

with plt.style.context('ggplot'):
    wind_data.plot(figsize=(12,8),color='mediumseagreen',kind='barh',linewidth=1)
    plt.ylabel('Wind direction',fontsize=12)
    plt.xlabel('PM10 concentration',fontsize=12)
    plt.title('Average hourly recorded PM10 concentration in the air grouped by wind direction',fontsize=16)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
wind_speed=df[['WS','WindDirection']]
wind_speed=wind_speed.groupby('WindDirection').median()

with plt.style.context('ggplot'):
    wind_speed.plot(figsize=(12,8),color='mediumseagreen',kind='barh',linewidth=1)
    plt.ylabel('WD',fontsize=12)
    plt.xlabel('AVG WS ',fontsize=12)
    plt.title('Average wind speed by wind direction',fontsize=16)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
   
#-----------------------------Taking care of missing data----------------------
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(exportdf.iloc[:,4:].values)
exportdf.iloc[:,4:] = imputer.transform(exportdf.iloc[:,4:].values)


#--------------------- Encoding categorical data -----------------------------
# indepedent variables(predictor variables) (1->N  2->NE  3->NW  4->S   5->SE  6->SW   7->W  8->E) TO 0 exei tis nan times
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
categorical_data = np.array(ct.fit_transform(exportdf))
print(categorical_data)
categorical_data=pd.DataFrame(categorical_data)                                #isws xreiastei na dwsw column names
categorical_data=categorical_data.rename(columns={13:"PM10"})                  #den allazw ta columns names twn 9:"datetime",10:"WS" giati exw thema sto standardisation 
categorical_data = categorical_data.astype({0:'category',1:'category',2:'category',3:'category',4:'category',
                                            5:'category',6:'category',7:'category',8:'category',9:'category',10:'float64','PM10':'float64'}) #kanw float tis times tis sthlhs 10 poy einai to WIND SPEED
categorical_data=categorical_data.drop(0,axis=1)
#categorical_data=categorical_data.drop(9,axis=1)
#vazw sto DataFrame stoixeia gia ton syntelesti xrhsimopoihshs twn monadwn paragwghs energeias
categorical_data=categorical_data.astype({11:'float64'})
categorical_data.loc[(categorical_data[11]== 1, 'Jan')]= 65
categorical_data.loc[(categorical_data[11]== 2, 'Feb')]= 54.3
categorical_data.loc[(categorical_data[11]== 3, 'Mar')]= 58.3
categorical_data.loc[(categorical_data[11]== 4, 'Apr')]= 43.5
categorical_data.loc[(categorical_data[11]== 5, 'May')]= 51.4
categorical_data.loc[(categorical_data[11]== 6, 'Jun')]= 55.3
categorical_data.loc[(categorical_data[11]== 7, 'Jul')]= 79
categorical_data.loc[(categorical_data[11]== 8, 'Aug')]= 64.2
categorical_data.loc[(categorical_data[11]== 9, 'Sep')]= 64.6
categorical_data.loc[(categorical_data[11]== 10, 'Oct')]= 49.5
categorical_data.loc[(categorical_data[11]== 11, 'Nov')]= 58.3
categorical_data.loc[(categorical_data[11]== 12, 'Dec')]= 60.2
categorical_data["Syntelestis Xrhsimopoihshs"]=categorical_data[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']].apply(lambda x: ''.join(x.dropna().astype(str)),axis=1)

#vazw sto DataFrame stoixeia gia tin mesh mhniaia thermokrasia 
categorical_data=categorical_data.astype({11:'float64'})
categorical_data.loc[(categorical_data[11]== 1, 'Jan')]= 2.9
categorical_data.loc[(categorical_data[11]== 2, 'Feb')]= 3.3
categorical_data.loc[(categorical_data[11]== 3, 'Mar')]= 5.8
categorical_data.loc[(categorical_data[11]== 4, 'Apr')]= 11.8
categorical_data.loc[(categorical_data[11]== 5, 'May')]= 18.5
categorical_data.loc[(categorical_data[11]== 6, 'Jun')]= 20.7
categorical_data.loc[(categorical_data[11]== 7, 'Jul')]= 26.8
categorical_data.loc[(categorical_data[11]== 8, 'Aug')]= 24.3
categorical_data.loc[(categorical_data[11]== 9, 'Sep')]= 21.5
categorical_data.loc[(categorical_data[11]== 10, 'Oct')]= 13.9
categorical_data.loc[(categorical_data[11]== 11, 'Nov')]= 11
categorical_data.loc[(categorical_data[11]== 12, 'Dec')]= 4.9
categorical_data["Tmean"]=categorical_data[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']].apply(lambda x: ''.join(x.dropna().astype(str)),axis=1)

#vazw sto DataFrame stoixeia gia tin miniaia mesh timi tou Yetou 
categorical_data=categorical_data.astype({11:'float64'})
categorical_data.loc[(categorical_data[11]== 1, 'Jan')]= 41
categorical_data.loc[(categorical_data[11]== 2, 'Feb')]= 90
categorical_data.loc[(categorical_data[11]== 3, 'Mar')]= 94
categorical_data.loc[(categorical_data[11]== 4, 'Apr')]= 28
categorical_data.loc[(categorical_data[11]== 5, 'May')]= 69
categorical_data.loc[(categorical_data[11]== 6, 'Jun')]= 37
categorical_data.loc[(categorical_data[11]== 7, 'Jul')]= 33
categorical_data.loc[(categorical_data[11]== 8, 'Aug')]= 69
categorical_data.loc[(categorical_data[11]== 9, 'Sep')]= 112
categorical_data.loc[(categorical_data[11]== 10, 'Oct')]= 120
categorical_data.loc[(categorical_data[11]== 11, 'Nov')]= 35
categorical_data.loc[(categorical_data[11]== 12, 'Dec')]= 2
categorical_data["Yetos"]=categorical_data[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']].apply(lambda x: ''.join(x.dropna().astype(str)),axis=1)

categorical_data=categorical_data[[1,2,3,4,5,6,7,8,9,10,11,12,'PM10','Syntelestis Xrhsimopoihshs','Tmean','Yetos']]
categorical_data=categorical_data.astype({9: 'float64'})
categorical_data=categorical_data.astype({'Syntelestis Xrhsimopoihshs':'float64'})
categorical_data=categorical_data.astype({12:'float64'})
categorical_data=categorical_data.astype({'Tmean' :'float64'})
categorical_data=categorical_data.astype({'Yetos' :'float64'})
categorical_data=categorical_data.rename(columns={9:'hours',10:'days',11:'months',12:'WS','Syntelestis Xrhsimopoihshs':'Synt Xrhs'})

print('Correlation matrix\n',categorical_data.corr())
sns.set()
plt.figure(figsize=(13,9))
correlation_data = categorical_data[['hours', 'days', 'months', 'WS','PM10','Synt Xrhs', 'Tmean', 'Yetos']]
sns.heatmap(correlation_data.corr(),cmap=plt.cm.Reds,annot=True)
plt.title('Heatmap displaying the correlation matrix of the variables',fontsize=16)
plt.show()
# ------ Splitting the dataset into the Training set and Test set -------------
X= categorical_data.drop('PM10',axis=1)                                        #indepedent variable
y=categorical_data['PM10']                                                     #depedent variable

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()                                                          #standardisation
colNumber=8                                                                    
X_train.iloc[:, colNumber:] = sc.fit_transform(X_train.iloc[:, colNumber:].values)#Me tin fit_transform upologizw tin mesi timi kai tin tupiki apoklisi 
X_test.iloc[:, colNumber:] = sc.transform(X_test.iloc[:, colNumber:].values)
