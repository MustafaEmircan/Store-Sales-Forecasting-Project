##########################################  FEATURE ENG.   ##########################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
from pandas import to_datetime
from datetime import date
# dataset yükleme
filepath_train = r"/Users/mustafaemircan/PycharmProjects/Miuul_Group_Project/DATASETS/train.csv";
filepath_test = r"/Users/mustafaemircan/PycharmProjects/Miuul_Group_Project/DATASETS/test.csv";
filepath_store = r"/Users/mustafaemircan/PycharmProjects/Miuul_Group_Project/DATASETS/store.csv";

test = pd.read_csv(filepath_test)
store = pd.read_csv(filepath_store)
train = pd.read_csv(filepath_train)

train.columns = [col.lower() for col in train.columns]
store.columns = [col.lower() for col in store.columns]
test.columns = [col.lower() for col in test.columns]
train = train[(train["open"] != 0) & (train['sales'] != 0)]
train_store = pd.merge(train, store, how = 'inner', on = 'store')

train['date'] = pd.to_datetime(train['date'])
train.set_index('date', inplace=True)

train_store['date'] = pd.to_datetime(train_store['date'])
train_store.set_index('date', inplace=True)
train_store["sales_customers"] = train_store["sales"] * train_store["customers"]


                                     #####       FEATURE    ######
# Extract date-based features
train_store['NEW_Day'] = train_store.index.day
train_store['NEW_Month'] = train_store.index.month
train_store['NEW_Year'] = train_store.index.year
train_store['NEW_WeekOfYear'] = train_store.index.isocalendar().week
train_store['NEW_DayOfYear'] = train_store.index.dayofyear
train_store['NEW_IsWeekend'] = train_store.index.weekday >= 5
train_store['NEW_IsMonthStart'] = train_store.index.is_month_start
train_store['NEW_IsMonthEnd'] = train_store.index.is_month_end

# Create lag features for sales
for lag in [1, 7, 30]:
    train_store[f'NEW_Sales_Lag_{lag}'] = train_store.groupby('store')['sales'].shift(lag)

# Create rolling window features for sales
for window in [7, 30]:
    train_store[f'NEW_Rolling_Mean_Sales_{window}'] = train_store.groupby('store')['sales'].transform(lambda x: x.rolling(window).mean())
    train_store[f'NEW_Rolling_Sum_Sales_{window}'] = train_store.groupby('store')['sales'].transform(lambda x: x.rolling(window).sum())
    train_store[f'NEW_Rolling_Std_Sales_{window}'] = train_store.groupby('store')['sales'].transform(lambda x: x.rolling(window).std())

# Create exponential moving average features for sales
for span in [7, 30]:
    train_store[f'NEW_EMA_Sales_{span}'] = train_store.groupby('store')['sales'].transform(lambda x: x.ewm(span=span, adjust=False).mean())

# Create interaction terms
train_store['NEW_Promo_StateHoliday'] = train_store['promo'] * train_store['stateholiday']
train_store['NEW_Promo_SchoolHoliday'] = train_store['promo'] * train_store['schoolholiday']
train_store['NEW_DayOfWeek_Promo'] = train_store['dayofweek'] * train_store['promo']

# Create ratio features
train_store['NEW_Sales_per_Customer'] = train_store['sales'] / train_store['customers']
train_store['NEW_Promo2_Ratio'] = train_store['promo2'] / (train_store['competitiondistance'] + 1)

# Encode categorical variables
train_store['NEW_StoreType'] = train_store['storetype'].astype('category').cat.codes
train_store['NEW_Assortment'] = train_store['assortment'].astype('category').cat.codes

# Display the first few rows of the modified train_store dataframe
print(train_store.head())

#### #### #### #### #### #### #### FEATURELARI KATEGORİZE #### #### #### #### #### #### ####
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(train_store)



#### #### #### #### #### #### #### FEATURE SONRASI MISSING VALUES #### #### #### #### #### #### ####

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    """ burda değişkenlerin isimleri işte neymiş n_miss diğeri de ratio bu iki dataframe i neye göre birleştireyim diyor biz de sütunlara göre birleştir diyoruz axis =1 ile """
    print(missing_df, end="\n") # bir boşluk bırakmak için \n koyuyoruz

    if na_name:
        return na_columns

missing_values_table(train_store, True)