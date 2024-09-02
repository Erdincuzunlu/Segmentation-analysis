from unittest.mock import inplace

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score

file_path = "/Users/erdinc/Desktop/persona.csv"
df = pd.read_csv(file_path)
df.head()
df.info()
df.size
df.describe()


### Görev 1
### Soru.1 persona.csvdosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz
df.head()

import pandas as pd


def all_describe(df, head=5):
    """
    Bu fonksiyon, bir pandas DataFrame'in temel özet bilgilerini ve istatistikleri yazdırır:
    - DataFrame'in boyutu (shape)
    - Veri türleri (dtypes)
    - İlk birkaç satır (head)
    - Son birkaç satır (tail)
    - Eksik değerlerin sayısı (isnull().sum())
    - Tanımlayıcı istatistikler (describe)

    Args:
    df (pd.DataFrame): İncelenecek pandas DataFrame.
    head (int): 'head' ve 'tail' fonksiyonlarında gösterilecek satır sayısı (varsayılan 5).
    """
    print("########## Shape #######")
    print(df.shape)

    print("\n########## dtypes #######")
    print(df.dtypes)

    print("\n########## head ########")
    print(df.head(head))

    print("\n########## tail ########")
    print(df.tail(head))

    print("\n########## Missing Values #######")
    print(df.isnull().sum())

    print("\n########## Describe #######")
    print(df.describe(percentiles=[0, 0.05, 0.50, 0.95, 0.99, 1]).T)

all_describe(df)

df["AGE"].value_counts()
df["SEX"].value_counts()
#### Soru 2 Kaç uniqueSOURCE vardır? Frekansları nedir?
df.value_counts()
### 2 adet unıque soruce vardır ... Android ve IOS .

### Soru 3:Kaç uniquePRICE vardır?

df["PRICE"].nunique()

### Soru 4:Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

df["PRICE"].isnull().sum()
df["PRICE"].value_counts()

#### Soru 5:Hangi ülkeden kaçar tane satış olmuş?

df.head()
df["COUNTRY"].value_counts()

#### Soru 6:Ülkelere göre satışlardan toplam ne kadar kazanılmış?

df.head()

df.groupby("COUNTRY").agg({"PRICE": "sum"})

### Soru 7:SOURCE türlerine göre satış sayıları nedir?

df.groupby("SOURCE").agg({"PRICE": "sum"})

### Soru 8:Ülkelere göre PRICE ortalamaları nedir?

df.groupby("COUNTRY").agg({"PRICE": "mean"})

## Soru 9:SOURCE'laragöre PRICE ortalamaları nedir?

df.groupby("SOURCE").agg({"PRICE": "mean"})

## Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

####  Görev 2:  COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
df.head()

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})

print(agg_df)

###Görev 3:  Çıktıyı PRICE’agöre sıralayınız.

###Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE’a göre uygulayınız.

### Çıktıyı agg_df olarak kaydediniz.

agg_df = agg_df.sort_values("PRICE", ascending=False)    ### **** ascending methodu azalan şekilde sıralar...
print(agg_df)

## Görev 4:  Indekste yer alan isimleri değişken ismine çeviriniz.
agg_df.columns
df.head()

agg_df.reset_index()
agg_df.reset_index(inplace=True)
agg_df

### Görev 5:  Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz
price_by_age_cat = agg_df.groupby("AGE_CAT").agg({"PRICE": "mean"}).reset_index()
label = ['0_25', '25_30', '30_35', '35_45', '45_70']

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], [0, 25, 30, 35, 45, 70], labels=label)
agg_df.groupby("AGE_CAT").agg({"PRICE": "mean"})


df.head()
agg_df["AGE"].value_counts()
agg_df["PRICE"].value_counts()
df.head()

#### Görev 6:  Yeni seviye tabanlı müşterileri (persona) tanımlayınız

#Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
#Yeni eklenecek değişkenin adı: customers_level_based
#Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based değişkenini oluşturmanız gerekmektedir
# Dikkat! Listcomprehensionile customers_level_baseddeğerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18. Bunları groupby'aalıp priceortalamalarını almak gerekmektedir.
# Müşteri tanımlarını oluşturma
agg_df.drop("AGG_CAT", axis=1, inplace=True)
agg_df.columns

based = ["_".join(agg_df.loc[i, ["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]]) for i in range(len(agg_df))]


agg_df["customers_level_based"] = based

agg_df["customers_level_based"].value_counts()

based_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})

based_df

based_df.reset_index(inplace=True)
based_df

## Görev 7:  Yeni müşterileri (personaları) segmentlereayırınız
## Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’agöre 4 segmente ayırınız.,
label = ["D", "C", "B", "A"]
based_df["SEGMENT"]= pd.qcut(based_df["PRICE"], q=4,  labels=label)
based_df.groupby("SEGMENT").agg({"PRICE": ["min", "max", "sum"]})
# Segment A: En yüksek fiyat aralığına (36.49 TL - 49.00 TL) sahip ve toplam gelirde en büyük katkıyı yapan segment. Premium müşterilere hitap ediyor.
# 	•	Segment B: Orta-yüksek fiyat aralığında (34.05 TL - 36.00 TL) yer alıyor, ancak toplam geliri Segment A’dan düşük. Daha fazla müşteri çekmek için fırsat olabilir.
# 	•	Segment C: İstikrarlı bir fiyat aralığı (31.69 TL - 34.00 TL) ile güvenilir bir gelir kaynağı. Orta segmentteki müşterilere hitap ediyor.
# 	•	Segment D: En düşük fiyat aralığı (9.00 TL - 31.63 TL) ve en düşük toplam gelire sahip. Fiyat hassasiyeti yüksek müşteriler için uygun fiyatlı ürünler sunulabilir.


### Görev 8:  Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini  tahmin ediniz


new_user1 = "bra_android_female_19_23"

new_user2 = "usa_ios_male_24_30"

based_df[based_df["customers_level_based"] == new_user1]

based_df[based_df["customers_level_based"] == new_user2]

based_df