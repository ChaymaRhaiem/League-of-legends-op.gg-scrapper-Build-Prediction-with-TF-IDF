import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


"""### DATA GATHERING"""
"""
P1 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\001-02-12-04.csv")
P2 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\101-02-12-06.csv")
P3 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\201-02-12-08.csv")
P4 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\301-02-12-10.csv")
P5 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\401-02-12-11.csv")
P6 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\501-02-12-13.csv")
P7 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\601-02-12-15.csv")
P8 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\701-02-12-17.csv")
P9 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\801-02-12-19.csv")
P10 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\901-02-12-21.csv")
P11 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\001-02-12-23.csv")

P12 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\201-02-12-27.csv")
P13 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\301-02-12-28.csv")
P14 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\101-02-12-25.csv")
P15 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\901-02-12-38.csv")
P16 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\301-02-12-46.csv")
P17 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\301-02-12-28.csv")
P18 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\201-02-12-44.csv")
P19 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\101-02-12-42.csv")
P20 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\001-02-12-40.csv")
P21 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\801-02-12-36.csv")
P22 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\701-02-12-35.csv")
P23 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\601-02-12-33.csv")
P24 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\501-02-12-31.csv")
P25 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\401-02-12-30.csv")
P26=pd.read_csv(r"C:\Users\Lenovo\Documents\ML\301-03-19-45.csv")
P27=pd.read_csv(r"C:\Users\Lenovo\Documents\ML\201-03-19-43.csv")
P28=pd.read_csv(r"C:\Users\Lenovo\Documents\ML\101-03-19-41.csv")
P29=pd.read_csv(r"C:\Users\Lenovo\Documents\ML\001-03-19-39.csv")
P30=pd.read_csv(r"C:\Users\Lenovo\Documents\ML\101-03-11-04.csv")
P31=pd.read_csv(r"C:\Users\Lenovo\Documents\ML\001-03-10-59.csv")
P32=pd.read_csv(r"C:\Users\Lenovo\Documents\ML\901-03-10-54.csv")
P33=pd.read_csv(r"C:\Users\Lenovo\Documents\ML\801-03-10-53.csv")
P34=pd.read_csv(r"C:\Users\Lenovo\Documents\ML\701-03-10-50.csv")
P35=pd.read_csv(r"C:\Users\Lenovo\Documents\ML\601-03-10-47.csv")
P36=pd.read_csv(r"C:\Users\Lenovo\Documents\ML\501-03-10-46.csv")
P37=pd.read_csv(r"C:\Users\Lenovo\Documents\ML\401-03-10-42.csv")
P38=pd.read_csv(r"C:\Users\Lenovo\Documents\ML\301-03-10-41.csv")
P39=pd.read_csv(r"C:\Users\Lenovo\Documents\ML\201-03-10-37.csv")
P40=pd.read_csv(r"C:\Users\Lenovo\Documents\ML\001-03-10-32.csv")

df5=pd.concat([P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18,P19,P20,P21,P22,P23,P24,P25,P26,P27,P28,P29,P30,P31,P32,P33,P34,P35,P36,P37,P38,P39,P40],ignore_index=True).drop_duplicates()

#image.x
champion_data.drop(['image.x'],axis=1,inplace=True)
champion_data.drop(['image.y'],axis=1,inplace=True)
champion_data.drop(['image.full'],axis=1,inplace=True)
champion_data.drop(['image.sprite'],axis=1,inplace=True)
champion_data.drop(['image.group'],axis=1,inplace=True)
champion_data.drop(['image.h'],axis=1,inplace=True)
champion_data.drop(['image.w'],axis=1,inplace=True)

champion_data.to_csv(r"C:\Users\Lenovo\Documents\ML\riot_champion.csv")

h1 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\912-20-21-04.csv")
h2 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\812-20-21-01.csv")
h3 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\712-20-20-58.csv")
h4 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\612-20-20-56.csv")
h5 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\512-20-20-53.csv")
h6 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\412-20-20-51.csv")
h7 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\312-20-20-49.csv")
h8 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\212-20-20-47.csv")
h9 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\112-20-20-44.csv")
h10 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\012-20-20-41.csv")
h11 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\012-20-20-26.csv")

df=pd.concat([h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11],ignore_index=True).drop_duplicates()

#keys=["champions","kda_Ratio","results","mode","item1","item2","item3","item4","item5","item6","item7"] )

df['kda_Ratio'] = df['kda_Ratio'].str.replace(':1','')

df5['kda_Ratio'] = df5['kda_Ratio'].str.replace(':1','')

d1 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\012-23-19-46.csv")
d2 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\112-23-19-51.csv")
d3 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\212-23-19-53.csv")
d4 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\312-23-19-55.csv")
d5 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\412-23-19-57.csv")
d6 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\512-23-20-00.csv")
d7 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\612-23-20-02.csv")
d8 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\712-23-20-04.csv")
d9 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\812-23-20-07.csv")
d10 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\912-23-20-09.csv")
d11 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\012-23-20-11.csv")

df2=pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11],ignore_index=True).drop_duplicates()

df2['kda_Ratio'] = df2['kda_Ratio'].str.replace(':1','')

a1 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\212-25-18-32.csv")
a2 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\112-25-18-29.csv")
a3 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\012-25-18-27.csv")
a4 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\912-25-18-25.csv")
a5 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\812-25-18-23.csv")
a6 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\712-25-18-21.csv")
a7 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\612-25-18-19.csv")
a8 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\512-25-18-17.csv")
a9 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\412-25-18-15.csv")
a10 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\312-25-18-13.csv")
a11 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\012-25-18-07.csv")

df3=pd.concat([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11],ignore_index=True).drop_duplicates()

df3['kda_Ratio'] = df3['kda_Ratio'].str.replace(':1','')

df3.results = df3['results'].map({'Victory' : 1, 'Defeat' : 0}).astype(int)

df5.results = df5['results'].map({'Victory' : 1, 'Defeat' : 0}).astype(int)

e1 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\412-27-12-22.csv")
e2 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\312-27-12-20.csv")
e3 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\212-27-12-19.csv")
e4 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\112-27-12-17.csv")
e5 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\012-27-12-16.csv")
e6 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\912-27-12-14.csv")
e7 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\812-27-12-13.csv")
e8 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\712-27-12-12.csv")
e9 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\612-27-12-11.csv")
e10 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\512-27-12-10.csv")
e11 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\412-27-12-08.csv")

df4=pd.concat([e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11],ignore_index=True).drop_duplicates()

df4.results = df4['results'].map({'Victory' : 1, 'Defeat' : 0}).astype(int)

df4['kda_Ratio'] = df4['kda_Ratio'].str.replace(':1','')

df['kda_Ratio'] = df['kda_Ratio'].astype('float') # type casting

for order in [True, False]:
  champions = df.sort_values(by="kda_Ratio", ascending=order)["champions"][:5].values
  # conencter API avec le résultat intermédiaire products
  kda_ratio = df.sort_values(by="kda_Ratio", ascending=order)["kda_Ratio"][:2].values
  # optimize time complexity w space complexity

  for i in range(2):
    print(f'{champions[i]}: {kda_ratio[i]}')

df2.results = df2['results'].map({'Victory' : 1, 'Defeat' : 0}).astype(float)

df1=pd.concat([df,df2,df3,df4,df5],ignore_index=True,sort=False)

df1.to_csv(r"C:\Users\Lenovo\Documents\ML/league_dataset2.csv", index = False)
"""

"""### STARTING POINT"""

df1 = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\league_dataset2.csv")

champion_data = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\riot_champion.csv")

print(list(df1.columns))

champion_data.drop(['Unnamed: 0'], axis=1, inplace=True)

df1.drop(['w<champions'], axis=1, inplace=True)

print(df1['mode'].unique())

df1['mode'] = df1['mode'].str.replace('\tNormal', 'Normal')

print(df1['results'].unique())

df1.mode = df1['mode'].map(
    {'Flex 5:5 Rank': 1, 'Ranked Solo': 0, 'ARAM': 2, 'Normal': 3, 'Ultimates': 4, 'Tutorial 1': 5})

# map game modes
# drop control wards,oracle lens,stealth ward

print(df1['champions'].value_counts())

counts = df1['results'].value_counts()

victories = df1[df1["results"] == 1]

gg = df1[df1.kda_Ratio > '2.6']

# gg.drop(['results'], axis=1, inplace=True)


"""### recommendation system v2

# Multiple categorical columns
categorical_cols = ["item1","item2","item3","item4","item5","item6","item7"]

ranked_df = pd.get_dummies(ranked_solo, columns=categorical_cols, prefix='', drop_first=True)
"""


def EncodeLabel(data, feature, binary=True):
    if binary:
        lb = LabelBinarizer()
        temp = lb.fit_transform(data[feature])
        data[feature] = temp
    else:
        le = LabelEncoder()
        temp = le.fit_transform(data[feature])
        data[feature] = temp


"""### Recommender System: collaborative filtering

"""

item_desc = pd.read_csv(r"C:\Users\Lenovo\Documents\ML\riot_item.csv")

item_desc.drop(['upper_item', 'item_id', 'Unnamed: 0'], axis=1, inplace=True)

item_desc = item_desc.rename(columns={'name': 'item'})

"""## Item 1 """

item_1 = gg[['champions', 'kda_Ratio', 'item1']].copy()

item_1 = item_1.rename(columns={'item1': 'item'})

item_2 = gg[['champions', 'kda_Ratio', 'item2']].copy()
item_2 = item_2.rename(columns={'item2': 'item'})
item_3 = gg[['champions', 'kda_Ratio', 'item3']].copy()
item_3 = item_3.rename(columns={'item3': 'item'})
item_4 = gg[['champions', 'kda_Ratio', 'item4']].copy()
item_4 = item_4.rename(columns={'item4': 'item'})
item_5 = gg[['champions', 'kda_Ratio', 'item5']].copy()
item_5 = item_5.rename(columns={'item5': 'item'})
item_6 = gg[['champions', 'kda_Ratio', 'item6']].copy()
item_6 = item_6.rename(columns={'item6': 'item'})
item_7 = gg[['champions', 'kda_Ratio', 'item7']].copy()
item_7 = item_7.rename(columns={'item7': 'item'})
items_n = pd.concat([item_2, item_3, item_4, item_5, item_6, item_7], ignore_index=True)

item_1 = pd.merge(item_1, items_n, how="outer")

champion_data = champion_data.rename(columns={'id': 'champions'})

gg1 = item_1.merge(item_desc, on="item", how='left')

gg1 = gg1.merge(champion_data, on="champions", how='left')

gg1.drop(['info.attack', 'info.defense', 'info.magic', 'info.difficulty', 'stats.hp', 'stats.hpperlevel', 'stats.mp',
          'stats.mpperlevel', 'stats.movespeed', 'stats.armor', 'stats.armorperlevel', 'stats.spellblock',
          'stats.spellblockperlevel', 'stats.attackrange', 'stats.hpregen', 'stats.hpregenperlevel', 'stats.mpregen',
          'stats.mpregenperlevel', 'stats.crit', 'stats.critperlevel', 'stats.attackdamage',
          'stats.attackdamageperlevel', 'stats.attackspeedperlevel', 'stats.attackspeed'], axis=1, inplace=True)

"""### ITEM RECOMMENDER
Let us first try to build a recommender using item explaination and partype. We do not have a quantitative metric to judge our machine's performance so this will have to be done qualitatively.


"""

champion_item_df = gg1.pivot_table(index=["item"], columns=["champions"], aggfunc='count')

sorted_df = champion_item_df.sort_values("item", ascending=False)

sorted_df.to_csv(r"C:\Users\Lenovo\Documents\ML\sort_def.csv")

champ = str(input("Enter your champion: "))

print(champ)

sorted_df[sorted_df[champ]].head()

gg1['explain'] = gg1['explain'].astype(str)

# Import TfIdfVectorizer from scikit-learn

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
gg1['explain'] = gg1['explain'].fillna('')

from sklearn.metrics.pairwise import linear_kernel

tfidf_matrix = tfidf.fit_transform(gg1['explain'])

# Output the shape of tfidf_matrix

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix, True)

indices = pd.Series(gg1.index, index=gg1['item'])


def get_recommendations(item, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[item]

    # Get the pairwsie similarity scores of all items with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the items based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar items
    sim_scores = sim_scores[1:11]

    # Get the items indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar items
    return gg1['item'].iloc[movie_indices]


"""## ITEM REC V2"""

tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')

plt.spy(tfidf_matrix)

"""##TF-IDF"""

# Import CountVectorizer and create the count matrix

count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(gg1)

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

get_recommendations("Goredrinker", cosine_sim2)

