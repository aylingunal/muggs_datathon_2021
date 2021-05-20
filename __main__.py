import seaborn as sns
import matplotlib.pyplot as mpl
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import gensim
from gensim.corpora.dictionary import Dictionary
from sklearn.decomposition import LatentDirichletAllocation as LDA
import nltk.corpus
import collections as col
import math


def init_EDA():
    # exploration of variable relationships 
    # visulizations
    df = pd.read_csv("AirBnBData_pluscategorytags.csv")
   # mpl.scatter(df["price"],df["latitude"])
   # mpl.xlabel("Price($)")
   # mpl.ylabel("Latitude")
   # mpl.show()

   # sns.boxplot(x=df["proportion_relativelocation_terms"],y=df["number_of_reviews"],
   #                 hue=df["neighbourhood_group"], showfliers=False)
   # mpl.show()
    colors = {'Brooklyn':'red','Manhattan':'blue','Queens':'yellow','Bronx':'green','Staten Island':'orange'}
    mpl.scatter(df["proportion_relativelocation_terms"],
               df["number_of_reviews"], c=df["neighbourhood_group"].apply(lambda x:colors[x]))
    mpl.show()
  #  sns.boxplot(x=df["neighbourhood_group"],y=df["price"],
  #                  hue=df["proportion_relativelocation_terms"], showfliers=False)
  #  mpl.show()

  #  sns.boxplot(x=df["neighbourhood_group"],y=df["price"],
  #                  hue=df["proportion_unitdescriptor_terms"], showfliers=False)
  #  mpl.show()

  #  sns.boxplot(x=df["neighbourhood_group"],y=df["price"],
  #                  hue=df["proportion_unittype_terms"], showfliers=False)
  #  mpl.show()

  #  mpl.plot(df['price'])
  #  mpl.show()

    # correlations TODO

def word_EDA(df):
    # get all unique words across listings + their frequencies 
    uniq_terms = col.Counter()
    df['name'].apply(uniq_terms.update)
    print(uniq_terms)




def text_processing():

    df = pd.read_csv("AB_NYC_2019.csv")

    df["name"] = df["name"].apply(lambda x: str(x))
    df["name"] = df["name"].apply(lambda x: x.split(' '))
    df["name"] = df["name"].apply(lambda x: [re.sub('[,\.!-?]', ' ', w) for w in x])
    df["name"] = df["name"].apply(lambda x: [w.lower() for w in x])

    stemmer = PorterStemmer()
    stop_words = stopwords.words('english')
    stop_words.append('room')

    # tokenize
    df["name"] = df["name"].apply(lambda x: [word_tokenize(w) for w in x])
    df["name"] = df["name"].apply(lambda x: [i for j in x for i in j])
    # remove stop words
    df["name"] = df["name"].apply(lambda x: [w for w in x if w not in stop_words])
    # stem
    df["name"] = df["name"].apply(lambda x: [stemmer.stem(w) for w in x])

    return df

def apply_LDA(df):
    listing_names = df["name"]
    common_dictionary = Dictionary(listing_names)
    common_corpus = [common_dictionary.doc2bow(text) for text in listing_names]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=common_corpus,
                                                id2word=common_dictionary,
                                                num_topics=10,
                                                update_every=1,
                                                chunksize=len(common_corpus),
                                                passes=20,
                                                alpha='auto',
                                                random_state=42)
    import pyLDAvis.gensim
    lda_display = pyLDAvis.gensim.prepare(lda_model,common_corpus,
                                          common_dictionary,sort_topics=True)
    pyLDAvis.show(lda_display)

    for i in range(0, 10):
        topic_col = []
        for j in common_corpus:
            tmp_dict = dict(lda_model[j])
            if i in tmp_dict.keys():
                topic_col.append(tmp_dict[i])
            else:
                topic_col.append(0)
        df["Topic_"+str(i)] = topic_col
    
    df.to_csv("AirBnBData_plustopics.csv", index=False)

def category_tags(df):
    # category 1 = unit type
    cat1 = ['bedroom','privat','apart','apt','studio',
            'br','bed','loft','home','bath','duplex','hous',
            'bathroom','floor','bd','entir','share','kitchen',
            'bdrm','townhous','balconi','backyard','build','gym',
            'doorman','rooftop','terrac','penthous','condo','rm',
            'flat','patio','deck']
    # category 2 = relative location (should add all unique neighborhoods)
    cat2 = ['brooklyn','manhattan','williamsburg','near','villag','nyc',
            'central','locat','heart','side','upper','midtown','brownston',
            'harlem','squar','subway','close','train','queen','citi','chelsea',
            'soho','york','astoria','greenpoint','place','ny','away','columbia',
            'downtown','street','hell','neighborhood','south','east','west',
            'north','greenwich','ave','clinton','airport','min','jfk','minut','center']
    # category 3 = descriptors
    cat3 = ['spacious','sunni','beauti','larg','cozi','luxuri','modern','bright',
            'new','charm','quiet','clean','view','huge','prime','amaz','comfort',
            'love','renov','space','big','artist','best','comfi','gorgeou',
            'histor','cute','oasi','nice','light','chic','stylish','conveni','furnish',
            'newli','gem']
    
    listings = df['name'].tolist()
    cat1_col = []
    cat2_col = []
    cat3_col = []

    for i in listings:
        cat1_tmp = [x for x in i if x in cat1]
        cat2_tmp = [x for x in i if x in cat2]
        cat3_tmp = [x for x in i if x in cat3]

        print(str(len(cat1_tmp))+" " +str(len(i)))
        print(str(len(cat2_tmp))+" " +str(len(i)))
        print(str(len(cat3_tmp))+" " +str(len(i)))

        if len(i) == 0:
            cat1_col.append(0)
            cat2_col.append(0)
            cat3_col.append(0)
        else:
            cat1_col.append(len(cat1_tmp)/len(i))
            cat2_col.append(len(cat2_tmp)/len(i))
            cat3_col.append(len(cat3_tmp)/len(i))  

    df["proportion_unittype_terms"] = cat1_col
    df["proportion_relativelocation_terms"] = cat2_col
    df["proportion_unitdescriptor_terms"] = cat3_col

    df.to_csv("AirBnBData_pluscategorytags.csv",index=False)

    return df

def adv_EDA(df):

    mpl.scatter(df["price"],df["proportion_unittype_terms"])
    mpl.xlabel("Price($)")
    mpl.ylabel("proportion of unit type terms")
    mpl.show()

    mpl.scatter(df["price"],df["proportion_relativelocation_terms"] )  
    mpl.xlabel("Price($)")
    mpl.ylabel("proportion of relative location terms")
    mpl.show()

    mpl.scatter(df["price"],df["proportion_unitdescriptor_terms"])
    mpl.xlabel("Price($)")
    mpl.ylabel("proportion of unit descriptor terms")
    mpl.show()

    mpl.scatter(df["number_of_reviews"],df["proportion_unittype_terms"])
    mpl.xlabel("num reviews")
    mpl.ylabel("proportion of unit type terms")
    mpl.show()

    mpl.scatter(df["number_of_reviews"],df["proportion_relativelocation_terms"])
    mpl.xlabel("num reviews")
    mpl.ylabel("proportion of relative location terms")
    mpl.show()

    mpl.scatter(df["number_of_reviews"],df["proportion_unitdescriptor_terms"])
    mpl.xlabel("num reviews")
    mpl.ylabel("proportion of unit desciptor terms")
    mpl.show()



def main():
    df = text_processing()
   # df_upd = category_tags(df)
   # df = pd.read_csv("AirBnBData_pluscategorytags.csv")
   # init_EDA()
  # adv_EDA(df)
  #  word_EDA(df)
    apply_LDA(df)



if __name__ == '__main__':
    main()