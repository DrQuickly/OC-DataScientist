import streamlit as st
import pickle
import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
#import streamlit.components.v1 as components

from PIL import Image

import json
from urllib.request import urlopen

#from sklearn.ensemble import RandomForestClassifier

from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(layout="wide")

#st.markdown(
#    f'''
#        <style>
#            .sidebar .sidebar-content {{
#                width: 200px;
#            }}
#        </style>
#    ''',
#    unsafe_allow_html=True
#)

def main():
    
    #cl1,cl2,cl3,cl4,cl5 = st.beta_columns(5)
    
    image = Image.open("logo.png")
    
    st.sidebar.image(image,use_column_width=True)
    
    st.markdown("<h1 style='text-align: center; color: black;'>Bienvenue sur votre portail de scoring client</h1>", unsafe_allow_html=True)

    #@st.cache
    def load_model(modelname):
        
        modelpath = 'models/'
        modelname = modelname.replace(" ","")
        
        if modelname == 'RandomForest':
            return pickle.load(open(modelpath+"random_forest_classifier_model.pkl",'rb'))
        elif modelname == 'XGBoost':
            return pickle.load(open(modelpath+"xgboost_classifier_model.pkl",'rb'))
        elif modelname == 'LogisticRegression':
            return pickle.load(open(modelpath+"logistic_regression_model.pkl",'rb'))
        else:
            return pickle.load(open(modelpath+"random_forest_classifier_model.pkl",'rb'))

    @st.cache
    def load_data(path,filename):
        data = pd.read_csv(path+filename)
        data.drop(['Unnamed: 0'],axis=1,inplace=True)
        list_id = data['SK_ID_CURR'].tolist()
        #data.set_index(['SK_ID_CURR'],inplace=True)
        
        return list_id,data
            
    def plot_features_importances(dataframe,modelname,n_features,width,height):
    
        model = load_model(modelname)
        
        cols = dataframe.drop(['SK_ID_CURR','TARGET'],axis=1).columns
        importances = model.best_estimator_.feature_importances_
        features_importances = pd.concat((pd.DataFrame(cols, columns = ['Variable']), 
                                         pd.DataFrame(importances, columns = ['Importance'])), 
                                        axis = 1).sort_values(by='Importance', ascending = False)
        
        fig,ax = plt.subplots(figsize=(width/96, height/96), dpi= 80, facecolor='w', edgecolor='k')

        colors = ['red' for x in range(n_features+1)]

        sns.barplot(x=features_importances['Importance'].head(n_features), 
                y=features_importances['Variable'].head(n_features),palette=colors,ax=ax)

        #plt.title("Importance des variables",fontsize=40)

        plt.xlabel("Importance",fontsize=30)
        plt.ylabel("Variable",fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)

        st.pyplot(fig)
        
        return features_importances.head(n_features)

    def write_client_info(ID_client,dataframe):
        
        def amount_formatter(amount):
    
            x = round(amount)
            x = "{:,.2f}".format(x)
            x = x.split(".")[0]
            x = x.replace(","," ")
            
            return x

        list_infos = ['SK_ID_CURR','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR',
                      'FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
                      'NAME_INCOME_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','DAYS_BIRTH','DAYS_EMPLOYED',
                      'CNT_FAM_MEMBERS','WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START']
        
        X = dataframe[list_infos]
        client_infos = X[X['SK_ID_CURR'] == int(ID_client)]
        
        st.text("")
        st.header("Informations descriptives du client:")
        st.text("")
        
        cls1,cls2 = st.beta_columns(2)
        
        age = int(np.abs(client_infos['DAYS_BIRTH'].tolist()[0])/365)
        genre = str(client_infos['CODE_GENDER'].tolist()[0])
        situation_familiale = str(client_infos['NAME_FAMILY_STATUS'].tolist()[0])
        nb_enfants = client_infos['CNT_CHILDREN'].tolist()[0]
        revenus_total = client_infos['AMT_INCOME_TOTAL'].tolist()[0]
        montant_credit = client_infos['AMT_CREDIT'].tolist()[0]
        cnt_fam_members = int(client_infos['CNT_FAM_MEMBERS'].tolist()[0])
        amt_annuity = client_infos['AMT_ANNUITY'].tolist()[0]
        name_contract_type = client_infos['NAME_CONTRACT_TYPE'].tolist()[0]
        
        with cls1:
            st.write("**IDentifiant: **"+str(ID_client))
            st.write("**Age: **"+str(age)+" ans")
            st.write("**Genre: **"+genre)
            st.write("**Situation familiale: **"+situation_familiale)
            st.write("**Nombre d'enfants: **"+str(nb_enfants))
            
        with cls2:
            st.write("**Composition de la famille: **"+str(cnt_fam_members))
            st.write("**Révenus total: **"+str(amount_formatter(revenus_total))+" $")
            st.write("**Montant du crédit: **"+str(amount_formatter(montant_credit))+" $")
            st.write("**Annuité: **"+str(amount_formatter(amt_annuity))+" $")
            st.write("**Type de prêt: **"+str(name_contract_type))
        
        return age,genre,situation_familiale,nb_enfants,revenus_total, montant_credit
    
    #def print_prediction_interpretation():
        
    def get_local_interpretation(ID_client,dataframe,modelname,features_importances,label):
        
        model = load_model(modelname)
        X = dataframe[dataframe['SK_ID_CURR'] == int(ID_client)]
        X = X.drop(['SK_ID_CURR', 'TARGET'], axis=1)
        dataframe = dataframe.drop(['SK_ID_CURR', 'TARGET'], axis=1)
   
        X_train = dataframe.sample(frac=0.1,random_state=42).values
        
        explainer = LimeTabularExplainer(training_data = X_train,
                                         mode='classification',
                                         feature_names = dataframe.columns,
                                         training_labels = dataframe.columns.tolist(),
                                         verbose=1,
                                         random_state=42)
        #st.write(np.array(X))
        #st.write(type(np.array(X)))
        explanation = explainer.explain_instance(np.ravel(np.array(X)), 
                                                 predict_fn = model.predict_proba,
                                                 labels=[0,1],
                                                 num_features = len(dataframe.columns))
                
        #fig = explanation.as_pyplot_figure(label=label)
        #st.pyplot(fig)
        
        return explanation
    
    def pyplot_lime_explanation(explanation, features_importances,label):
    
        #features_importances = ['EXT_SOURCE_3','EXT_SOURCE_2','AMT_REQ_CREDIT_BUREAU_YEAR','WEEKDAY_APPR_PROCESS_START',
        #                        'DAYS_LAST_PHONE_CHANGE','AMT_GOOD_PRICE','DAYS_ID_PUBLISH','REGION_POPULATION_RELATIVE',
        #                        'AMT_CREDIT','AMT_INCOME_TOTAL','DAYS_BIRTH','OCCUPATION_TYPE','HOUR_APPR_PROCESS_START',
        #                        'AMT_ANNUITY','DAYS_REGISTRATION']
        
        features_list = features_importances["Variable"].values
        
        exp = explanation.as_list(label=label)
        
        fig = plt.figure()
        vals = []
        names = []
        
        import re
        
        for x in exp:
            i = re.split('<|>|=',x[0])
        
            try:
                float(i[0].strip(" "))
                i = i[1].strip(" ")
            except:
                i = i[0].strip(" ")
        
            if i in features_list:
                vals.append(x[1])
                names.append(x[0])
      
        vals.reverse()
        names.reverse()
        
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(vals)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        
        #plt.xticks(fontsize=35)
        plt.yticks(pos, names)
        plt.xlabel("Importance")
        plt.ylabel("Variable")

        st.pyplot(fig)
        
        return True

    def gauge_plot(arrow_index, labels):
    
        list_colors = np.linspace(0,1,int(len(labels)/2))
        size_of_groups=np.ones(len(labels))
        
        white_half = np.ones(len(list_colors))*.5
        color_half = list_colors
    
        cs1 = cm.RdYlGn_r(color_half)
        cs2 = cm.seismic(white_half)
        cs = np.concatenate([cs1,cs2])
    
        fig, ax = plt.subplots()
        
        ax.pie(size_of_groups, colors=cs, labels=labels)
    
        my_circle=plt.Circle( (0,0), 0.6, color='white')
        ax.add_artist(my_circle)
    
        arrow_angle = (arrow_index/float(len(list_colors)))*3.14159
        arrow_x = 0.8*math.cos(arrow_angle)
        arrow_y = 0.8*math.sin(arrow_angle)
        arr = plt.arrow(0,0,-arrow_x,arrow_y, width=.02, head_width=.05, \
            head_length=.1, fc='k', ec='k')
    
        ax.add_artist(arr)
        ax.add_artist(plt.Circle((0, 0), radius=0.04, facecolor='k'))
        ax.add_artist(plt.Circle((0, 0), radius=0.03, facecolor='w', zorder=11))
            
        ax.set_aspect('equal')
        
        st.pyplot(fig)
        
        return True
    
    #@st.cache
    def get_client_score(ID_client,prediction,proba):
        
        T = pd.DataFrame(columns=["A","B","Note"])
        
        j = 0
        for i in range(1,21):
            T.loc[i-1,"A"] = j
            T.loc[i-1,"B"] = j + 0.05
            T.loc[i-1,"Note"] = i
            j = j + 0.05
        
        if prediction == 1:
            proba = 1 - proba
        
        prob_data = pd.DataFrame(columns=["Proba","Note","arrow_idx"])
        
        X_val = T[(proba >=T["A"]) & (proba <T["B"])]
        
        prob_data = prob_data.append(pd.DataFrame(
                {'Proba' : [proba],'Note' : [X_val["Note"].values[0]],
                 'arrow_idx' : [X_val["Note"].values[0] * 50]}),
                ignore_index=True)
             
        prob_data['SK_ID_CURR'] = int(ID_client)
        prob_data['TARGET'] = prediction
    
        return prob_data
        
    def plot_client_stats(ID_client,dataframe,filters):
            
        cols = ["SK_ID_CURR","Age","CODE_GENDER","CNT_CHILDREN","AMT_INCOME_TOTAL",
                "AMT_CREDIT","NAME_FAMILY_STATUS"]
        
        data = dataframe[cols]
           
        X = data[data['SK_ID_CURR'] == int(ID_client)]
        new_data = data[data['SK_ID_CURR'] != int(ID_client)]
         
        if filters == "Age":
            new_data = new_data[new_data['Age'] == X['Age'].values[0]]
            
        if filters == "Sexe":
            new_data = new_data[new_data['CODE_GENDER'] == X['CODE_GENDER'].values[0]]
           
        fam_status = new_data[new_data['NAME_FAMILY_STATUS'] == X['NAME_FAMILY_STATUS'].values[0]]
        
        columns=["Client","Médiane/dataset","Médiane/Status"]
        
        amt_income_total = pd.concat((pd.DataFrame(columns, columns = ['labels']), 
                              pd.DataFrame([X['AMT_INCOME_TOTAL'].values[0],
                                            new_data['AMT_INCOME_TOTAL'].median(),
                                            fam_status['AMT_INCOME_TOTAL'].median(),],
                                           columns = ['Values'])),axis = 1)
        
        fig = plt.figure(figsize=(7, 2), dpi= 80, facecolor='w', edgecolor='k')
        
        ax1 = plt.subplot(1,3,1)
        sns.barplot(x="labels", y="Values", data=amt_income_total,ax=ax1)
        plt.xticks(rotation=90)
        plt.title("Revenu total")
        
        amt_credit = pd.concat((pd.DataFrame(columns, columns = ['labels']), 
                              pd.DataFrame([X['AMT_CREDIT'].values[0],
                                            new_data['AMT_CREDIT'].median(),
                                            fam_status['AMT_CREDIT'].median(),],
                                           columns = ['Values'])),axis = 1)
        ax2 = plt.subplot(1,3,2)
        sns.barplot(x="labels", y="Values", data=amt_credit,ax=ax2)
        plt.xticks(rotation=90)
        plt.title("Montant du crédit")
        
        cnt_children = pd.concat((pd.DataFrame(columns, columns = ['labels']), 
                              pd.DataFrame([X['CNT_CHILDREN'].values[0],
                                            new_data['CNT_CHILDREN'].median(),
                                            fam_status['CNT_CHILDREN'].median(),],
                                           columns = ['Values'])),axis = 1)
        
        ax3 = plt.subplot(1,3,3)
        sns.barplot(x="labels", y="Values", data=cnt_children,ax=ax3)
        plt.xticks(rotation=90)
        plt.title("Nombre d'enfants")
        
        plt.subplots_adjust(wspace=0.9)
        
        st.pyplot(fig)
    
        return True

    #### Code principal ####
    
    path_read = 'data/'
    list_id,df = load_data(path_read, "data.csv")
    list_id2,dat = load_data(path_read, "app_data.csv")
    
    df2 = dat.copy()
    
    st.text("")
    st.text("")            
            
    id_client = st.sidebar.text_input("Entrez l'identifiant d'un client:",)
    
    if id_client == '':
        st.write("S'il vous plait entrez un identifiant correct.")
    elif int(id_client) in list_id:
        
        age,genre,situation_familiale,nb_enfants,revenus_total, montant_credit = write_client_info(int(id_client), df2)

        width = 1440
        height = 948
        n_features = 15
                
        modelname = 'Random Forest'
        
        url = "http://localhost:5000/client/" + id_client +"/"+ modelname.replace(" ","")

        with st.spinner('Chargement du score du client...'):
            json_url = urlopen(url)
        
            prediction_data = json.loads(json_url.read())
         
            score_list = get_client_score(id_client,prediction_data["prediction"],prediction_data["proba"])
            X = score_list[score_list['SK_ID_CURR'] == int(id_client)]
    
            df2['Age'] = (df2['DAYS_BIRTH'].abs()/365).astype("int32")
            
            new_data = df2[df2['SK_ID_CURR'] == int(id_client)]
            
            st.text("")
            st.header("Prédiction:")
            st.text("")
            
            cols1,cols2 = st.beta_columns(2)
            with cols1:     
                clt_score = X['Note'].values[0]
                if clt_score > 14:
                    st.markdown(f"Ce client a une note de **{clt_score:02d}/20** pour rembourser son crédit. Le risque de défaut de paiement de ce client est **faible**.")
                else:
                    st.markdown(f"Ce client a une note de **{clt_score:02d}/20** pour rembourser son crédit. Le risque de défaut de paiement de ce client est **élevé**.")
                
                if not new_data['TARGET'].isnull().values.any():
                    if int(new_data['TARGET'] == 1):
                        st.markdown("Pour rappel, ce client a été en défaut de paiement auparavant.")
                    else:
                        st.markdown("Pour rappel, ce client n'a pas été en défaut de paiement auparavant.")
                        
                st.text("")
                st.text("")
                st.markdown("**NB:** Le seuil de **14/20** a été défini pour évaluer le niveau du risque de défaut de paiement d'un client: pour une note **inférieure à 14/20**, le risque est **élevé** et pour une note **supérieure à 14/20** le risque est **faible**. Plus la note se rapproche de **20/20** plus le risque est faible et plus la note est faible plus le client est risqué.")
                
            with cols2:
                values = np.linspace(0,1,1000)
                labels = [' ']*len(values)*2
                labels[25] = '20'
                labels[250] = '15'
                labels[500] = '10'
                labels[750] = '5'
                labels[975] = '0'
                
                arrow_index = X['arrow_idx'].values[0]
                gauge_plot(arrow_index,labels=labels)
                
        col1,col2 = st.beta_columns(2)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Explication globale de la prédiction</h3>", unsafe_allow_html=True)
            features_importances = plot_features_importances(df, modelname, n_features,width,height)
        
        labels = prediction_data['prediction']
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Explication locale de la prédiction</h3>", unsafe_allow_html=True)
            exp = get_local_interpretation(id_client,df,modelname,features_importances,labels)
            pyplot_lime_explanation(exp, features_importances,labels)
        
        select = st.sidebar.selectbox("Afficher la comparaison avec d'autres clients:", 
                                      ["All datasets","Age","Sexe"],key='1')
        
        if not st.sidebar.checkbox("Hide", True, key='1'):
            if select == "All datasets":
                filters = "All"
                st.header(f"Comparaison du client **{id_client}** aux autres clients du jeu de données:")
                
            if select == "Age":
                filters = "Age"
                st.header(f"Comparaison du client **{id_client}** aux autres clients du même age:")

            if select == "Sexe":
                filters = "Sexe"
                st.header(f"Comparaison du client **{id_client}** aux autres clients du même sexe:")
        
            plot_client_stats(id_client,df2,filters)
                
        #age_values = np.abs(df2['DAYS_BIRTH'].tolist())/365
        #age_values = [ int(i) for i in age_values]
        #st_age = st.sidebar.slider("Age (ans)",min(age_values),max(age_values),value=(20,30))
        #st.write('Range values:', st_age)
        #selection_list = ['DAYS_BIRTH','CODE_GENDER','CNT_CHILDREN','AMT_CREDIT','AMT_INCOME_TOTAL']
        #selections = st.sidebar.multiselect("Sélection",selection_list,default="DAYS_BIRTH")
        
        
        #st.dataframe(df2[:100])
       
    else:
        st.warning("Identifiant incorrect. Veuillez saisir un identifiant correct")

    
if __name__ == '__main__':
    main()