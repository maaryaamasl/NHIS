import pandas as pd

variable_list_df = pd.read_excel('NHIS variable list_Modified.xlsx')
selected_columns = variable_list_df['variable(s)'].tolist()

loaded_column_names = pd.read_csv('./shap/columns.csv')['Column Names'].values
print(loaded_column_names )


arr_shape = np.loadtxt("Results/shape.csv")
shap_values = [ [] for i in range(int(arr_shape))]
for i in range(int(arr_shape)):
    shap_values[i] = np.loadtxt("Results/shap_"+str(i)+".csv")

print('\nD1 Classes:',len(shap_values),'\nD2 samples:', len(shap_values[0]),'\nD3 Columns/features:',len(shap_values[0][0]),'\nvalue:',shap_values[0][0][0])
print('type: ',type(shap_values))
print('type [0]: ', type(shap_values[0]))

exit(-1)
if True:

        # shap.summary_plot(shap_values, glob_x_train, feature_names=final_features, plot_type="bar")
        overall_classes = np.array( [ 0 for _ in range(len(shap_values[0][0]))  ] )
        overall_classes = np.array( [overall_classes  for _ in range(len(shap_values[0]))   ] )
        for x in range(len(shap_values)) :
            overall_classes = np.add(overall_classes,shap_values[x] )  # ToDO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        overall_classes = np.divide(overall_classes, len(shap_values)) # ToDO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        print('overall_classes.shape', overall_classes.shape)

        overall_variables = np.array([0 for _ in range(len(shap_values[0][0]))])
        for x in range(overall_classes.shape[0]) :
            overall_variables = np.add(overall_variables,overall_classes[x] )
        overall_variables = np.divide(overall_variables, overall_classes.shape[0])
        print('overall_variables.shape',overall_variables.shape)

        df = pd.DataFrame(overall_variables, columns=['values'],index=final_features)
        print(df.head(),"\n",str(list(df.index)))



        ### Ploting
        print("\nPloting")
        df[['label','color']] = np.nan
        df = df.sort_values(by='values', ascending=False, )
        print(df.head(2),"\n")
        env_disc = pd.read_csv(dataLocation + "env_disc" + ".csv")
        env_disc['Variable Name'] = env_disc['Variable Name'].str.lower()
        print(env_disc.head(2),"\n")

        palette = sns.color_palette("bright", 10) # pastel

        no_desc = 0
        # df = df.drop('occupation_2.58509567', 0)
        # geo_risk_dict = {'fakecensustractid':'Fake census tract ID'}
        standard_risk_dict = {


                             'sbp': 'Systolic blood pressure', 'dbp': 'Diastolic blood pressure', 'abi': 'Ankle brachial index',
                             'bpmeds': 'Blood pressure medication status',
                             'totchol': 'Total cholesterol', 'ldl': 'LDL cholesterol', 'hdl': 'HDL cholesterol ',
                             'trigs': 'Triglycerides ', 'fpg': 'Fasting glucose',
                             'hba1c': 'Hemoglobin A1C', 'alc': 'Alcohol (drinking in past 12 months)',
                             'alcw': 'Alcohol (average drinks per week)',
                             'currentsmoker': 'Current cigarette smoker', 'eversmoker': 'History of cigarette smoking',
                             'pa3cat': 'AHA physical activity category', 'activeindex': 'Physical activity during leisure time',
                             'nutrition3cat': 'AHA nutrition categorization',

                             'sex': 'Sex - Male', 'age': 'Age', 'waist': 'Waist', 'bmi': 'BMI',  # Demographic

             # 'sbp':'Systolic blood pressure', 'dbp':'Diastolic blood pressure', 'abi':'Ankle brachial index', 'bpmeds':'Blood pressure medication status',
             #                     'totchol':'Total cholesterol','ldl':'LDL cholesterol','hdl':'HDL cholesterol ', 'trigs':'Triglycerides ', 'fpg':'Fasting glucose',
             #                     'hba1c':'Hemoglobin A1C', 'alc':'Alcohol drinking in the past 12 months','alcw':'Average number of drinks per week',
             #                     'currentsmoker':'Cigarette Smoking Status','eversmoker':'History of Cigarette Smoking',
             #                     'pa3cat': 'Physical Activity', 'activeindex':'Physical activity during leisure time','nutrition3cat':'Nutrition Categorization',
             #
             #                     'sex': 'Sex - Male', 'age': 'Age','waist':'Waist','bmi':'BMI', #  Demographic

         }
        # demo + insurance + psycho
        scio_risk_dict = { # Psychosocial
                          # .apply(lambda x: "HSgrad" if x > 0 else "less_HSgrad") # .apply(lambda x: "employed" if x <= 7 else "not_employed")

                          # 'occupation_1.0': 'occupation - Management/Professional', 'occupation_2.0': 'occupation - Service', 'occupation_3.0': 'occupation - Sales',
                          # 'occupation_4.0': 'occupation - Farming', 'occupation_5.0': 'occupation - Construction',
                          # 'occupation_6.0': 'occupation - Production', 'occupation_7.0': 'occupation - Military', 'occupation_8.0': 'occupation - Sick',
                          # 'occupation_9.0': 'occupation - Unemployed', 'occupation_10.0': 'occupation - Homemaker', 'occupation_11.0': 'occupation - Retired',
                          # 'occupation_12.0': 'occupation - Student', 'occupation_13.0': 'occupation - Other',

                          # 'edu3cat_0':'Education - Less thank high school','edu3cat_1':'Education - LHigh school graduate/GED','edu3cat_2':'Education - attended vocational or colledge',

            'dailydiscr': 'Daily discrimination', 'lifetimediscrm': 'Lifetime discrimination',
            'discrmburden': 'Discrimination burden',

            'depression': 'Depressive symptoms', 'weeklystress': 'Weekly stress', 'perceivedstress': 'Global stress',

            # 'dailydiscr':'Daily discrimination','lifetimediscrm':'Lifetime discrimination','discrmburden':'Discrimination burden',
                          #
                          #
                          # 'depression': 'Depressive Symptoms Score', 'weeklystress': 'Weekly stress score', 'perceivedstress': 'Global Stress Score',
                          }
        socioeconomic = {

                        'insured': 'Insured',  # insurance
                        'privatepublicins_0.0': 'Insurance status - Uninsured',
                        'privatepublicins_1.0': 'Insurance status - Private Only',
                        'privatepublicins_2.0': 'Insurance status - Public Only',
                        'privatepublicins_3.0': 'Insurance status - Private & Public',
                        'fmlyinc': 'Family income',
                        'occupation_employed': 'Currently employed)',
                        'occupation_not_employed': 'Not employed)',
                        'edu3cat_HSgrad': 'High school graduate',
                        'edu3cat_less_HSgrad': 'Did not complete high school',

                        # 'insured':'Insured', # insurance
                        #   'privatepublicins_0.0': 'Insurance status - Uninsured', 'privatepublicins_1.0': 'Insurance status - Private Only',
                        #   'privatepublicins_2.0': 'Insurance status - Public Only', 'privatepublicins_3.0': 'Insurance status - Private & Public',
                        #  'fmlyinc': 'Family income',
                        #  'occupation_employed': 'Employment Status (Employed)',
                        #  'occupation_not_employed': 'Employment Status (Not Employed)',
                        #  'edu3cat_HSgrad': 'Education - High school graduated',
                        #  'edu3cat_less_HSgrad': 'Education - Less than high school Graduate',
        }
        for x in df.index:

            if x in env_disc['Variable Name'].values:
                # print(env_disc.loc[env_disc['Variable Name']==x] )
                df.loc[x,'label']= str(env_disc.loc[env_disc['Variable Name']==x,'Label'].values[0]).split('\n')[0] #+ ' ('+str(x)+')'
                df.loc[x, 'color'] = 2
                df.loc[x, 'color_label'] = 'Environmental'
            elif x.split('_')[0] in env_disc['Variable Name'].values:
                print('First part exist: ',x)
                # print(env_disc.loc[env_disc['Variable Name'] == x.split('_')[0]])
                df.loc[x, 'label'] = str(env_disc.loc[env_disc['Variable Name']==x.split('_')[0],'Label'].values[0]).split('\n')[0] #+ ' ('+str(x)+')'
                df.loc[x, 'color'] = 2
                df.loc[x, 'color_label'] = 'Environmental'
            # elif x in geo_risk_dict.keys():
            #     df.loc[x, 'label'] = geo_risk_dict[x]+' ('+str(x)+')'
            #     df.loc[x, 'color'] = 2
            #     df.loc[x, 'color_label'] = 'Environmental'
            elif x in standard_risk_dict.keys():
                df.loc[x, 'label'] = standard_risk_dict[x]#+' ('+str(x)+')'
                df.loc[x, 'color'] = 3
                df.loc[x, 'color_label'] = 'Standard'
            elif x in scio_risk_dict.keys():
                df.loc[x, 'label'] = scio_risk_dict[x]#+' ('+str(x)+')'
                df.loc[x, 'color'] = 0
                df.loc[x, 'color_label'] = 'Psychosocial'
            elif x in socioeconomic.keys():
                df.loc[x, 'label'] = socioeconomic[x]#+' ('+str(x)+')'
                df.loc[x, 'color'] = 5
                df.loc[x, 'color_label'] = 'Socioeconomic'
            else:
                no_desc+=1
                print('no_desc: '+str(no_desc)+" "+x)


        df['label'] = df['label'].str.replace("  "," ").replace("\n"," ").replace("\t"," ")
        # print(df.dropna().head())
        # exit()

        palette = {"Environmental": palette[2], "Standard": palette[3], "Psychosocial": palette[0], "Socioeconomic":palette[9]} # 7 grey 5 dark red
        hue_order = [ "Standard", "Environmental", "Socioeconomic", "Psychosocial"]
        # ### Positive
        # my_dpi = 200
        # df_filtered = df[df['values']>=0].sort_values(by='values', ascending=False).reset_index(drop=True).copy()
        # for i in range(0,df_filtered.shape[0],51):
        #     partial_df = df_filtered.iloc[i:i+50].copy()
        #     print("partial_df: ", partial_df.shape)
        #     plt.figure(figsize=(400 / my_dpi, (1400 / my_dpi)*((partial_df.shape[0]+10)/(51+10))   ), dpi=my_dpi)
        #     # sns.set(style="ticks")
        #     sns.set_style("darkgrid",{"axes.facecolor": ".9"})
        #     # sns.set_context("paper")
        #     sns.barplot(y='label', x="values", data=partial_df,hue='color_label',dodge=False,palette=palette,hue_order=hue_order)#,palette=[palette[i] for i in partial_df.color.astype(int)])
        #     # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
        #     # plt.tight_layout()
        #     plt.legend(title="Predictors",loc='lower right')
        #     plt.xlabel('Mean SHAP (average impact on model output magnitude)',Fontsize=12 )
        #     plt.ylabel('Variables',Fontsize=12)
        #     # plt.show()
        #     plt.xlim(df_filtered['values'].min() - 0.0001, df_filtered['values'].max() * 1.02)
        #     plt.grid()
        #     plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
        #     plt.savefig(dataLocation+"Figs/"+"Positive-"+str(i),  bbox_inches="tight",pad_inches=0.3) # facecolor='y', , transparent=True, dpi=200
        #     plt.clf()
        #     # exit()
        #
        # ### Negative
        # plt.clf()
        # plt.close()
        # df_filtered = df[df['values'] < 0].sort_values(by='values', ascending=False).reset_index(drop=True).copy()
        # for i in range(0, df_filtered.shape[0], 51):
        #     partial_df = df_filtered.iloc[i:i + 50].copy()
        #     print("partial_df: ", partial_df.shape)
        #     plt.figure(figsize=(400 / my_dpi, (1400 / my_dpi)*((partial_df.shape[0]+10)/(51+10))   ), dpi=my_dpi)
        #     # sns.set(style="ticks")
        #     sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        #     # sns.set_context("paper")
        #     sns.barplot(y='label', x="values", data=partial_df, hue='color_label',
        #                 dodge=False,palette=palette,hue_order=hue_order)  # ,palette=[palette[i] for i in partial_df.color.astype(int)])
        #     # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
        #     # plt.tight_layout()
        #     plt.legend(title="Predictors", loc='upper left')
        #     plt.xlabel('Mean SHAP (average impact on model output magnitude)', Fontsize=12)
        #     plt.ylabel('Variables', Fontsize=12)
        #     # plt.show()
        #     plt.xlim(df_filtered['values'].min()* 1.02 , df_filtered['values'].max() + 0.0002)
        #     plt.grid()
        #     plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
        #     plt.savefig(dataLocation + "Figs/" + "Negative-" + str(i), bbox_inches="tight",
        #                 pad_inches=0.3)  # facecolor='y', , transparent=True, dpi=200
        #     plt.clf()
        #     # exit()


        my_dpi = 200
        ### abs
        plt.clf()
        plt.close()
        # TODO: Result_shap_abs_sorted # 123
        df_filtered = df.copy()
        df_filtered['index_df'] = df.index
        df_filtered['values'] = df_filtered['values'].abs()
        df_filtered = df_filtered.sort_values(by='values', ascending=False).reset_index(drop=True)
        df_filtered['label']= df_filtered['label'].apply(lambda x: x.capitalize())
        df_filtered['label'] = df_filtered['label'].apply(lambda x: x.replace("Ldl","LDL").replace(" a1c "," A1C ")
                                                          .replace("+instructional+w","+w").replace("(not employed)","- not employed")
                                                          .replace("Hdl", "HDL")
                                                          )
        print(df_filtered.head())
        for i in range(0, df_filtered.shape[0], 51):

            import matplotlib as mpl
            mpl.rcParams['font.family'] = 'Arial'
            sns.set(font="Arial")

            partial_df = df_filtered.iloc[i:i + 50].copy()
            print("partial_df: ", partial_df.shape)
            plt.figure(figsize=(600 / my_dpi, (2000 / my_dpi)*((partial_df.shape[0]+10)/(51+10))  ), dpi=my_dpi)
            # sns.set(style="ticks")
            sns.set_style("darkgrid", {"axes.facecolor": ".9"})
            # sns.set_context("paper")
            ax = sns.barplot(y='label', x="values", data=partial_df, hue='color_label',
                        dodge=False,hue_order=hue_order,palette=palette)  # ,palette=[palette[i] for i in partial_df.color.astype(int)])
            # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
            # plt.tight_layout()
            plt.legend(title="Predictors", loc='lower right', prop={'size': 23})
            plt.xlabel('Mean |SHAP| (average impact on model output magnitude)', Fontsize=12)
            plt.ylabel('Variables', Fontsize=12, rotation=0)
            # plt.show()
            plt.xlim(df_filtered['values'].min() - 0.0001, df_filtered['values'].max() * 1.02)
            plt.grid()

            import matplotlib as mpl
            mpl.rcParams['font.family'] = 'Arial'
            sns.set(font="Arial")
            legend = plt.legend( loc='lower right',prop={'size': 13})
            frame = legend.get_frame()
            frame.set_facecolor('white')
            # plt.axhline(y=14.5, color='r', linestyle='-')
            ax.yaxis.set_label_coords(-1.1,1.02)
            # ax.set_ylabel() # position=(x, y)
            # ax.tick_params(axis='y', rotation=90)

            print("write")
            plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
            plt.savefig(dataLocation + "Figs\\" + "Abs-" + str(i)+'.svg', bbox_inches="tight",
                        pad_inches=0.3, format='svg')  # facecolor='y', , transparent=True, dpi=200 , format='eps'
            # plt.savefig(dataLocation + "Figs/" + "Abs-" + str(i), bbox_inches="tight",
            #             pad_inches=0.3)
            plt.clf()
        categories = [ "Standard", "Environmental", "Socioeconomic", "Psychosocial"]
        for x in [ "Standard", "Environmental", "Socioeconomic", "Psychosocial"]:
            # df.loc[x,'label']= str(env_disc.loc[env_disc['Variable Name']==x,'Label']
            # df.loc[x, 'color_label'] = 'Socioeconomic'
            df_category = df_filtered[df_filtered['color_label']==x]
            for i in range(0, df_category.shape[0], 51):
                partial_df = df_category.iloc[i:i + 50].copy()
                partial_df_shape = partial_df.shape[0]
                if x in ["Socioeconomic"]:
                    partial_df_shape = int(partial_df_shape/2)
                if x in ["Psychosocial"]:
                    partial_df_shape = int(partial_df_shape / 4)
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                print("partial_df: ", partial_df.shape)
                print(len(palette),len(categories),categories.index(x))
                plt.figure(figsize=(600 / my_dpi, (2000 / my_dpi) * ((partial_df_shape + 10) / (51 + 10))),
                           dpi=my_dpi)
                # sns.set(style="ticks")
                sns.set_style("darkgrid", {"axes.facecolor": ".9"})
                # sns.set_context("paper")
                ax = sns.barplot(y='label', x="values", data=partial_df, hue='color_label',
                                 dodge=False, #hue_order=hue_order,
                                 palette=palette)  # [categories.index(x)+1] ,palette=[palette[i] for i in partial_df.color.astype(int)])
                # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
                # plt.tight_layout()
                plt.legend(title="Predictors", loc='lower right', prop={'size': 23})
                plt.xlabel('Mean |SHAP| (average impact on model output magnitude)', Fontsize=12)
                plt.ylabel('Variables', Fontsize=12, rotation=0)
                # plt.show()
                plt.xlim(df_filtered['values'].min() - 0.0001, df_filtered['values'].max() * 1.02)
                plt.grid()

                import matplotlib as mpl
                mpl.rcParams['font.family'] = 'Arial'
                sns.set(font="Arial")
                legend = plt.legend(loc='lower right', prop={'size': 13})
                frame = legend.get_frame()
                frame.set_facecolor('white')
                # plt.axhline(y=14.5, color='r', linestyle='-')
                ax.yaxis.set_label_coords(-1.1, 1.02)
                # ax.set_ylabel() # position=(x, y)
                # ax.tick_params(axis='y', rotation=90)

                print("write")
                plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
                plt.savefig(dataLocation + "Figs\\" + "cat-"+str(x)+"-" + str(i) + '.svg', bbox_inches="tight",
                            pad_inches=0.3, format='svg')
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                break

        # TODO: Result_shap_abs_sorted # 123
        xlsx = pd.read_excel(dataLocation +'_Analysis Data Dictionary.xlsx', sheet_name=None)
        df_filtered['Description'] = np.NAN
        df_filtered['Definition'] = np.NAN
        df_filtered['Formats'] = np.NAN
        for x in df_filtered.index_df:
            # df_filtered.loc[x,"index_name"] = str(x)
            print(str(x))
            y=x
            for prefix in ["occupation_","privatepublicins_","edu3cat_", ]:
                # 'privatepublicins_0.0': 'Insurance status - Uninsured',
                # 'privatepublicins_1.0': 'Insurance status - Private Only',
                # 'privatepublicins_2.0': 'Insurance status - Public Only',
                # 'privatepublicins_3.0': 'Insurance status - Private & Public',
                # 'fmlyinc': 'Family income',
                # 'occupation_employed': 'Currently employed)',
                # 'occupation_not_employed': 'Not employed)',
                # 'edu3cat_HSgrad': 'High school graduate',
                # 'edu3cat_less_HSgrad': 'Did not complete high school',
                if x.startswith(prefix):
                    y =x.split("_")[0]
                    print("   ",y)
            if x == "fmlyinc":
                y='Income'
            for sheet_name, sheet_df in xlsx.items():
                if ('Description' in sheet_df.columns) and (str(y).lower() in sheet_df['Name'].str.lower().values):
                    print("\tin Names - Description")
                    try:
                        z = sheet_df.loc[sheet_df['Name'].str.lower() == str(y).lower(),'Description' ].values
                        print("\t\tsheet_df Description: ", z )
                        df_filtered.loc[(df_filtered.index_df.str.lower()==x.lower()),'Description'] = z
                    except:
                        xyz =1
                if ('Definition' in sheet_df.columns) and (str(y).lower() in sheet_df['Name'].str.lower().values):
                    print("\tin Names - Definition")
                    try:
                        z = sheet_df.loc[sheet_df['Name'].str.lower() == str(y).lower(), 'Definition'].values
                        print("\t\tsheet_df Definition: ", z)
                        df_filtered.loc[(df_filtered.index_df.str.lower()==x.lower()),'Definition'] = z
                    except:
                        xyz =1
                if ('Formats' in sheet_df.columns) and (str(y).lower() in sheet_df['Name'].str.lower().values):
                    print("\tin Names - Formats")
                    try:
                        z = sheet_df.loc[sheet_df['Name'].str.lower() == str(y).lower(), 'Formats'].values
                        print("\t\tsheet_df Formats: ", z)
                        df_filtered.loc[(df_filtered.index_df.str.lower()==x.lower()),'Formats'] = z
                    except:
                        xyz =1
            # if x in env_disc['Variable Name'].values:
            #     # print(env_disc.loc[env_disc['Variable Name']==x] )
            #     df.loc[x,'label']= str(env_disc.loc[env_disc['Variable Name']==x,'Label'].values[0]).split('\n')[0] #+ ' ('+str(x)+')'
            #     df.loc[x, 'color'] = 2
            #     df.loc[x, 'color_label'] = 'Environmental'
        # df_filtered["index_name"]=df_filtered.index
        df_filtered.to_csv(dataLocation +"Figs/" + "Result_shap_abs_sorted" + ".csv", index=False) # index_label=True,
        print(df_filtered.head(5))
        print(df_filtered[df_filtered[['index_df','Description']].isna().any(axis=1)])
        exit()

            # exit()





        # ### Similarity
        # my_dpi = 100
        # df_filtered = df.sort_index(ascending=False).reset_index(drop=True).copy()
        # for i in range(0, df_filtered.shape[0], 51):
        #     partial_df = df_filtered.iloc[i:i + 50].copy()
        #     print("partial_df: ", partial_df.shape)
        #     plt.figure(figsize=(400 / my_dpi,  (1400 / my_dpi)*((partial_df.shape[0]+10)/(51+10))   ), dpi=my_dpi)
        #     # sns.set(style="ticks")
        #     sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        #     # sns.set_context("paper")
        #     sns.barplot(y='label', x="values", data=partial_df, hue='color_label',
        #                 dodge=False,hue_order=hue_order,palette=palette)  # ,palette=[palette[i] for i in partial_df.color.astype(int)])
        #     # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
        #     # plt.tight_layout()
        #     plt.legend(title="Predictors", loc='lower right')
        #     plt.xlabel('Mean SHAP (average impact on model output magnitude)', Fontsize=12)
        #     plt.ylabel('Variables', Fontsize=12)
        #     # plt.show()
        #     plt.xlim(df_filtered['values'].min() - 0.0001, df_filtered['values'].max() * 1.02)
        #     plt.grid()
        #     plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
        #     plt.savefig(dataLocation + "Figs/" + "Similarity-" + str(i), bbox_inches="tight",
        #                 pad_inches=0.3)  # facecolor='y', , transparent=True, dpi=200
        #     plt.clf()
        #     # exit()

        # shap.summary_plot(shap_values[target_shap_class], shap_testing, feature_names=pts_col)
        plt.figure(figsize=(1200 / my_dpi, (1200 / my_dpi)), dpi=my_dpi)
        print(overall_classes.shape)
        print(glob_x_test.shape)
        print(glob_x_train.shape)
        print(len(final_features))
        # shap.summary_plot(overall_classes, glob_x_test, feature_names=final_features)
        # 11

        df.to_csv(resultsLocation + "Result_shap_" + name +".csv", index=True)
        exit()

        df = df.sort_index(ascending=False) # key=lambda x: x.str.lower(),
        # df['labels'] = df.index.str.lower()
        # df = df.sort_values('labels').drop('labels', axis=1)
        sns.barplot(y=df.index, x="values", data=df)
        plt.xticks(fontsize=8, rotation=90)  # ax.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()
        # plt.show()



        df.to_csv(resultsLocation + "Result_shap_"+name + ".csv", index=True)
        exit()





    # print_log("Final Model")
    # print_log("Final Model Accuracy: ", eval_model(param_grid)['acc'])

    # TODO: ### Cross-validation

    kfold_accuracy = pd.DataFrame(columns= [ str(k+1)+'-fold