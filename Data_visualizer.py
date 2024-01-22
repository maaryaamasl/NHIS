import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# outcome = "Chronic_Pain" ### size
outcome = "High_impact_chronic_pain"
sub_folder = 'shap1'
if outcome == "High_impact_chronic_pain": sub_folder = 'shap2'
print("outcome: ",outcome)

variable_list_df = pd.read_excel('NHIS variable list_Modified.xlsx')
selected_columns = variable_list_df['variable(s)'].tolist()
# description # category
column_desc = dict(zip(variable_list_df['variable(s)'].str.upper(),variable_list_df['description']))
column_cat = dict(zip(variable_list_df['variable(s)'].str.upper(), variable_list_df['category']))
print("column_desc ",len(column_desc), column_desc)
print("column_cat ",len(column_cat), column_cat)
feature_names = pd.read_csv('./'+sub_folder+'/columns.csv')['Column Names'].values
print("feature_names ", len(feature_names), feature_names )

arr_shape = np.loadtxt('./'+sub_folder+'/shape.csv')
shap_values = [ [] for i in range(int(arr_shape))]
for i in range(int(arr_shape)):
    shap_values[i] = np.loadtxt('./'+sub_folder+'/shap_'+str(i)+".csv")
print('\nD1 Classes:',len(shap_values),'\nD2 samples:', len(shap_values[0]))#,'\nD3 Columns/features:',len(shap_values[0][0]),'\nvalue:',shap_values[0][0][0])
print('type: ',type(shap_values))
print('type [0]: ', type(shap_values[0]))
average_shap_values = np.mean(np.abs(shap_values), axis=0)
print("average_shap_values shape", average_shap_values.shape)

df = pd.DataFrame(average_shap_values, columns=['values'],index=feature_names)
print(df.head(),"\n",str(list(df.index)),"\n", len(list(df.index)),"\n")
df[['label','cat','color']] = np.nan
df["inx"]=df.index
df['label'] = df["inx"].str.split('__', expand=True).apply(lambda x: f"{column_desc.get(x[0], '')} [{x[0]}] ({x[1]})", axis=1)#[0].map(column_desc)
# df['label'] = df['label'].str.replace('(None)', '')
df['cat'] = df["inx"].str.split('__', expand=True)[0].map(column_cat)
print(set(df['cat'].unique()))
df['color'] = df.cat.map({'risk factor':1, 'covariate':2, 'filter':3, 'risk factor and moderator':4, 'SES':5 })
df['color_label'] = df['cat']
df = df.sort_values(by='values', ascending=False, )
print(df.head(75),"\n")
# df['label'] = df['label'].str.replace("  "," ").replace("\n"," ").replace("\t"," ")

palette = sns.color_palette("bright", 10) # pastel
palette = {"risk factor": palette[2], "covariate": palette[3], "filter": palette[0],
           "risk factor and moderator": palette[9] , 'SES': palette[5]}  # 7 grey 5 dark red
hue_order = ["risk factor", "covariate", "filter", "risk factor and moderator", "SES"]

df_filtered = df.copy()
df_filtered['index_df'] = df.index
df_filtered['values'] = df_filtered['values'].abs()
df_filtered = df_filtered.sort_values(by='values', ascending=False).reset_index(drop=True)
df_filtered['label']= df_filtered['label'].apply(lambda x: x.capitalize() if isinstance(x, str) else x)
# df_filtered['label'] = df_filtered['label'].apply(lambda x: x.replace("Ldl","LDL").replace(" a1c "," A1C ")
#                                                   .replace("+instructional+w","+w").replace("(not employed)","- not employed")
#                                                   .replace("Hdl", "HDL")
#                                                   )

my_dpi =200
for i in range(0, df_filtered.shape[0], 51):
    import matplotlib as mpl

    mpl.rcParams['font.family'] = 'Arial'
    sns.set(font="Arial")

    partial_df = df_filtered.iloc[i:i + 50].copy()
    print("partial_df: ", partial_df.shape)
    plt.figure(figsize=(900 / my_dpi, (2000 / my_dpi) * ((partial_df.shape[0] + 10) / (51 + 10))), dpi=my_dpi) ### size
    # sns.set(style="ticks")
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    # sns.set_context("paper")
    ax = sns.barplot(y='label', x="values", data=partial_df, hue='color_label',
                     dodge=False, hue_order=hue_order,
                     palette=palette)  # ,palette=[palette[i] for i in partial_df.color.astype(int)])
    # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
    # plt.tight_layout()
    plt.legend(title="Predictors", loc='lower right', prop={'size': 23})
    plt.xlabel('Mean |SHAP| (average impact on model output magnitude)', fontsize=12)
    plt.ylabel('Variables', fontsize=12, rotation=0)
    # plt.show()
    plt.xlim(df_filtered['values'].min() - 0.0001, df_filtered['values'].max() * 1.02)
    plt.grid()

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
    plt.savefig( "Figs\\" +outcome+ "-Abs-" + str(i) + '.svg', bbox_inches="tight",
                pad_inches=0.3, format='svg')  # facecolor='y', , transparent=True, dpi=200 , format='eps'
    # plt.savefig(dataLocation + "Figs/" + "Abs-" + str(i), bbox_inches="tight",
    #             pad_inches=0.3)
    plt.clf()

exit()
if True:

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

        df_filtered.to_csv(dataLocation +"Figs/" + "Result_shap_abs_sorted" + ".csv", index=False) # index_label=True,
        print(df_filtered.head(5))
        print(df_filtered[df_filtered[['index_df','Description']].isna().any(axis=1)])
        exit()

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

