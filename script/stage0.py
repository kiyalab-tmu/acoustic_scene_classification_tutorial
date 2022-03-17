"""
Stage0 データの準備をするプログラム
やってること:
    1. こんなデータフレームをTrain, Validation, Testの3セット分作る
    2. 作ったデータフレームを.picklleファイルで保存する

* クラスとラベルは同じ意味
"""







import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import io
from matplotlib import pyplot as plt







if __name__ == '__main__':
    # ===========
    # 手動設定事項
    # ===========
    dataset_name = 'sinsnode2' #これから作るデータセットの名前を設定
    top_path = '/home/shiroma7/my_sins/audio/10s_10s/Node2/' #音ファイルのパスを設定 (データセットが/home/audioじゃなくて、僕のディレクトリにありますm(_ _)m)
    save_dir_path = '../output/' + dataset_name + '/dataframe/' #.pickleファイルを保存する場所を設定








    # ========================================
    # 後から確認できるように、このプログラム自体をメモ
    # ========================================
    filename = os.path.splitext(os.path.basename(__file__))[0]
    with open(filename + '.py', mode='r') as f:
        code_contents = f.read()
    







    # =========================================================
    # ファイル名やラベルの情報が入ったMATLABファイルをデータフレームへ変換
    # MATLABファイルはsourceディレクトリにある
    # =========================================================
    matdata = io.loadmat('../source/living_segment_10s_10slabels.mat', squeeze_me=True)["label_info"]
    df = pd.DataFrame(data=matdata, columns=['path', 'label', 'segment'])
    df = df[df['label'] != 'dont use'] #使っちゃいけないデータを捨てる
    df.loc[df['label'] == 'watching tv', 'label'] = 'watching_tv' #ラベル名を整頓
    df.loc[df['label'] == 'calling', 'label'] = 'social_activity' #callingはsocial_activityに改名
    df.loc[df['label'] == 'visit', 'label'] = 'social_activity' #callingはsocial_activityに改名
    
    df['path'] = top_path + df['path'] #ファイル名に、音ファイルのパスを追加
    print(df['label'].unique()) #最終的なラベルの種類を可視化







    

    # ==================================
    """
    ★音は時系列だから、
    過去のデータがTestに入るのはまずい。
    Train+Valid / Test に8:2分けてから、
    Train+Validを8:2分けてる。(でシャッフル)


    SINSは各クラスでサンプル数が違う。
        →各クラスごとに8:2に分ける


    各クラスごとにデータフレームを作って、それを分けてる
    データ数がめちゃくちゃ多いから、コンペと同じデータ数になるように削減してる
    """
    # ==================================
    #各クラスごとにデータフレームを作成
    df_absence = df[df['label'] == 'absence']
    df_other = df[df['label'] == 'other']
    df_working = df[df['label'] == 'working']
    df_cooking = df[df['label'] == 'cooking']
    df_eating = df[df['label'] == 'eating']
    df_social_activity = df[df['label'] == 'social_activity']
    df_dishwashing = df[df['label'] == 'dishwashing']
    df_vacuumcleaner = df[df['label'] == 'vacuumcleaner']
    df_watching_tv = df[df['label'] == 'watching_tv']
    
    #データ数を制限
    df_absence = df_absence[0:4715]
    df_other = df_other[0:515]
    df_working = df_working[0:4661]
    df_cooking = df_cooking[0:1281]
    df_eating = df_eating[0:577]
    df_social_activity = df_social_activity[0:1236]
    df_dishwashing = df_dishwashing[0:356]
    df_vacuumcleaner = df_vacuumcleaner[0:243]
    df_watching_tv = df_watching_tv[0:4661]


    #(各ラベルごとに)データフレームを train+valid と testに8:2で分割
    df_absence_train, df_absence_test = train_test_split(df_absence, test_size=0.2, shuffle = False)
    df_other_train, df_other_test = train_test_split(df_other, test_size=0.2, shuffle = False)
    df_working_train, df_working_test = train_test_split(df_working, test_size=0.2, shuffle = False)
    df_cooking_train, df_cooking_test = train_test_split(df_cooking, test_size=0.2, shuffle = False)
    df_eating_train, df_eating_test = train_test_split(df_eating, test_size=0.2, shuffle = False)
    df_social_activity_train, df_social_activity_test = train_test_split(df_social_activity, test_size=0.2, shuffle = False)
    df_dishwashing_train, df_dishwashing_test = train_test_split(df_dishwashing, test_size=0.2, shuffle = False)
    df_vacuumcleaner_train, df_vacuumcleaner_test = train_test_split(df_vacuumcleaner, test_size=0.2, shuffle = False)
    df_watching_tv_train, df_watching_tv_test = train_test_split(df_watching_tv, test_size=0.2, shuffle = False)
    trainvalid_df = pd.concat([df_absence_train, df_other_train, df_working_train, df_cooking_train, df_eating_train, df_social_activity_train, df_dishwashing_train, df_vacuumcleaner_train, df_watching_tv_train], axis=0)
    test_df = pd.concat([df_absence_test, df_other_test, df_working_test, df_cooking_test, df_eating_test, df_social_activity_test, df_dishwashing_test, df_vacuumcleaner_test, df_watching_tv_test], axis=0)


    #train+validを train と valid に8:2で分割
    train_df, valid_df = train_test_split(trainvalid_df,  test_size=0.2, random_state=0, stratify=trainvalid_df['label'])







    # ===============================================================
    # Train, Validation, Testの3つのセットに分けられたら、最後シャッフルする
    # ===============================================================
    train_df = train_df.sample(frac=1)
    valid_df = valid_df.sample(frac=1)
    test_df = test_df.sample(frac=1)
    #データフレームのindexを整える
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)






    # ============================
    # 最終的に完成したdataframeを表示
    # ============================
    print('=====Train=====')
    print(len(train_df))
    print(train_df['label'].value_counts())
    print('=====Validation=====')
    print(len(valid_df))
    print(valid_df['label'].value_counts())
    print('=====Test=====')
    print(len(test_df))
    print(test_df['label'].value_counts())






    # ======
    # 保存！
    # ======
    # 保存する前に、保存する用のディレクトリなかったら作る
    if not os.path.isdir('../' + 'output/' + dataset_name + '/dataframe/'):
        os.makedirs('../' + 'output/' + dataset_name + '/dataframe/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/dataframe/train/'):
        os.makedirs('../' + 'output/' + dataset_name + '/dataframe/train/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/dataframe/valid/'):
        os.makedirs('../' + 'output/' + dataset_name + '/dataframe/valid/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/dataframe/test/'):
        os.makedirs('../' + 'output/' + dataset_name + '/dataframe/test/')

    #保存
    with open(save_dir_path + 'train/dataframe.pickle', 'wb') as f:
        pickle.dump(train_df, f, protocol=4)
    with open(save_dir_path + 'valid/dataframe.pickle', 'wb') as f:
        pickle.dump(valid_df, f, protocol=4)
    with open(save_dir_path + 'test/dataframe.pickle', 'wb') as f:
        pickle.dump(test_df, f, protocol=4)
    with open(save_dir_path + 'test/memo.txt', 'w') as f:
        f.write(code_contents)




    
