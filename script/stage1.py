"""
Stage1 特徴抽出するプログラム
やってること:
    0. 今後の処理に害をあたえるので、音から直流成分を除去する
    1. 音にSTFTを適用して、その出力の絶対値をとって、振幅スペクトログラムを作る
    2. 振幅スペクトログラムをメル振幅スペクトログラムにする。
        ->メル〇〇とは・・・人間の知覚に沿った単位に変換。(人間の耳は低い音には敏感で、高い音には鈍感)
    3. メル振幅スペクトログラムにlogをかけて、対数メル振幅スペクトログラムをつくる
    ★ 今回の分類では、【対数メル振幅スペクトログラム】を特徴量としてつかう。よく使われるやつ。
"""






import matplotlib.pyplot as plt
import numpy as np
import librosa
from tqdm import tqdm
import os
import pickle
import pandas as pd







if __name__ == '__main__':
    # =============
    # 必須手動設定項目
    # =============
    dataset_name = 'sinsnode2' #どのデータセットの特徴抽出をするか







    # ===========
    # 手動設定項目
    # ===========
    window_length = 1024 #STFTの窓長
    hop_length = 320 #STFTの遷移幅
    feature_name = 'logmelspectrogram' #今回作る、特徴量の名前を設定







    # ========================================
    # 後から確認できるように、このプログラム自体をメモ
    # ========================================
    filename = os.path.splitext(os.path.basename(__file__))[0]
    with open(filename + '.py', mode='r') as f:
        code_contents = f.read()


    

    # =================================
    # Stage0で作ったデータフレームを読み込む
    # =================================
    with open('../output/' + dataset_name + '/dataframe/train/dataframe.pickle', 'rb') as f:
        train_df = pickle.load(f)
    with open('../output/' + dataset_name + '/dataframe/valid/dataframe.pickle', 'rb') as f:
        valid_df = pickle.load(f)
    with open('../output/' + dataset_name + '/dataframe/test/dataframe.pickle', 'rb') as f:
        test_df = pickle.load(f)







    def calc_logmelspectrogram(wave): #音からメル対数振幅スペクトログラムを計算する関数
        #直流除去
        wave = wave - np.mean(wave)
        #STFTして、さらにメルスペクトログラムの生成
        melspectrogram = librosa.feature.melspectrogram(wave, sr=16000, n_fft=window_length, hop_length=hop_length, n_mels=40, center=False)
        #対数を取る
        logmelspectrogram = np.log(melspectrogram + 0.0000001)
        return logmelspectrogram








    def create_one_sample(path): #分類モデルに入力するときの、1サンプルを生成するプログラム
        one_sample = list()
        wave, _ = librosa.load(path, mono=True, sr=16000) #サンプリング周波数を16kHzに変更しつつ、音を読み込む
        logmelspectrogram = calc_logmelspectrogram(wave) #メル対数振幅スペクトログラムを計算
        #numpyの配列で、(周波数, 時間, チャネル)の順に次元が並ぶようにする
        one_sample.append(logmelspectrogram)
        one_sample = np.array(one_sample)
        one_sample = one_sample.transpose(1,2,0)
        return one_sample


        


    
    def do_create_feature(df, data_type): #特徴量を作る関数
        label_data = []
        for i in tqdm(range(df.shape[0])): #すべてのサンプルで特徴量を作る
            returned_data = create_one_sample(path=df['path'].iloc[i]) #特徴量を計算
            with open('../' + 'output/' + dataset_name + '/' + feature_name + '/' + data_type + '/feature_' + str(i) + '.pickle', 'wb') as f:
                pickle.dump(returned_data, f) #特徴量を1サンプルずつ、.pickleファイルで保存
            label_data.append(df['label'].iloc[i]) #ラベルもリストに追加していく

        label_data = np.array(label_data)
        with open('../' + 'output/' + dataset_name + '/' + feature_name + '/' + data_type + '/' + '/label.pickle', 'wb') as f:
            pickle.dump(label_data, f) #ラベルを保存
        with open('../' + 'output/' + dataset_name + '/' + feature_name + '/' + data_type + '/' + '/memo.txt', 'w') as f:
            f.write(code_contents) #このファイル自体の内容も保存

    
    # ===================
    # 特徴量の作成を実行する
    # ===================
    # 保存する前に、保存する用のディレクトリなかったら作る
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/train/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/train/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/valid/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/valid/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/test/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/test/')



    print('======Train======')
    do_create_feature(df=train_df, data_type='train')
    print('======Valid======')
    do_create_feature(df=valid_df, data_type='valid')
    print('======Test======')
    do_create_feature(df=test_df, data_type='test')
    

