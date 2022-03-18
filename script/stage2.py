"""
分類モデルを学習して、テストデータに対する精度を評価するプログラム
"""







# =======
# 初期設定
# =======
import sys
import datetime
import hashlib
import os
print('今回の実験を保存するディレクトリの名前を設定')
result_dir_name = input()
print('使うGPUの番号を入力')
gpu_id = int(input())
ex_name = 'asc_' + hashlib.md5(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').encode()).hexdigest() #毎回実験を回すごとに、日時からハッシュ値を計算。世界で一つだけの実験IDとする。
#あとで確認できるように、このファイル自体の内容をメモ
filename = os.path.splitext(os.path.basename(__file__))[0]
with open(filename + '.py', mode='r') as f:
    code_contents = f.read()
print('=====Program started=====')
print('今回の実験ID', ex_name)








# ==========
# 手動設定項目
# ==========
import torchvision.models as models
import torch.nn as nn
import timm
from torchvision import transforms
dataset_name = 'sinsnode2'
feature_name = 'logmelspectrogram'
epoch_num = 100
batch_size = 32
learning_rate = 0.0001
transform=transforms.Compose([
    transforms.ToTensor()
    ])
"""
モデルの設定
色々設定できる。Convmicerはtimmというライブラリからゲットしてる
"""
# ==========ResNet50==========
# model = models.resnet50(num_classes=9, pretrained=False)
# model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# ==========Convmixer==========
model = timm.create_model('convmixer_768_32', pretrained = False, num_classes = 9)
model.stem[0] = nn.Conv2d(1, 768, kernel_size=7, stride=7)
# ==========ViT-B16==========
# model = timm.create_model('vit_base_patch16_224', pretrained = True, num_classes = 9)












# ==================
# 必要な変数とかの設定
# ==================
import pickle
import torch
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets
from asc_my_dataset import AscMyDataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
device = torch.device('cuda', index=gpu_id if torch.cuda.is_available() else 'cpu') #GPUの設定
loss_func = nn.CrossEntropyLoss() #損失関数の設定
loss_history = []
acc_history = []
valid_loss_history = []
valid_acc_history = []








# ===================
# データセットを読み込む
# ===================
train_dataset = AscMyDataset(transform, dataset_name, feature_name, 'train')
valid_dataset = AscMyDataset(transform, dataset_name, feature_name, 'valid')
test_dataset  = AscMyDataset(transform, dataset_name, feature_name, 'test')









# ====================================
# 結果を保存するディレクトリがなかったら生成
# ====================================
if not os.path.isdir('../' + 'result/'):
    os.makedirs('../' + 'result/')
if not os.path.isdir('../' + 'result/' + result_dir_name + '/'):
    os.makedirs('../' + 'result/' + result_dir_name + '/')









# ==========================
# 学習を実行する関数 (1epoch分)
# ==========================
def train(model, device, train_loader, optimizer, epoch, epoch_num):
    total_loss = 0
    total_correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)): #全サンプルを学習する
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 勾配を初期化
        output = model(data) # 順伝播の計算
        loss = loss_func(output, target) # 誤差を計算
        loss.backward() # 誤差逆伝播
        optimizer.step() # 重みを更新する
        total_loss += loss.item() # lossの値をPythonのfloatとして取り出す
        pred = output.argmax(dim=1, keepdim=True)
        total_correct += pred.eq(target.view_as(pred)).sum().item()

    """
    Lossの値や正解率を計算・表示
    """
    loss_result = total_loss / len(train_loader.dataset)
    total_correct = total_correct / len(train_loader.dataset)
    print('Epoch' + str(epoch) + '/' + str(epoch_num) +  ': loss: ' + '{:.4f}'.format(loss_result) + ' acc:' + '{:.4f}'.format(total_correct), end = '')
    return loss_result, total_correct









# =======================
# Validationを実行する関数
# =======================
def valid(model, device, valid_loader, optimizer, epoch, epoch_num, prev_metric):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)
            total_loss += loss.item() # lossの値をPythonのfloatとして取り出す
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()

    """
    Lossの値や正解率を計算・表示
    """
    loss_result = total_loss / len(valid_loader.dataset)
    accuracy = total_correct / len(valid_loader.dataset)
    print(' valid_loss: ' + '{:.4f}'.format(loss_result) + ' valid_acc:' + '{:.4f}'.format(accuracy))
    if prev_metric is None or accuracy > prev_metric: #もし、今までより性能がよかったらモデルを保存する
        prev_metric = accuracy
        torch.save(model.state_dict(), '../' + 'result/' + result_dir_name + '/' + ex_name + '_model.pt')
        print('=====Model is saved !=====')
    return loss_result, accuracy, prev_metric










# ==================
# テストを実行する関数
# ==================
def test(model, device, test_loader, epoch_num):
    model.eval()
    total_loss = 0
    total_correct = 0
    all_target = torch.tensor([]).to(device)
    all_output = torch.tensor([]).to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output2 = model(data)
            all_target = torch.cat([all_target, target], dim=0)
            all_output = torch.cat([all_output, output2.argmax(dim=1, keepdim=False)], dim=0)
            loss = loss_func(output, target)
            total_loss += loss.item() # lossの値をPythonのfloatとして取り出す
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            
    
    all_target = all_target.to('cpu')
    all_output = all_output.to('cpu')
        
    """
    Lossの値や正解率を計算・表示
    """
    loss_result = total_loss / len(test_loader.dataset)
    total_correct = total_correct / len(test_loader.dataset)
    f_score = calc_f_score(all_target, all_output)
    print('\n\n=====TEST=====\n' + ' loss: ' + '{:.4f}'.format(loss_result) + ' acc:' + '{:.4f}'.format(total_correct), ' f-score:' + str(f_score))
    create_confusion_matrix(all_target, all_output)
    return loss_result, total_correct, f_score








# =============================
"""
コンフュージョンマトリックスを作成する関数
各クラスでデータ数が違うから、正解数を載せたコンフュージョンマトリックスは意味ない。
正解率を%で表したコンフュージョンマトリックスを作成する
"""
# =======================
def create_confusion_matrix(true_label, predicted_label):
    cm_yoko_percent = confusion_matrix(true_label, predicted_label)
    cm_yoko_percent = np.array(cm_yoko_percent, dtype='f4')
    for i in range(cm_yoko_percent.shape[0]):
        row_sum = np.sum(cm_yoko_percent[i])
        for j in range(cm_yoko_percent.shape[1]):
            if row_sum == 0:
                cm_yoko_percent[i][j] = 0
            else:
                cm_yoko_percent[i][j] = cm_yoko_percent[i][j] / row_sum * 100
    with open('../output/' + dataset_name + '/' + feature_name + '/test/label.pickle', 'rb') as f:
        labels = pickle.load(f)
    classes = np.unique(labels)
    classes = {v: i for i, v in enumerate(sorted(classes))}
    plt.figure(figsize=(8,8))
    sns.heatmap(cm_yoko_percent, annot=True, xticklabels=classes, yticklabels=classes, cmap=plt.cm.Blues, cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # plt.xticks(rotation=45)
    # plt.yticks(rotation=45)
    plt.show()
    plt.savefig('../' + 'result/' + result_dir_name + '/' + ex_name + '_yoko_percent_cm.png')
    plt.clf()





# ===============================
"""
評価指標であるF-scoreを計算する関数
評価指標はAccuracyではなく、F-scoreを使う。
これも、クラス間でデータ数に差があるため
"""
# ===============================
def calc_f_score(true_label, predicted_label):
    f1 = f1_score(true_label, predicted_label, average="weighted")
    return f1





# =======
# 終了処理
# =======
def do_finish(train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc, test_f_score, loss_history, acc_history, valid_loss_history, valid_acc_history):
    # =================
    # 学習結果のグラフを保存
    # =================
    #loss
    plt.figure(figsize=(8,8))
    plt.title('Loss Value')
    plt.plot(loss_history)
    plt.plot(valid_loss_history)
    plt.legend(['loss', 'valid_loss'])
    plt.show()
    plt.savefig('../' + 'result/' + result_dir_name + '/' + ex_name + '_loss_history.png')
    plt.clf()
    #acc
    plt.figure(figsize=(8,8))
    plt.title('Accuracy')
    plt.plot(acc_history)
    plt.plot(valid_acc_history)
    plt.legend(['acc', 'valid_acc'])
    plt.xlim(0, epoch_num) 
    plt.ylim(0, 1.0)
    plt.show()
    plt.savefig('../' + 'result/' + result_dir_name + '/' + ex_name + '_acc_history.png')
    plt.clf()

    # =================
    # ログを保存
    # =================
    with open('../' + 'result/' + result_dir_name + '/' + ex_name + '_memo.txt', mode='w') as f:
        f.write('\ntrain_loss:' + str(train_loss))
        f.write('\nvalid_loss:' + str(valid_loss))
        f.write('\ntest_loss:' + str(test_loss))
        f.write('\ntrain_acc:' + str(train_acc))
        f.write('\nvalid_acc:' + str(valid_acc))
        f.write('\ntest_acc:' + str(test_acc))
        f.write('\ntest_f_score:' + str(test_f_score))
        f.write('\n\n\n\n\n=====実際のコード=====\n\n\n\n')
        f.write(code_contents)
    print('=====All information is saved !=====')
    print('今回の実験ID', ex_name)

    
       

# =========
# メイン処理
# =========
if __name__ == '__main__':
    # ====================
    # データローダーの読み込み
    # ====================
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    

    # =============
    # 学習周りの設定
    # =============
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


    # ====
    # 学習
    # ====
    prev_metric = None
    for epoch in range(1, epoch_num + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, epoch_num)
        loss_history.append(train_loss)
        acc_history.append(train_acc)
        valid_loss, valid_acc, prev_metric = valid(model, device, valid_loader, optimizer, epoch, epoch_num, prev_metric)
        valid_loss_history.append(valid_loss)
        valid_acc_history.append(valid_acc)
    
    """
    最良モデルを読み込んでからテスト
    """
    model.load_state_dict(torch.load('../result/' + result_dir_name + '/' + ex_name + '_model.pt', map_location='cuda:'+str(gpu_id)))
    model.to(device)
    test_loss, test_acc, test_f_score = test(model, device, test_loader, epoch_num)
    

    # =======
    # 終了処理
    # =======
    do_finish(train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc, test_f_score, loss_history, acc_history, valid_loss_history, valid_acc_history)
