# 画图
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_with_acc(loss_history, is_train):
    fig = plt.figure()
    # 坐标系ax1画曲线1
    if is_train:
        ax1 = fig.add_subplot(111)  # 指的是将plot界面分成1行1列，此子图占据从左到右从上到下的1位置
        ax1.plot(range(len(loss_history)), loss_history,
                c=np.array([255, 71, 90]) / 255.)  # c为颜色
        plt.ylabel('Loss')
    else:
        ax1 = fig.add_subplot(111)  # 指的是将plot界面分成1行1列，此子图占据从左到右从上到下的1位置
        loss_x_index = len(loss_history)
        loss_x = []
        for i in range(loss_x_index):
            loss_x.append(i * 20)
        ax1.plot(loss_x, loss_history, c=np.array([255, 71, 90]) / 255.)  # c为颜色
        plt.ylabel('Loss')

    plt.xlabel('Epoch')
    if is_train:
        plt.title('Training Loss')
    else:
        plt.title('Val Loss & Val Accuracy')
    plt.show()
    

file_jsonl_path = "output.txt"

i = 1
j = 1
train_loss = []
train_acc = []
val_loss = []
val_acc = []
with open(file_jsonl_path, 'r', encoding='utf-8') as file:
    for line in file:
        train_loss.append(float(line.strip()))
plot_loss_with_acc(train_loss, True)
