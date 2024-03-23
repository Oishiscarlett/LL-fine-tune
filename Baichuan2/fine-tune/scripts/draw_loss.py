import matplotlib.pyplot as plt
import numpy as np
import ast # 用于转换txt中的每一行to python字典

data_path = "fine-tune\loss_v1.txt"
loss_val = []


# 移动平均平滑
def moving_average(values, window_size):
    """计算移动平均"""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(values, weights, 'valid')

# 指数平滑
def exponential_smoothing(loss_val, alpha):
    smoothed_loss = [loss_val[0]]
    for i in range(1, len(loss_val)):
        smoothed_loss.append(alpha * loss_val[i] + (1-alpha)*smoothed_loss[i-1])
    return smoothed_loss

def draw_loss_img():
    with open (data_path, "r") as f:
        for line in f:
            dict_line = ast.literal_eval(line.strip()) # 将每行转换为字典
            loss_val.append(dict_line['loss']) # 获取loss值并添加到列表中

    # loss_sampled = loss_val[::50]
    loss_sampled = loss_val
    # smoothed_loss = smooth_loss(loss_sampled, 0.9)
    smoothed_loss = exponential_smoothing(loss_sampled, 0.1)

    plt.plot(loss_sampled,label="loss_val", alpha=0.6)
    plt.plot(smoothed_loss,label="smoothed_loss")

    plt.xlabel("iters")
    plt.ylabel("loss")
    plt.legend() #图例；要设置plot的label
    plt.grid() #格子
    # plt.show()
    plt.savefig("fine-tune\img\loss_v1.png") # 保存图片
    plt.show()


if __name__ == "__main__":
    draw_loss_img()