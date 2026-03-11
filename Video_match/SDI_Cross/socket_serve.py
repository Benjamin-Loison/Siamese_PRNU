import pickle
import socket

import numpy as np
import torch
from torch.autograd import Variable

from arch.effnet_pconv_eca_rep import EfficientNet_pconv_eca_rep,reparameterize_model

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def cnn(cross,model):

    data = torch.Tensor(cross[None, None, :, :])
    data = Variable(data)
    data = data.cuda()
    result = model(data).cpu().detach().numpy()
    return result

def socketServer():
    s = socket.socket()  # 创建socket对象
    host = '0.0.0.0'  # 设置本地主机作为地址
    port = 20000  # 设置端口
    s.bind((host, port))  # 绑定地址和端口
    s.listen(6)  # 开始监听，等待主机连接,表示可以使用6个链接排队

    model = EfficientNet_pconv_eca_rep(width_coeff=1.0, depth_coeff=1.0, dropout_rate=0.2, drop_connect_rate=0.2)
    pretrained_model = torch.load('./weight_Eff_repB0/Pixle_256_140/checkpoint.pt')
    model.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_model.items()})
    model = reparameterize_model(model)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print("服务端已就绪...")
    while 1:
        conn, addr = s.accept()  # 建立客户连接
        data_recv = conn.recv(1024)  # 先接收数据长度
        data_recv = str(data_recv.decode())  # 解码的数据长度int
        data_len,shape0,shape1 = int(data_recv.split('_')[0]),int(data_recv.split('_')[1]),int(data_recv.split('_')[2])
        data = b""  # 建立一个空的字节流，来存储数据
        while True:
            receive_data = conn.recv(1024)  # 注意512只是最多读取的字节数而不是实际字节数
            data = data + receive_data  # 存到字节流
            if len(data) == data_len:  # 判断数据长度是否达到总长度， 到了就说明数据接收完整，要跳出循环去解析
                break  # 数据解析，返回对应的numpy数组
        info_data = pickle.loads(data)
        info_data = np.frombuffer(info_data, dtype=np.float32).reshape((shape0,shape1))

        result = cnn(info_data,model)
        result = str(softmax(result)[0, 1])

        conn.send(result.encode())  # 接受到数据后再发送编码‘’发送出去。

        conn.close()  # 连接关闭

if __name__ == '__main__':

    socketServer()