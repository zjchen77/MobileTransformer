import numpy as np
import onnx
from onnx import helper, TensorProto

def positional_encoding(seq_length, hidden_size):
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, hidden_size, 2) * -(np.log(10000.0) / hidden_size))
    pos_enc = np.zeros((seq_length, hidden_size))
    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)
    return pos_enc.astype(np.float32)

seq_length = 50
hidden_size = 512
pos_enc = positional_encoding(seq_length, hidden_size)

# 创建ONNX图
graph = helper.make_graph(
    [
        # ONNX操作
    ],
    "positional_encoding_example",
    [
        # 输入
        helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, seq_length, hidden_size]),
    ],
    [
        # 输出
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, seq_length, hidden_size]),
    ],
    [
        # 初始化位置编码为常量
        helper.make_tensor("positional_encoding", TensorProto.FLOAT, [seq_length, hidden_size], pos_enc.flatten()),
    ],
)

# 创建ONNX模型
model = helper.make_model(graph, producer_name="positional_encoding_example")
onnx.save(model, "positional_encoding_example.onnx")
