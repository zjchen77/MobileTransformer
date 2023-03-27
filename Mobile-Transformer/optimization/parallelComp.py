import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

# 定义参数
seq_length = 512
hidden_size = 64
num_heads = 8
head_dim = hidden_size // num_heads
block_size = 64  # 分块大小

# 创建ONNX图
graph = helper.make_graph(
    [
        # 分割输入序列为较小的块
        helper.make_node("Slice", ["input", "block_starts", "block_ends", "axes", "steps"],
                         ["sliced_input"], name="slice_input"),

        # 计算Q, K, V矩阵
        helper.make_node("MatMul", ["sliced_input", "w_q"], ["q"], name="matmul_q"),
        helper.make_node("MatMul", ["sliced_input", "w_k"], ["k"], name="matmul_k"),
        helper.make_node("MatMul", ["sliced_input", "w_v"], ["v"], name="matmul_v"),

        # 计算注意力得分
        helper.make_node("MatMul", ["q", "k"], ["attention_score"], name="matmul_attention_score"),
        helper.make_node("Div", ["attention_score", "scale"], ["scaled_attention_score"], name="div_scale"),
        helper.make_node("Softmax", ["scaled_attention_score"], ["attention_weights"], axis=-1, name="softmax"),

        # 计算加权和
        helper.make_node("MatMul", ["attention_weights", "v"], ["weighted_sum"], name="matmul_weighted_sum"),

        # 拼接结果
        helper.make_node("Concat", ["weighted_sum"], ["output"], axis=1, name="concat_output"),
    ],
    "blockwise_attention_example",
    [
        # 输入
        helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, seq_length, hidden_size]),
    ],
    [
        # 输出
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, seq_length, hidden_size]),
    ],
    [
        # 初始化参数
        helper.make_tensor("w_q", TensorProto.FLOAT, [hidden_size, hidden_size],
                           np.random.randn(hidden_size, hidden_size).flatten().astype(np.float32)),
        helper.make_tensor("w_k", TensorProto.FLOAT, [hidden_size, hidden_size],
                           np.random.randn(hidden_size, hidden_size).flatten().astype(np.float32)),
        helper.make_tensor("w_v", TensorProto.FLOAT, [hidden_size, hidden_size],
                           np.random.randn(hidden_size, hidden_size).flatten().astype(np.float32)),
        helper.make_tensor("block_starts", TensorProto.INT64, [1], [0]),
        helper.make_tensor("block_ends", TensorProto.INT64, [1], [block_size]),
        helper.make_tensor("axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("steps", TensorProto.INT64, [1], [block_size]),
        helper.make_tensor("scale", TensorProto.FLOAT, [], [np.sqrt(head_dim).astype(np.float32)]),
    ],
)

# 创建ON
