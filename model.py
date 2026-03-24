import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

torch.manual_seed(42)
np.random.seed(42)

class PositionalEncoding(nn.Module):
    """
    位置编码模块：为序列中的每个位置添加位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        参数:
            d_model: 模型的维度（比如512）
            max_len: 支持的最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵(一个全为0的矩阵)
        pe = torch.zeros(max_len, d_model)  # shape: [max_len, d_model]
        
        # 创建位置索引 [0, 1, 2, ..., max_len-1]，并将其变为一个[max_len, 1]的列向量
        position = torch.arange(0, max_len).unsqueeze(1).float()  # shape: [max_len, 1]
        
        # 计算分母项 (arange计算2i，div_term = e^(2i*(-ln(10000)/d_model)) 即位置编码的分母)
        # 这里使用了一个技巧：用log和exp避免大数运算
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )  # shape: [d_model/2]
        
        # 偶数维度使用sin
        pe[:, 0::2] = torch.sin(position * div_term)  # shape: [max_len, d_model/2]
        # 奇数维度使用cos
        pe[:, 1::2] = torch.cos(position * div_term)  # shape: [max_len, d_model/2]
        
        # 增加batch维度，并注册为buffer（不参与梯度更新）
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
        print(f"   位置编码初始化完成！")
        print(f"   最大序列长度: {max_len}")
        print(f"   模型维度: {d_model}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        输入:
            x: [batch_size, seq_len, d_model] -- 词嵌入
            
        输出:
            [batch_size, seq_len, d_model] -- 添加位置编码后的结果
            
        数据流示例:
            输入 x: [32, 100, 512]  # 32个样本，每个100个词，每个词512维
            位置编码: [1, 100, 512]  # 前100个位置的编码
            输出: [32, 100, 512]     # 相加后的结果
        """
        seq_len = x.size(1)
        
        # 获取对应长度的位置编码并相加
        # self.pe[:, :seq_len] 的shape: [1, seq_len, d_model]
        # 广播机制会自动扩展到batch维度
        output = x + self.pe[:, :seq_len]
        
        return output


# 测试位置编码
def test_positional_encoding():
    """测试位置编码模块"""
    print("\n" + "="*50)
    print("测试位置编码")
    print("="*50)
    
    batch_size = 2
    seq_len = 10
    d_model = 8
    
    # 创建模拟输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入shape: {x.shape}")
    
    # 创建位置编码层
    pos_encoder = PositionalEncoding(d_model, max_len=100)
    
    # 前向传播
    output = pos_encoder(x)
    print(f"输出shape: {output.shape}")
    print(f"位置编码测试通过！\n")
    
    return output

# 运行测试
# _ = test_positional_encoding()

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    缩放点积注意力机制
    
    公式: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    
    参数:
        query: [batch_size, n_heads, seq_len, d_k] -- 查询矩阵
        key: [batch_size, n_heads, seq_len, d_k] -- 键矩阵
        value: [batch_size, n_heads, seq_len, d_v] -- 值矩阵
        mask: [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len] -- 掩码
        dropout: Dropout层（可选）
    
    返回:
        output: [batch_size, n_heads, seq_len, d_v] -- 注意力输出
        attention_weights: [batch_size, n_heads, seq_len, seq_len] -- 注意力权重
    
    数据流示例:
        翻译 "我爱北京" -> "I love Beijing"
        query: [32, 8, 10, 64]  # 32个样本，8个头，10个词，每个词64维
        key:   [32, 8, 10, 64]  
        value: [32, 8, 10, 64]
        
        步骤1: QK^T -> [32, 8, 10, 10]  # 每个词对每个词的注意力分数
        步骤2: 缩放 -> [32, 8, 10, 10] / sqrt(64) = [32, 8, 10, 10] / 8
        步骤3: softmax -> [32, 8, 10, 10]  # 归一化为概率
        步骤4: 乘以V -> [32, 8, 10, 64]  # 加权求和得到输出
    """
    
    # 获取d_k（最后一个维度的大小）
    d_k = query.size(-1)
    
    # 步骤1: 计算注意力分数 QK^T
    # query: [batch_size, n_heads, seq_len_q, d_k]
    # key转置: [batch_size, n_heads, d_k, seq_len_k]
    # scores: [batch_size, n_heads, seq_len_q, seq_len_k]
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 步骤2: 缩放
    scores = scores / math.sqrt(d_k)
    
    # 步骤3: 如果有mask，应用mask
    if mask is not None:
        # mask中为0的位置设为-inf，softmax后会变成0
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 步骤4: Softmax归一化
    attention_weights = F.softmax(scores, dim=-1)
    
    # 步骤5: 如果有dropout，应用dropout
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # 步骤6: 乘以V得到输出
    # attention_weights: [batch_size, n_heads, seq_len_q, seq_len_k]
    # value: [batch_size, n_heads, seq_len_k, d_v]
    # output: [batch_size, n_heads, seq_len_q, d_v]
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


# 测试注意力机制
def test_attention():
    """测试缩放点积注意力"""
    print("\n" + "="*50)
    print("测试缩放点积注意力")
    print("="*50)
    
    batch_size = 2
    n_heads = 4
    seq_len = 6
    d_k = 16
    
    # 创建Q, K, V
    Q = torch.randn(batch_size, n_heads, seq_len, d_k)
    K = torch.randn(batch_size, n_heads, seq_len, d_k)
    V = torch.randn(batch_size, n_heads, seq_len, d_k)
    
    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    
    # 计算注意力
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"输出 shape: {output.shape}")
    print(f"注意力权重 shape: {weights.shape}")
    print(f"注意力权重和: {weights[0, 0, 0].sum():.4f}")
    print(f"注意力机制测试通过！\n")
    
    return output, weights

# 运行测试
# _ = test_attention()

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    核心思想：
    - 单个注意力可能只关注某一种关系
    - 多个注意力头可以关注不同类型的关系
    - 最后将所有头的输出拼接起来
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        参数:
            d_model: 模型维度（必须能被n_heads整除）
            n_heads: 注意力头数
            dropout: Dropout概率
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0, f"d_model({d_model})必须能被n_heads({n_heads})整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # - 共4个Linear
        # - 3个用于生成Q、K、V
        # - 1个用于最后的输出映射
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # [d_model, d_model]
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # [d_model, d_model]
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # [d_model, d_model]
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # [d_model, d_model]
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
        
        print(f"   多头注意力初始化完成！")
        print(f"   模型维度: {d_model}")
        print(f"   注意力头数: {n_heads}")
        print(f"   每个头的维度: {self.d_k}")
    
    def _init_weights(self):
        """Xavier初始化权重"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        输入:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: [batch_size, 1, 1, seq_len] or None
            
        输出:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, n_heads, seq_len_q, seq_len_k]
            
        数据流示例:
            输入句子: "Hello world" (2个词)
            query/key/value: [32, 2, 512]  # 32个样本，2个词，512维
            
            步骤1: 线性变换
                Q = W_q * query -> [32, 2, 512]
                K = W_k * key   -> [32, 2, 512]
                V = W_v * value -> [32, 2, 512]
            
            步骤2: 分头（reshape）
                Q -> [32, 2, 8, 64] -> [32, 8, 2, 64]  # 8个头，每个头64维
                K -> [32, 2, 8, 64] -> [32, 8, 2, 64]
                V -> [32, 2, 8, 64] -> [32, 8, 2, 64]
            
            步骤3: 计算注意力
                每个头独立计算注意力 -> [32, 8, 2, 64]
            
            步骤4: 拼接所有头
                [32, 8, 2, 64] -> [32, 2, 8, 64] -> [32, 2, 512]
            
            步骤5: 最终线性变换
                W_o * concat -> [32, 2, 512]
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        
        # 步骤1: 线性变换生成Q、K、V
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        Q = self.w_q(query)  # [32, 10, 512]
        K = self.w_k(key)    # [32, 10, 512]
        V = self.w_v(value)  # [32, 10, 512]
        
        # 步骤2: 分头
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_k]
        # -> [batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # 现在: Q, K, V 的shape都是 [32, 8, 10, 64]
        
        # 步骤3: 计算缩放点积注意力
        # [batch_size, n_heads, seq_len_q, d_k] -> [batch_size, n_heads, seq_len_q, d_k]
        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask, self.dropout
        )
        # attention_output: [32, 8, 10, 64]
        # attention_weights: [32, 8, 10, 10]
        
        # 步骤4: 拼接多头输出
        # [batch_size, n_heads, seq_len_q, d_k] -> [batch_size, seq_len_q, n_heads, d_k]
        # -> [batch_size, seq_len_q, d_model]
        # contiguous的原因是转置导致内存不连续，而view的前提是内存连续，因此contiguous使转置后的张量连续
        # n_heads和seq_len_q两个维度换来换去有两个原因：1、为了批量进行矩阵乘法，因此将n_heads放到前面当作batch来并行处理2、为了合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        # 现在: [32, 10, 512]
        
        # 步骤5: 最终的线性变换
        output = self.w_o(attention_output)
        # output: [32, 10, 512]
        
        return output, attention_weights


# 测试多头注意力
def test_multihead_attention():
    """测试多头注意力机制"""
    print("\n" + "="*50)
    print("测试多头注意力")
    print("="*50)
    
    batch_size = 2
    seq_len = 5
    d_model = 256
    n_heads = 8
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入 shape: {x.shape}")
    
    # 创建多头注意力层
    mha = MultiHeadAttention(d_model, n_heads)
    
    # 自注意力：Q=K=V
    output, weights = mha(x, x, x)
    
    print(f"输出 shape: {output.shape}")
    print(f"注意力权重 shape: {weights.shape}")
    print(f"多头注意力测试通过！\n")
    
    return output, weights

# 运行测试
# _ = test_multihead_attention()

class FeedForward(nn.Module):
    """
    前馈神经网络
    
    公式: FFN(x) = max(0, xW1 + b1)W2 + b2  -- 这个max(0,xW1+b1)其实干的就是ReLU的事情
    
    特点：
    - 对每个位置独立进行相同的操作
    - 包含两个线性变换和一个ReLU激活
    - 通常中间层维度是模型维度的4倍
    """
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        """
        参数:
            d_model: 模型维度
            d_ff: 前馈网络的中间层维度（默认为4*d_model）
            dropout: Dropout概率
        """
        super(FeedForward, self).__init__()
        
        # 如果没有指定d_ff，默认为4倍的d_model（Transformer论文的设置）
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 两个线性层（一个升维矩阵，一个降维矩阵）
        self.linear1 = nn.Linear(d_model, d_ff)    # [d_model, d_ff]
        self.linear2 = nn.Linear(d_ff, d_model)    # [d_ff, d_model]
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数（使用ReLU）
        self.activation = nn.ReLU()
        
        print(f"   前馈网络初始化完成！")
        print(f"   输入/输出维度: {d_model}")
        print(f"   中间层维度: {d_ff}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        输入:
            x: [batch_size, seq_len, d_model]
            
        输出:
            [batch_size, seq_len, d_model]
            
        数据流示例:
            输入: [32, 100, 512]      # 32个样本，100个位置，512维
            
            步骤1: 第一个线性层
                [32, 100, 512] -> [32, 100, 2048]  # 扩展到4倍
            
            步骤2: ReLU激活
                [32, 100, 2048] -> [32, 100, 2048]
            
            步骤3: Dropout
                [32, 100, 2048] -> [32, 100, 2048]
            
            步骤4: 第二个线性层
                [32, 100, 2048] -> [32, 100, 512]  # 压缩回原始维度
            
            步骤5: Dropout
                [32, 100, 512] -> [32, 100, 512]
        """
        # 第一个线性变换 + 激活函数
        hidden = self.linear1(x)           # [batch_size, seq_len, d_ff]
        hidden = self.activation(hidden)   # ReLU激活
        hidden = self.dropout(hidden)      # Dropout
        
        # 第二个线性变换
        output = self.linear2(hidden)      # [batch_size, seq_len, d_model]
        output = self.dropout(output)      # Dropout
        
        return output


# 测试前馈网络
def test_feedforward():
    """测试前馈神经网络"""
    print("\n" + "="*50)
    print("测试前馈网络")
    print("="*50)
    
    batch_size = 2
    seq_len = 10
    d_model = 512
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入 shape: {x.shape}")
    
    # 创建前馈网络
    ff = FeedForward(d_model)
    
    # 前向传播
    output = ff(x)
    print(f"输出 shape: {output.shape}")
    print(f"前馈网络测试通过！\n")
    
    return output

# 运行测试
# _ = test_feedforward()

class LayerNorm(nn.Module):
    """
    层归一化
    
    为什么需要LayerNorm？
    - BatchNorm在序列任务中效果不好（序列长度可变）
    - LayerNorm对每个样本独立进行归一化
    - 有助于稳定训练，加速收敛
    
    公式: y = γ * (x - μ) / σ + β
    其中：
    - μ: 均值
    - σ: 标准差
    - γ: 可学习的缩放参数
    - β: 可学习的偏移参数
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        参数:
            d_model: 特征维度
            eps: 防止除零的小常数
        """
        super(LayerNorm, self).__init__()
        
        self.d_model = d_model
        self.eps = eps
        
        # 可学习的缩放和偏移参数
        self.gamma = nn.Parameter(torch.ones(d_model))   # 初始化为1
        self.beta = nn.Parameter(torch.zeros(d_model))   # 初始化为0
        
        print(f"LayerNorm初始化完成！维度: {d_model}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        输入:
            x: [batch_size, seq_len, d_model]
            
        输出:
            [batch_size, seq_len, d_model]
            
        数据流示例:
            输入: [32, 100, 512]
            
            步骤1: 计算均值和方差（在最后一个维度上）
                mean: [32, 100, 1]
                var: [32, 100, 1]
            
            步骤2: 归一化
                x_norm = (x - mean) / sqrt(var + eps)
            
            步骤3: 缩放和偏移
                output = gamma * x_norm + beta
        """
        # 计算均值和方差（在最后一个维度上）
        mean = x.mean(dim=-1, keepdim=True)    # [batch_size, seq_len, 1]
        var = x.var(dim=-1, keepdim=True)      # [batch_size, seq_len, 1]
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和偏移
        output = self.gamma * x_norm + self.beta
        
        return output

class EncoderLayer(nn.Module):
    """
    Transformer编码器层
    
    结构：
    1. 多头自注意力
    2. Add & Norm（残差连接 + 层归一化）
    3. 前馈网络
    4. Add & Norm（残差连接 + 层归一化）
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1
    ):
        """
        参数:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络中间层维度
            dropout: Dropout概率
        """
        super(EncoderLayer, self).__init__()
        
        # 多头自注意力
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 两个LayerNorm层
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        print(f"编码器层初始化完成！")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        输入:
            x: [batch_size, seq_len, d_model] - 输入序列
            mask: [batch_size, 1, 1, seq_len] - 注意力掩码（可选）
            
        输出:
            [batch_size, seq_len, d_model] - 编码后的序列
            
        数据流示例（文本编码）:
            输入句子: "我爱北京天安门" (7个字)
            x: [32, 7, 512]  # 32个样本，7个字，512维
            
            步骤1: 自注意力
                attn_output: [32, 7, 512]
            
            步骤2: 残差连接 + LayerNorm
                x = LayerNorm(x + attn_output)
            
            步骤3: 前馈网络
                ff_output: [32, 7, 512]
            
            步骤4: 残差连接 + LayerNorm
                x = LayerNorm(x + ff_output)
            
            输出: [32, 7, 512]
        """
        # 子层1: 自注意力
        # 保存残差
        residual = x
        
        # 自注意力（Q=K=V都是x）
        attn_output, _ = self.self_attention(x, x, x, mask)
        
        # Dropout
        attn_output = self.dropout(attn_output)
        
        # 残差连接 + LayerNorm
        x = self.norm1(residual + attn_output)
        
        # 子层2: 前馈网络
        # 保存残差
        residual = x
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        
        # 残差连接 + LayerNorm
        x = self.norm2(residual + ff_output)
        
        return x

class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    
    结构：
    1. Masked多头自注意力（防止看到未来信息）
    2. Add & Norm
    3. 编码器-解码器交叉注意力
    4. Add & Norm
    5. 前馈网络
    6. Add & Norm
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1
    ):
        """
        参数:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络中间层维度
            dropout: Dropout概率
        """
        super(DecoderLayer, self).__init__()
        
        # Masked自注意力（用于目标序列）
        self.masked_self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 交叉注意力（用于关注编码器输出）
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 三个LayerNorm层
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        print(f"解码器层初始化完成！")
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        输入:
            x: [batch_size, tgt_len, d_model] - 目标序列
            encoder_output: [batch_size, src_len, d_model] - 编码器输出
            src_mask: [batch_size, 1, 1, src_len] - 源序列掩码
            tgt_mask: [batch_size, 1, tgt_len, tgt_len] - 目标序列掩码
            
        输出:
            [batch_size, tgt_len, d_model] - 解码后的序列
            
        数据流示例:
            源句子: "我爱北京" (编码器已处理)
            目标句子: "I love Beijing" (正在生成)
            
            x: [32, 3, 512]  # "I love Beijing"
            encoder_output: [32, 4, 512]  # "我爱北京"的编码
            
            步骤1: Masked自注意力（只能看到已生成的词）
                生成"Beijing"时只能看到"I love"
            
            步骤2: 交叉注意力（关注源句子）
                决定"Beijing"应该对应"北京"
            
            步骤3: 前馈网络
                进一步处理特征
        """
        # 子层1: Masked自注意力
        residual = x
        
        # Masked自注意力
        masked_attn_output, _ = self.masked_self_attention(x, x, x, tgt_mask)
        masked_attn_output = self.dropout(masked_attn_output)
        
        # 残差连接 + LayerNorm
        x = self.norm1(residual + masked_attn_output)
        
        # 子层2: 交叉注意力
        residual = x
        
        # 交叉注意力：Q来自解码器，K和V来自编码器
        cross_attn_output, _ = self.cross_attention(
            x,                # Query来自解码器
            encoder_output,   # Key来自编码器
            encoder_output,   # Value来自编码器
            src_mask
        )
        cross_attn_output = self.dropout(cross_attn_output)
        
        # 残差连接 + LayerNorm
        x = self.norm2(residual + cross_attn_output)
        
        # 子层3: 前馈网络
        residual = x
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        
        # 残差连接 + LayerNorm
        x = self.norm3(residual + ff_output)
        
        return x

class TransformerEncoder(nn.Module):
    """
    完整的Transformer编码器
    
    包含:
    - 词嵌入层
    - 位置编码
    - N个编码器层的堆叠
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        参数:
            vocab_size: 词汇表大小
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: 编码器层数
            d_ff: 前馈网络维度
            max_len: 最大序列长度
            dropout: Dropout概率
        """
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # 词嵌入层（将词ID转换为向量）
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # 堆叠N个编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化嵌入层权重
        self._init_embeddings()
        
        print(f"Transformer编码器初始化完成！")
        print(f"   词汇表大小: {vocab_size}")
        print(f"   编码器层数: {n_layers}")
        print(f"   模型维度: {d_model}")
    
    def _init_embeddings(self):
        """初始化嵌入层权重"""
        # 使用正态分布初始化
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model**-0.5)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        输入:
            src: [batch_size, src_len] - 源序列的词ID
            src_mask: [batch_size, 1, 1, src_len] - 源序列掩码
            
        输出:
            [batch_size, src_len, d_model] - 编码后的序列
            
        数据流示例（文本编码）:
            输入句子ID: [101, 2023, 456, 102]  # 4个词的ID
            src: [32, 4]  # 32个样本，4个词
            
            步骤1: 词嵌入
                [32, 4] -> [32, 4, 512]
            
            步骤2: 缩放（为了和位置编码平衡）
                [32, 4, 512] * sqrt(512)
            
            步骤3: 位置编码
                [32, 4, 512] + 位置编码
            
            步骤4: 通过6个编码器层
                层1: [32, 4, 512] -> [32, 4, 512]
                层2: [32, 4, 512] -> [32, 4, 512]
                ...
                层6: [32, 4, 512] -> [32, 4, 512]
            
            输出: [32, 4, 512]
        """
        # 获取序列长度和批次大小
        batch_size, seq_len = src.shape
        
        # 步骤1: 词嵌入
        x = self.embedding(src)  # [batch_size, seq_len, d_model]
        
        # 步骤2: 缩放嵌入（Transformer论文中的技巧）
        x = x * math.sqrt(self.d_model)
        
        # 步骤3: 添加位置编码
        x = self.positional_encoding(x)
        
        # Dropout
        x = self.dropout(x)
        
        # 步骤4: 通过N个编码器层
        for i, layer in enumerate(self.layers):
            x = layer(x, src_mask)
            # print(f"编码器层 {i+1} 输出shape: {x.shape}")
        
        return x


# 测试完整编码器
def test_encoder():
    """测试Transformer编码器"""
    print("\n" + "="*50)
    print("测试Transformer编码器")
    print("="*50)
    
    # 参数设置
    vocab_size = 10000
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    n_layers = 6
    
    # 创建编码器
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    )
    
    # 创建输入（随机的词ID）
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入shape: {src.shape}")
    
    # 前向传播
    output = encoder(src)
    print(f"输出shape: {output.shape}")
    print(f"编码器测试通过！\n")
    
    return output

# 运行测试
# _ = test_encoder()

class TransformerDecoder(nn.Module):
    """
    完整的Transformer解码器
    
    包含:
    - 词嵌入层
    - 位置编码
    - N个解码器层的堆叠
    - 输出层（词汇表概率）
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        参数:
            vocab_size: 词汇表大小
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: 解码器层数
            d_ff: 前馈网络维度
            max_len: 最大序列长度
            dropout: Dropout概率
        """
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # 堆叠N个解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出层：将d_model维映射到vocab_size（预测下一个词）
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
        
        print(f"Transformer解码器初始化完成！")
        print(f"   词汇表大小: {vocab_size}")
        print(f"   解码器层数: {n_layers}")
        print(f"   模型维度: {d_model}")
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model**-0.5)
        nn.init.xavier_uniform_(self.output_projection.weight)
    
    def forward(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        输入:
            tgt: [batch_size, tgt_len] - 目标序列的词ID
            encoder_output: [batch_size, src_len, d_model] - 编码器输出
            src_mask: [batch_size, 1, 1, src_len] - 源序列掩码
            tgt_mask: [batch_size, 1, tgt_len, tgt_len] - 目标序列掩码
            
        输出:
            [batch_size, tgt_len, vocab_size] - 每个位置的词汇表概率
            
        数据流示例:
            源句子: "我爱北京" (已编码)
            目标句子: "I love Beijing"
            
            tgt: [32, 3]  # 32个样本，3个词
            encoder_output: [32, 4, 512]  # 编码器输出
            
            步骤1: 词嵌入 + 位置编码
                [32, 3] -> [32, 3, 512]
            
            步骤2: 通过6个解码器层
                每层都会关注编码器输出
            
            步骤3: 输出投影
                [32, 3, 512] -> [32, 3, vocab_size]
                得到每个位置的词汇表概率分布
        """
        # 获取目标序列长度
        batch_size, tgt_len = tgt.shape
        
        # 步骤1: 词嵌入
        x = self.embedding(tgt)  # [batch_size, tgt_len, d_model]
        
        # 步骤2: 缩放嵌入
        x = x * math.sqrt(self.d_model)
        
        # 步骤3: 添加位置编码
        x = self.positional_encoding(x)
        
        # Dropout
        x = self.dropout(x)
        
        # 步骤4: 通过N个解码器层
        for i, layer in enumerate(self.layers):
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # 步骤5: 输出投影（预测词汇表概率）
        output = self.output_projection(x)  # [batch_size, tgt_len, vocab_size]
        
        return output

class Transformer(nn.Module):
    """
    完整的Transformer模型
    
    组合了编码器和解码器，实现序列到序列的转换
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        参数:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型维度
            n_heads: 注意力头数
            n_encoder_layers: 编码器层数
            n_decoder_layers: 解码器层数
            d_ff: 前馈网络维度
            max_len: 最大序列长度
            dropout: Dropout概率
        """
        super(Transformer, self).__init__()
        
        # 编码器
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        # 解码器
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        print(f"完整Transformer模型初始化完成！")
        print(f"   源词汇表: {src_vocab_size}")
        print(f"   目标词汇表: {tgt_vocab_size}")
        print(f"   编码器层数: {n_encoder_layers}")
        print(f"   解码器层数: {n_decoder_layers}")
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        输入:
            src: [batch_size, src_len] - 源序列
            tgt: [batch_size, tgt_len] - 目标序列
            src_mask: [batch_size, 1, 1, src_len] - 源序列掩码
            tgt_mask: [batch_size, 1, tgt_len, tgt_len] - 目标序列掩码
            
        输出:
            [batch_size, tgt_len, tgt_vocab_size] - 预测的词汇表概率
            
        完整数据流示例（中英翻译）:
            任务: "我爱北京天安门" -> "I love Beijing Tiananmen"
            
            输入:
                src: [32, 7]  # "我爱北京天安门"的ID
                tgt: [32, 5]  # "I love Beijing Tiananmen"的ID
            
            步骤1: 编码器处理源序列
                [32, 7] -> [32, 7, 512]
            
            步骤2: 解码器生成目标序列
                解码器输入: [32, 5]
                编码器输出: [32, 7, 512]
                -> [32, 5, vocab_size]
            
            输出: 每个位置预测的词的概率分布
        """
        # 编码源序列
        encoder_output = self.encoder(src, src_mask)
        
        # 解码生成目标序列
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        return decoder_output
    
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 100,
        start_token: int = 2,  # <SOS> token
        end_token: int = 3      # <EOS> token
    ) -> torch.Tensor:
        """
        生成序列（推理时使用）
        
        输入:
            src: [batch_size, src_len] -- 源序列
            max_len: 最大生成长度
            start_token: 开始token的ID
            end_token: 结束token的ID
            
        输出:
            [batch_size, generated_len] -- 生成的序列
        """
        self.eval()
        
        batch_size = src.size(0)
        device = src.device
        
        # 编码源序列(没有掩码)
        encoder_output = self.encoder(src, None)
        
        # 初始化目标序列（以<SOS>开始）
        tgt = torch.full((batch_size, 1), start_token, device=device)
        
        for _ in range(max_len):
            # 创建目标掩码（下三角矩阵）tgt.size(1)就是现在目标序列的长度
            tgt_mask = self.create_tgt_mask(tgt.size(1)).to(device)
            
            # 解码
            output = self.decoder(tgt, encoder_output, None, tgt_mask)
            
            # 获取最后一个位置的预测（output的第二个维度是tgt_len)  next_token的维度应该是（batch_size,1)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # 拼接到目标序列
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 检查是否所有样本都生成了<EOS>
            if (next_token == end_token).all():
                break
        
        return tgt
    
    @staticmethod
    def create_tgt_mask(tgt_len: int) -> torch.Tensor:
        """
        创建目标序列的掩码（下三角矩阵）
        防止解码器看到未来的信息
        
        示例:
            tgt_len = 4
            mask = [[1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 1]]
        """
        mask = torch.tril(torch.ones(tgt_len, tgt_len))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]


# 测试完整Transformer
def test_transformer():
    """测试完整的Transformer模型"""
    print("\n" + "="*50)
    print("测试完整Transformer模型")
    print("="*50)
    
    # 参数设置
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    batch_size = 2
    src_len = 10
    tgt_len = 12
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6
    )
    
    # 创建输入
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    print(f"源序列shape: {src.shape}")
    print(f"目标序列shape: {tgt.shape}")
    
    # 创建掩码
    tgt_mask = Transformer.create_tgt_mask(tgt_len)
    
    # 前向传播
    output = model(src, tgt, tgt_mask=tgt_mask)
    print(f"输出shape: {output.shape}")
    print(f"Transformer测试通过！\n")
    
    # 测试生成
    print("测试生成功能...")
    generated = model.generate(src, max_len=20)
    print(f"生成序列shape: {generated.shape}")
    print(f"生成测试通过！\n")
    
    return output

# 运行测试
_ = test_transformer()
