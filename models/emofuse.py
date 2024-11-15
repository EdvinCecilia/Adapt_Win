import torch
import torch.nn as nn
import numpy as np
from transformers.models.bert.modeling_bert import BertPooler, BertSelfAttention
from torch.nn import functional as F

# Self-Attention 模块，继承自 PyTorch 的 nn.Module
class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        """
        初始化 Self-Attention 模块。
        :param config: BERT 的配置文件。
        :param opt: 用户自定义的参数字典。
        """
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)  # 使用 BERT 的自带注意力机制
        self.tanh = torch.nn.Tanh()  # 激活函数，限制输出范围

    def forward(self, inputs):
        """
        前向传播方法。
        :param inputs: 输入的张量，形状为 (batch_size, seq_len, hidden_dim)。
        :return: 使用 Tanh 激活函数后的注意力输出。
        """
        zero_tensor = torch.zeros((inputs.size(0), 1, 1, self.opt['max_seq_len']),
                                  dtype=torch.float32).to(self.opt['device'])  # 初始化全零注意力掩码
        SA_out = self.SA(inputs, zero_tensor)  # 计算自注意力输出
        return self.tanh(SA_out[0])  # 返回激活后的输出

# VAE 模型，继承自 PyTorch 的 nn.Module
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        """
        初始化变分自编码器（VAE）。
        :param input_dim: 输入特征维度。
        :param latent_dim: 潜变量维度。
        """
        super(VAE, self).__init__()
        # 编码器部分
        self.encoder_lstm = nn.LSTM(input_dim, 256, batch_first=True, bidirectional=True)  # 双向 LSTM
        self.fc_mu = nn.Linear(256 * 2, latent_dim)  # 生成均值
        self.fc_logvar = nn.Linear(256 * 2, latent_dim)  # 生成对数方差

        # 解码器部分
        self.decoder_input = nn.Linear(latent_dim, 256 * 2)  # 将潜变量解码为高维特征
        self.decoder_lstm = nn.LSTM(256 * 2, input_dim, batch_first=True)  # 单层 LSTM 解码

    def encode(self, x):
        """
        编码器：提取潜变量的均值和对数方差。
        :param x: 输入数据，形状为 (batch_size, seq_len, input_dim)。
        :return: 均值 (mu) 和对数方差 (logvar)。
        """
        _, (h_n, _) = self.encoder_lstm(x)  # 提取最后一层的隐藏状态
        h_n = h_n.permute(1, 0, 2).contiguous().view(x.size(0), -1)  # 拼接双向隐藏状态
        mu = self.fc_mu(h_n)  # 计算均值
        logvar = self.fc_logvar(h_n)  # 计算对数方差
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        重参数化：通过随机采样生成潜变量。
        :param mu: 均值张量。
        :param logvar: 对数方差张量。
        :return: 潜变量。
        """
        std = torch.exp(0.5 * logvar)  # 方差的平方根（标准差）
        eps = torch.randn_like(std)  # 生成标准正态分布噪声
        return mu + eps * std  # 使用重参数化公式生成潜变量

    def decode(self, z, seq_len):
        """
        解码器：从潜变量生成重构数据。
        :param z: 潜变量张量，形状为 (batch_size, latent_dim)。
        :param seq_len: 序列长度。
        :return: 重构数据，形状为 (batch_size, seq_len, input_dim)。
        """
        decoder_input = self.decoder_input(z).unsqueeze(1).repeat(1, seq_len, 1)  # 将潜变量扩展到序列维度
        recon_x, _ = self.decoder_lstm(decoder_input)  # 使用 LSTM 解码
        return recon_x

    def forward(self, x):
        """
        前向传播方法。
        :param x: 输入数据，形状为 (batch_size, seq_len, input_dim)。
        :return: 重构数据、均值、对数方差以及潜变量。
        """
        batch_size, seq_len, input_dim = x.size()
        mu, logvar = self.encode(x)  # 编码
        z = self.reparameterize(mu, logvar)  # 重参数化
        recon_x = self.decode(z, seq_len)  # 解码
        return recon_x, mu, logvar, z  # 返回所有中间结果

# 主模型 AdaptWin，继承自 nn.Module
class AdaptWin(nn.Module):
    def __init__(self, bert, opt):
        """
        初始化 AdaptWin 模型。
        :param bert: 预训练的 BERT 模型。
        :param opt: 参数字典。
        """
        super(AdaptWin, self).__init__()

        self.bert_spc = bert  # BERT 模块
        self.opt = opt
        self.dropout = nn.Dropout(opt['dropout'])  # Dropout 正则化

        # 自定义注意力机制
        self.bert_SA = SelfAttention(bert.config, opt)

        # 线性层，用于特征压缩
        combined_dim = opt['bert_dim'] * 2 + opt['latent_dim']
        self.linear_double = nn.Linear(combined_dim, opt['bert_dim'])
        self.linear_single = nn.Linear(opt['bert_dim'], opt['bert_dim'])

        # BERT 的池化层
        self.bert_pooler = BertPooler(bert.config)

        # 分类器
        self.dense = nn.Linear(opt['bert_dim'], opt['polarities_dim'])

        # 平均池化
        self.pool = nn.AvgPool1d(opt['threshold'] + 1)

        # 参数 alpha，用于自适应加权
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 初始化为 1.0

        # VAE 模块
        self.vae = VAE(input_dim=opt['bert_dim'], latent_dim=opt['latent_dim'])

    def moving_mask(self, bert_output, aspect_indices, mask_len):
        """
        创建滑动窗口掩码，屏蔽不需要的部分。
        :param bert_output: BERT 的输出张量。
        :param aspect_indices: aspect 索引张量。
        :param mask_len: 滑动窗口的长度。
        :return: 滑动窗口掩码。
        """
        masked_text_raw_indices = torch.ones_like(bert_output)

        for batch_idx in range(bert_output.size(0)):
            asp_start = aspect_indices[batch_idx].nonzero(as_tuple=True)[0][0]  # 获取 aspect 起始位置
            asp_len = (aspect_indices[batch_idx] != 0).sum().item()  # 计算 aspect 长度

            mask_start = max(asp_start - mask_len, 0)
            mask_end = min(asp_start + asp_len + mask_len, self.opt['max_seq_len'])

            # 在窗口外的部分置 0
            masked_text_raw_indices[batch_idx, :mask_start] = 0
            masked_text_raw_indices[batch_idx, mask_end:] = 0

        return masked_text_raw_indices

    def forward(self, inputs, vae_only=False):
        """
        前向传播方法。
        :param inputs: 输入张量元组。
        :param vae_only: 是否仅运行 VAE 部分。
        :return: 分类结果或 VAE 输出。
        """
        # 解包输入
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2]
        aspect_indices = inputs[3]

        # BERT 编码器输出
        bert_spc_out, _ = self.bert_spc(
            text_bert_indices,
            token_type_ids=bert_segments_ids,
            return_dict=False
        )
        bert_spc_out = self.dropout(bert_spc_out)

        if vae_only:
            # 使用 VAE
            recon_x, mu, logvar, z = self.vae(bert_spc_out)
            return recon_x, mu, logvar, z
        else:
            # 邻近区域特征
            neighboring_span, _ = self.bert_spc(text_local_indices, return_dict=False)
            neighboring_span = self.dropout(neighboring_span)

            # 计算注意力权重
            attention_weights = ...
            neighboring_span_weighted = neighboring_span * attention_weights

            # 拼接
            enhanced_text = torch.cat((neighboring_span_weighted, bert_spc_out), dim=-1)

            # 经过线性层和自注意力模块
            mean_pool = self.linear_double(enhanced_text)
            self_attention_out = self.bert_SA(mean_pool)

            # 池化并分类
            pooled_out = self.bert_pooler(self_attention_out)
            logits = self.dense(pooled_out)

            return logits, pooled_out
