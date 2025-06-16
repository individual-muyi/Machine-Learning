# 计算注意力分数
state_expanded = tf.expand_dims(state, 1)  # [batch, 1, hidden]
attn_scores = tf.matmul(enc_out, tf.expand_dims(self.dense_attn(state), -1), 
                 axes=[2, 2])  # [batch, enc_len, 1]

# 计算注意力权重
attn_weights = tf.nn.softmax(attn_scores, axis=1)  # [batch, enc_len, 1]

# 计算上下文向量
context_vector = tf.reduce_sum(enc_out * attn_weights, axis=1)  # [batch, hidden]