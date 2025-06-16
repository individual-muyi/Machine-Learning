# 合并上下文向量和decoder输出
combined = tf.concat([output, context_vector], axis=-1)

# 输出层
logits_t = self.dense(combined)  # [batch, vocab_size]