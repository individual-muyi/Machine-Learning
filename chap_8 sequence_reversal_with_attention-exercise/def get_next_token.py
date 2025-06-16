def get_next_token(self, x, state, enc_out):
    # 运行decoder cell
    output, state = self.decoder_cell(x_emb, state)
    
    # 计算注意力
    # ...（同上）
    
    # 合并上下文向量
    combined = tf.concat([output, context_vector], axis=-1)
    
    # 预测下一个token
    logits = self.dense(combined)
    out = tf.argmax(logits, axis=-1)
    return out, state