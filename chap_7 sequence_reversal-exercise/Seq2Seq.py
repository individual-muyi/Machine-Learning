import numpy as np
import tensorflow as tf
import random
import string
from tensorflow.keras import optimizers

# 数据生成函数
def randomString(stringLength):
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def get_batch(batch_size, length):
    batched_examples = [randomString(length) for i in range(batch_size)]
    enc_x = [[ord(ch)-ord('A')+1 for ch in list(exp)] for exp in batched_examples]
    y = [[o for o in reversed(e_idx)] for e_idx in enc_x]
    dec_x = [[0]+e_idx[:-1] for e_idx in y]
    return (batched_examples, tf.constant(enc_x, dtype=tf.int32), 
            tf.constant(dec_x, dtype=tf.int32), tf.constant(y, dtype=tf.int32))

# Seq2Seq模型定义
class mySeq2SeqModel(tf.keras.Model):
    def __init__(self):
        super(mySeq2SeqModel, self).__init__()
        self.v_sz = 27  # 26字母+1(padding)
        self.embed_layer = tf.keras.layers.Embedding(self.v_sz, 64)
        
        # Encoder部分
        self.encoder_cell = tf.keras.layers.SimpleRNNCell(128)
        self.encoder = tf.keras.layers.RNN(self.encoder_cell, return_state=True)
        
        # Decoder部分
        self.decoder_cell = tf.keras.layers.SimpleRNNCell(128)
        self.dense = tf.keras.layers.Dense(self.v_sz)
    
    def call(self, enc_ids, dec_ids):
        # Encoder处理
        enc_emb = self.embed_layer(enc_ids)  # [batch, seq_len, emb_dim]
        _, enc_state = self.encoder(enc_emb)  # enc_state: [batch, hidden_dim]
        
        # Decoder处理
        dec_emb = self.embed_layer(dec_ids)  # [batch, seq_len, emb_dim]
        dec_outputs, _ = tf.keras.layers.RNN(
            self.decoder_cell, return_sequences=True, return_state=True)(dec_emb, initial_state=enc_state)
        
        # 输出层
        logits = self.dense(dec_outputs)  # [batch, seq_len, vocab_size]
        return logits
    
    def encode(self, enc_ids):
        enc_emb = self.embed_layer(enc_ids)
        _, enc_state = self.encoder(enc_emb)
        return enc_state
    
    def get_next_token(self, x, state):
        inp_emb = self.embed_layer(x)  # [batch, emb_dim]
        h, state = self.decoder_cell(inp_emb, state)  # [batch, hidden_dim]
        logits = self.dense(h)  # [batch, vocab_size]
        out = tf.argmax(logits, axis=-1)  # [batch]
        return out, state

# 训练函数
@tf.function
def compute_loss(logits, labels):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    return tf.reduce_mean(losses)

@tf.function
def train_one_step(model, optimizer, enc_x, dec_x, y):
    with tf.GradientTape() as tape:
        logits = model(enc_x, dec_x)
        loss = compute_loss(logits, y)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train(model, optimizer, seqlen=10, steps=3000):
    for step in range(steps):
        _, enc_x, dec_x, y = get_batch(32, seqlen)
        loss = train_one_step(model, optimizer, enc_x, dec_x, y)
        if step % 500 == 0:
            print(f'step {step}: loss {loss.numpy()}')

# 测试函数
def sequence_reversal(model, seq_len=10):
    def decode(init_state, steps):
        b_sz = tf.shape(init_state)[0]
        cur_token = tf.zeros(shape=[b_sz], dtype=tf.int32)
        state = init_state
        collect = []
        
        for _ in range(steps):
            cur_token, state = model.get_next_token(cur_token, state)
            collect.append(tf.expand_dims(cur_token, axis=-1))
        
        out = tf.concat(collect, axis=-1).numpy()
        return [''.join([chr(idx+ord('A')-1) for idx in exp]) for exp in out]
    
    batched_examples, enc_x, _, _ = get_batch(32, seq_len)
    state = model.encode(enc_x)
    return decode(state, seq_len), batched_examples

def is_reverse(seq, rev_seq):
    return seq == ''.join(reversed(rev_seq))

# 主程序
if __name__ == "__main__":
    # 初始化模型和优化器
    model = mySeq2SeqModel()
    optimizer = optimizers.Adam(0.0005)
    
    # 训练模型
    print("开始训练...")
    train(model, optimizer, seqlen=10)
    
    # 测试模型
    print("\n测试结果:")
    reversed_seqs, original_seqs = sequence_reversal(model)
    
    for orig, rev in zip(original_seqs, reversed_seqs):
        print(f"原始: {orig} -> 逆置: {rev} | {'正确' if is_reverse(orig, rev) else '错误'}")
    
    # 计算准确率
    accuracy = np.mean([is_reverse(o, r) for o, r in zip(original_seqs, reversed_seqs)])
    print(f"\n准确率: {accuracy*100:.2f}%")