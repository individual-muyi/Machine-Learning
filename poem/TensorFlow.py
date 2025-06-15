import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt
import random
import time
from IPython.display import clear_output

# 设置随机种子以确保结果可重现
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# 生成示例唐诗数据集
def generate_sample_poems(num_poems=500):
    """生成示例唐诗数据集"""
    templates = [
        "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
        "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
        "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
        "空山新雨后，天气晚来秋。明月松间照，清泉石上流。",
        "海上生明月，天涯共此时。情人怨遥夜，竟夕起相思。",
        "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
        "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
        "湖光秋月两相和，潭面无风镜未磨。遥望洞庭山水翠，白银盘里一青螺。",
        "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。"
    ]
    
    variations = [
        ("春", "夏", "秋", "冬"),
        ("山", "水", "风", "云"),
        ("花", "草", "树", "木"),
        ("日", "月", "星", "辰"),
        ("红", "黄", "蓝", "绿"),
        ("江", "河", "湖", "海"),
        ("夜", "昼", "晨", "暮")
    ]
    
    poems = []
    for _ in range(num_poems):
        poem = random.choice(templates)
        # 添加一些变化
        for i in range(random.randint(1, 3)):
            old, new = random.choice(variations)
            poem = poem.replace(random.choice(old), random.choice(new), 1)
        poems.append(poem)
    
    return poems

# 创建数据集
poems = generate_sample_poems(500)
print(f"共生成 {len(poems)} 首唐诗示例")
print("\n前5首示例:")
for i in range(5):
    print(f"{i+1}. {poems[i]}")

# 文本预处理
all_text = ''.join(poems)
chars = sorted(set(all_text))
print(f"\n总字符数: {len(all_text)}")
print(f"唯一字符数: {len(chars)}")
print(f"字符集: {''.join(chars)}")

# 创建字符到索引的映射
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

# 将文本转换为索引序列
sequences = []
next_chars = []
max_sequence_length = 20

for poem in poems:
    for i in range(0, len(poem) - max_sequence_length, 3):
        sequences.append(poem[i:i + max_sequence_length])
        next_chars.append(poem[i + max_sequence_length])

print(f"\n创建了 {len(sequences)} 个训练样本")

# 创建训练数据
X = np.zeros((len(sequences), max_sequence_length), dtype=np.int32)
y = np.zeros((len(sequences),), dtype=np.int32)

for i, seq in enumerate(sequences):
    for t, char in enumerate(seq):
        X[i, t] = char_to_idx[char]
    y[i] = char_to_idx[next_chars[i]]

# 构建RNN模型
vocab_size = len(chars)
embedding_dim = 128
lstm_units = 256

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
              input_length=max_sequence_length),
    LSTM(lstm_units, return_sequences=True),
    LSTM(lstm_units),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 训练模型（简化版，实际应用需要更多epoch）
history = model.fit(X, y, batch_size=128, epochs=20, validation_split=0.2)

# 绘制训练过程
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('训练和验证损失')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('训练和验证准确率')
plt.legend()
plt.tight_layout()
plt.show()

# 诗歌生成函数
def generate_poem(seed_text, num_chars=50, temperature=0.7):
    """生成唐诗"""
    generated = seed_text
    sys.stdout.write(seed_text)
    
    for i in range(num_chars):
        # 准备输入序列
        x_pred = np.zeros((1, max_sequence_length))
        for t, char in enumerate(seed_text[-max_sequence_length:]):
            if char in char_to_idx:
                x_pred[0, max_sequence_length - len(seed_text) + t] = char_to_idx[char]
        
        # 生成预测
        preds = model.predict(x_pred, verbose=0)[0]
        
        # 应用温度采样
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        # 从预测分布中采样下一个字符
        next_idx = np.random.choice(range(vocab_size), p=preds)
        next_char = idx_to_char[next_idx]
        
        # 添加到生成文本
        generated += next_char
        seed_text = seed_text[-max_sequence_length:] + next_char
        
        # 打印字符（模拟打字效果）
        sys.stdout.write(next_char)
        sys.stdout.flush()
        time.sleep(0.05)
    
    return generated

# 生成不同开头的诗歌
start_chars = ['日', '红', '山', '夜', '湖', '海', '月']

print("\n\n===== 唐诗生成演示 =====")
print("使用循环神经网络(RNN)生成唐诗")
print("模型结构：嵌入层 -> LSTM层 -> LSTM层 -> 输出层")
print(f"词汇表大小: {vocab_size}, 序列长度: {max_sequence_length}")
print("="*40 + "\n")

for char in start_chars:
    print(f"\n生成以 '{char}' 开头的唐诗:")
    poem = generate_poem(seed_text=char, num_chars=random.randint(24, 48))
    print("\n\n" + "="*60)
    time.sleep(1.5)