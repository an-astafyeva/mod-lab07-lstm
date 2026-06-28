import random
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop

# 1. Загрузка текста
with open('src/input.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

# 2. Токенизация по словам
words = text.split()

print(f"Всего слов: {len(words)}")

# 3. Словарь
vocabulary = sorted(list(set(words)))

word_to_indices = dict((w, i) for i, w in enumerate(vocabulary))
indices_to_word = dict((i, w) for i, w in enumerate(vocabulary))

print(f"Размер словаря: {len(vocabulary)}")

# 4. Цепочки слов (скользящее окно)
max_length = 5
step = 1
sentences = []
next_words = []

for i in range(0, len(words) - max_length, step):
    sentences.append(words[i: i + max_length])
    next_words.append(words[i + max_length])

print(f"Обучающих примеров: {len(sentences)}")

# 5. One-hot векторизация
X = np.zeros(
    (len(sentences), max_length, len(vocabulary)),
    dtype=bool)

y = np.zeros(
    (len(sentences), len(vocabulary)),
    dtype=bool)

for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_to_indices[word]] = 1

    y[i, word_to_indices[next_words[i]]] = 1

print(f"Форма X: {X.shape}, форма y: {y.shape}")

# 6. Построение LSTM-модели
model = Sequential()

model.add(
    LSTM(
        128,
        input_shape=(max_length, len(vocabulary))
    )
)

model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer
)

model.summary()

# 7. Функция температуры (sampling)
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')

    preds = np.log(preds + 1e-8) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)

# 8. Обучение
print("\nНачало обучения...")
model.fit(X, y, batch_size=128, epochs=50)
print("Обучение завершено.")

# 9. Функция генерации текста
def generate_text(length, diversity):
    start_index = random.randint(
        0,
        len(words) - max_length - 1
    )

    sentence = words[
        start_index:start_index + max_length
    ]

    generated = ' '.join(sentence)

    for _ in range(length):
        x_pred = np.zeros(
            (1, max_length, len(vocabulary))
        )

        for t, word in enumerate(sentence):
            x_pred[0, t, word_to_indices[word]] = 1.

        preds = model.predict(
            x_pred,
            verbose=0
        )[0]

        next_index = sample(preds, diversity)

        next_word = indices_to_word[next_index]

        generated += ' ' + next_word

        sentence = sentence[1:] + [next_word]

    return generated

# 10. Генерация и сохранение
print("\nГенерация текста (температура=0.2)...")
generated_text = generate_text(1000, 0.2)

print("\n" + "="*60)
print(generated_text)
print("="*60)

with open(
    'result/gen.txt',
    'w',
    encoding='utf-8'
) as file:
    file.write(generated_text)

print("\nРезультат сохранён в result/gen.txt")
