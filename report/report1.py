# 필요한 라이브러리 불러오기
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# 0: 티셔츠/탑, 1: 바지, 2: 풀오버, 3: 드레스, 4: 코트
# 5: 샌들, 6: 셔츠, 7: 스니커즈, 8: 가방, 9: 앵클 부츠
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# 1. 데이터 로드
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 데이터 확인
print("훈련 데이터 이미지 shape:", train_images.shape)
print("훈련 데이터 라벨 shape:", train_labels.shape)
print("테스트 데이터 이미지 shape:", test_images.shape)
print("테스트 데이터 라벨 shape:", test_labels.shape)

# 첫 번째 이미지 출력
plt.figure(figsize=(4, 4))
plt.imshow(train_images[0], cmap='gray')
plt.title(f"첫 번째 훈련 이미지 / 라벨: {class_names[train_labels[0]]}")
plt.axis('off')
plt.show()

# 1. 데이터 전처리
# 픽셀 값을 0~255 범위에서 0~1 범위로 정규화
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# CNN 입력 형태에 맞게 채널 차원 추가
# (60000, 28, 28) -> (60000, 28, 28, 1)
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

print("전처리 후 훈련 데이터 shape:", train_images.shape)
print("전처리 후 테스트 데이터 shape:", test_images.shape)

# 2. CNN 모델 구성
model = models.Sequential([
    # 입력층
    layers.Input(shape=(28, 28, 1)),

    # 첫 번째 합성곱 층 + 풀링 층
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 두 번째 합성곱 층 + 풀링 층
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 2차원 특징맵을 1차원 벡터로 변환
    layers.Flatten(),

    # 완전 연결 층
    layers.Dense(128, activation='relu'),

    # 출력층 (10개 클래스 분류)
    layers.Dense(10, activation='softmax')
])

# 모델 구조 출력
model.summary()

# 3. 모델 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. 모델 훈련
history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

# 5. 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("\n테스트 손실:", test_loss)
print("테스트 정확도:", test_acc)

# 6. 예측 및 시각화
predictions = model.predict(test_images)

# 예측 결과 하나 확인
index = 0
predicted_label = np.argmax(predictions[index])
true_label = test_labels[index]

plt.figure(figsize=(4, 4))
plt.imshow(test_images[index].squeeze(), cmap='gray')
plt.title(f"예측: {class_names[predicted_label]} / 실제: {class_names[true_label]}")
plt.axis('off')
plt.show()

print("예측 확률:", predictions[index])
print("예측 클래스:", class_names[predicted_label])
print("실제 클래스:", class_names[true_label])