#  Проекты по глубокому обучению (Deep Learning)

Репозиторий содержит учебные и исследовательские проекты, посвящённые применению **глубокого обучения (Deep Learning)**  
для задач классификации изображений и текстов. Все работы выполнены в среде **Python / PyTorch** с акцентом на  
структуру экспериментов, визуализацию метрик и анализ результатов.

---

##  Структура репозитория

| Проект | Описание |
|--------|-----------|
| [**Image_Classification_using_AlexNet**](./Image_Classification_using_AlexNet) | Реализация архитектуры **AlexNet** с нуля на PyTorch. Изучение свёрточных слоёв, активаций ReLU, Dropout и оптимизации. |
| [**CIFAR-10_Image_Classification_using_CNNs**](./CIFAR-10_Image_Classification_using_CNNs) | Классическая свёрточная сеть для классификации изображений CIFAR-10. Аугментации, BatchNorm, Dropout, визуализация обучения. |
| [**Fine-Tuning_and_Transfer_Learning_with_ResNet50V2**](./Fine-Tuning_and_Transfer_Learning_with_ResNet50V2) | Применение **Transfer Learning**: обучение “головы” и поэтапный Fine-Tuning предобученной **ResNet50V2**. |
| [**Text_Classification_using_RNN_and_Transformer_Models**](./Text_Classification_using_RNN_and_Transformer_Models) | Классификация текстов с использованием рекуррентных сетей (**BiGRU+Attention**) и моделей **Transformer**. |

---

##  Основные темы и навыки

- **Свёрточные нейросети (CNN):** Conv2d, Pooling, BatchNorm, Dropout, ReLU  
- **Transfer Learning:** заморозка слоёв, обучение классификатора, Fine-Tuning  
- **Оптимизация и регуляризация:** AdamW, StepLR, EarlyStopping, HistoryRecorder  
- **Рекуррентные сети и Attention:** GRU, двунаправленные RNN, Attention-механизмы  
- **Transformer-архитектуры:** Encoder-блоки, позиционные эмбеддинги, Multi-Head Attention  
- **Визуализация:** графики метрик, confusion matrix, классификационные отчёты

---

##  Используемые технологии

- **Язык:** Python 3.10+  
- **Фреймворки:** PyTorch, Torchvision  
- **Библиотеки:** NumPy, Pandas, scikit-learn, Matplotlib, tqdm, NLTK  
- **Среда:** Jupyter Notebook  

---

##  Ключевые результаты

| Проект | Val Accuracy | Macro F1 | Комментарий |
|--------|---------------|-----------|--------------|
| AlexNet | ~0.72 | ~0.70 | Собственная CNN-архитектура, устойчивая к переобучению |
| CIFAR-10 CNN | ~0.68 | ~0.67 | Базовая модель без предобученных весов |
| ResNet50V2 (Transfer Learning) | **0.93** | **0.92** | Fine-Tuning предобученной модели дал прирост +25 % к baseline |
| RNN / Transformer | ~0.87 | ~0.85 | Классификация текстов с Attention и трансформерами |
