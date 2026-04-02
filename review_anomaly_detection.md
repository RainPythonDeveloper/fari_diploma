# Экспертный обзор пайплайна обнаружения аномальных транзакций

> Ревью подготовлено с позиции специалиста по fraud detection.
> Дата: 29.03.2026

---

## Оглавление
1. [Общая оценка](#общая-оценка)
2. [Критические проблемы](#1-критические-проблемы)
3. [Серьезные проблемы](#2-серьезные-проблемы)
4. [Методологические замечания](#3-методологические-замечания)
5. [Что сделано хорошо](#4-что-сделано-хорошо)
6. [Приоритеты исправлений](#5-приоритеты-исправлений)

---

## Общая оценка

Пайплайн имеет правильную базовую структуру (загрузка → очистка → preprocessing → обучение → оценка), но содержит ряд критических и серьезных проблем, которые влияют на достоверность результатов. Лучшая модель (LOF, F1=0.55) показывает приемлемый для unsupervised подхода результат, однако без supervised baseline и ряда исправлений выводы дипломной работы будут уязвимы для критики.

---

## 1. КРИТИЧЕСКИЕ ПРОБЛЕМЫ

### 1.1 NaN в признаках не обрабатываются

**Проблема:**
Столбцы V16–V28 и Amount содержат по 1 пропущенному значению (видно в выводе `df.isnull().sum()`). Код удаляет NaN только в столбце Class:
```python
df = df.dropna(subset=['Class']).reset_index(drop=True)
```
Но NaN в признаках **остаются**. `StandardScaler` пропускает NaN дальше, и модели получают данные с пропусками. Это приводит к непредсказуемому поведению — некоторые модели sklearn молча игнорируют NaN, другие ломаются.

**Решение:**
```python
# Вариант 1: удалить все строки с NaN (потеряем ~1 строку)
df = df.dropna().reset_index(drop=True)

# Вариант 2: заполнить медианой (если пропусков больше)
df = df.fillna(df.median(numeric_only=True))
```

---

### 1.2 Датасет урезан наполовину без объяснения

**Проблема:**
Оригинальный Kaggle creditcard.csv содержит **284,807 транзакций** и **492 фрода**. В ноутбуке загружается **128,821 строк** и **261 фрод**. Половина данных (включая ~231 мошенническую транзакцию) отсутствует.

Это критично:
- Теряются паттерны мошенничества, которые могли быть в потерянной половине
- Уменьшается и без того крошечный класс fraud
- Результаты нельзя сравнивать с публикациями, использующими полный датасет

**Решение:**
- Если это намеренная выборка — **явно описать метод** выборки и обосновать
- Если ошибка при загрузке — загрузить полный датасет
- В дипломе обязательно указать размер используемого датасета и причину отличия от оригинала

---

### 1.3 Случайный train/test split вместо временного — утечка данных (data leakage)

**Проблема:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
Случайный split перемешивает транзакции из разных временных периодов. Модель обучается на транзакциях из «будущего» и тестируется на транзакциях из «прошлого». В реальном мире это невозможно — мы всегда предсказываем будущие транзакции на основе прошлых.

Мошенники **адаптируются со временем**: схемы меняются, появляются новые паттерны. Случайный split маскирует эту проблему и завышает метрики.

**Решение:**
```python
# Сортируем по времени и делим хронологически
df = df.sort_values('Time').reset_index(drop=True)
split_point = int(len(df) * 0.8)

train_df = df.iloc[:split_point]
test_df = df.iloc[split_point:]

X_train = train_df.drop('Class', axis=1)
y_train = train_df['Class']
X_test = test_df.drop('Class', axis=1)
y_test = test_df['Class']
```

---

### 1.4 Supervised-метки есть, но не используются

**Проблема:**
Датасет содержит размеченный столбец `Class` (0/1), но все 6 моделей — unsupervised/semi-supervised. Это принципиальная потеря информации. Supervised модели на этом датасете обычно дают **F1 > 0.80** (vs. ваш лучший unsupervised F1 = 0.55).

Без supervised baseline нельзя ответить на вопрос: **«Оправдан ли unsupervised подход?»**

**Решение — добавить минимум 2 supervised модели:**

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',  # автоматически балансирует классы
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
y_score_rf = rf.predict_proba(X_test_scaled)[:, 1]
evaluate_predictions("Random Forest", y_test, y_pred_rf, y_score_rf)

# XGBoost
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=200,
    scale_pos_weight=scale_pos,  # компенсация дисбаланса
    eval_metric='aucpr',
    random_state=42,
    use_label_encoder=False
)
xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)
y_score_xgb = xgb.predict_proba(X_test_scaled)[:, 1]
evaluate_predictions("XGBoost", y_test, y_pred_xgb, y_score_xgb)
```

Это позволит в дипломе написать: *«Unsupervised подходы уступают supervised по F1-score на X%, однако имеют преимущество в сценариях, где разметка недоступна»* — это сильный вывод.

---

## 2. СЕРЬЕЗНЫЕ ПРОБЛЕМЫ

### 2.1 Elliptic Envelope — полный провал (Precision=0, Recall=0, F1=0)

**Проблема:**
Модель нашла **0 мошеннических транзакций** (TP=0), но пометила 65 нормальных как фрод (FP=65). Она работает **хуже случайного угадывания**.

**Причина:** Elliptic Envelope предполагает, что данные подчиняются **многомерному нормальному распределению** (Гауссову). PCA-компоненты кредитных транзакций этому распределению не подчиняются — у них тяжёлые хвосты и мультимодальность.

**Решение:**
```python
# Проверить распределение перед использованием:
from scipy.stats import shapiro
for col in ['V1', 'V2', 'V3']:
    stat, p = shapiro(X_train_normal[:5000, X.columns.get_loc(col)])
    print(f"{col}: p-value = {p:.6f}")
# Если p-value < 0.05, данные НЕ нормальные → Elliptic Envelope неприменим
```
В дипломе: указать это как **ограничение метода** и объяснить причину провала, а не просто показывать нули.

---

### 2.2 One-Class SVM — 774 ложных срабатывания

**Проблема:**
Precision = 0.0515 означает: из каждых **20 заблокированных** транзакций **19 — легитимные**. В реальной системе это приведёт к массовым жалобам клиентов и блокировке нормальных операций.

**Причины:**
- Обучение всего на 20,000 примерах (из 102,000 доступных)
- RBF-ядро чувствительно к масштабу и `gamma`
- `nu=contamination` может быть слишком мал

**Решение:**
```python
# 1. Увеличить обучающую выборку
X_train_normal_small_svm = sample_rows(X_train_normal, 50000)

# 2. Подобрать гиперпараметры
from sklearn.model_selection import ParameterGrid

params = {'nu': [0.001, 0.005, 0.01], 'gamma': ['scale', 'auto', 0.01, 0.001]}
best_f1 = 0
for p in ParameterGrid(params):
    svm = OneClassSVM(kernel='rbf', **p)
    svm.fit(X_train_normal_small_svm)
    preds = np.where(svm.predict(X_test_scaled) == -1, 1, 0)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_params = p
print(f"Best params: {best_params}, F1: {best_f1:.4f}")
```

---

### 2.3 Autoencoder недообучен

**Проблема:**
Loss снижается **все 15 эпох** (с 0.856 до 0.423). EarlyStopping (patience=3) **ни разу не сработал**, потому что val_loss монотонно уменьшался. Это значит модель **не вышла на плато** и ей нужно больше эпох.

**Решение:**
```python
# Увеличить эпохи и patience
history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=100,        # было 15
    batch_size=256,
    validation_split=0.2,
    shuffle=True,
    callbacks=[EarlyStopping(
        monitor='val_loss',
        patience=10,    # было 3
        restore_best_weights=True
    )],
    verbose=1
)
```
Также уменьшить learning rate при plateau:
```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
)
# Добавить в callbacks=[early_stop, reduce_lr]
```

---

### 2.4 Архитектура Autoencoder слишком простая

**Проблема:**
Текущая архитектура 30 → 16 → 8 → 16 → 30 (всего 1,286 параметров) слишком мала для захвата сложных паттернов нормальных транзакций.

**Решение — улучшенная архитектура:**
```python
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

input_dim = X_train_normal.shape[1]
input_layer = Input(shape=(input_dim,))

# Encoder
x = Dense(64, activation='relu')(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
encoded = Dense(16, activation='relu')(x)

# Decoder
x = Dense(32, activation='relu')(encoded)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
decoded = Dense(input_dim, activation='linear')(x)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
```

---

### 2.5 Нет feature engineering

**Проблема:**
Признаки используются «как есть» без каких-либо преобразований. В индустрии fraud detection feature engineering — ключевой этап, который часто важнее выбора модели.

**Решения:**

```python
# 1. Логарифм суммы — фрод часто на экстремальных суммах
df['Amount_log'] = np.log1p(df['Amount'])

# 2. Время суток — фрод чаще ночью
df['Hour'] = (df['Time'] % 86400) / 3600  # секунды → часы
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

# 3. Статистические признаки суммы (если есть группировка по пользователю)
# В данном датасете нет user_id, но для реального проекта:
# df['Amount_zscore'] = df.groupby('user_id')['Amount'].transform(
#     lambda x: (x - x.mean()) / x.std()
# )

# 4. Взаимодействия признаков
df['V1_V2'] = df['V1'] * df['V2']
df['Amount_V1'] = df['Amount_log'] * df['V1']
```

---

### 2.6 В тестовой выборке всего 51 мошенническая транзакция

**Проблема:**
Каждая транзакция = ~2% recall. Разница в 1 пойманном фроде меняет recall на 2%. Метрики **статистически ненадёжны** при таком размере.

**Решение — использовать Stratified K-Fold Cross-Validation:**
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []

for train_idx, test_idx in skf.split(X_scaled, y):
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    X_tr_normal = X_tr[y_tr == 0]

    model = LocalOutlierFactor(n_neighbors=20, contamination=0.002, novelty=True)
    model.fit(X_tr_normal)

    preds = np.where(model.predict(X_te) == -1, 1, 0)
    f1_scores.append(f1_score(y_te, preds))

print(f"F1: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
```
Это даст **среднее и стандартное отклонение** метрик — гораздо надёжнее одного split'а.

---

## 3. МЕТОДОЛОГИЧЕСКИЕ ЗАМЕЧАНИЯ

### 3.1 ROC AUC завышает качество при сильном дисбалансе

**Проблема:**
При дисбалансе 99.8% / 0.2% ROC AUC всегда будет высоким, потому что True Negative Rate доминирует. У Isolation Forest ROC AUC = 0.93, но Precision = 0.23 — модель ловит мало фрода и при этом много ложных срабатываний.

**Решение — добавить PR AUC:**
```python
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

# В функцию evaluate_predictions добавить:
pr_auc = average_precision_score(y_true, y_score)
print(f"PR AUC: {pr_auc:.4f}")

# Построить PR-кривую
precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_score)
plt.figure(figsize=(8, 5))
plt.plot(recall_arr, precision_arr)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve — {model_name} (PR AUC={pr_auc:.4f})')
plt.show()
```

---

### 3.2 Нет подбора порога (threshold tuning)

**Проблема:**
Все модели используют `contamination = 0.002` как фиксированный порог. Но оптимальный порог зависит от бизнес-контекста:
- **Банк** предпочтёт высокий Recall (поймать максимум фрода, даже ценой FP)
- **Платёжная система** предпочтёт высокий Precision (не блокировать легитимных клиентов)

**Решение:**
```python
from sklearn.metrics import precision_recall_curve

precision_arr, recall_arr, thresholds = precision_recall_curve(y_test, score_lof)

# Найти порог для максимального F1
f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Лучший порог: {best_threshold:.4f}")
print(f"Precision: {precision_arr[best_idx]:.4f}")
print(f"Recall: {recall_arr[best_idx]:.4f}")
print(f"F1: {f1_scores[best_idx]:.4f}")
```

---

### 3.3 Нет ансамблевого подхода

**Проблема:**
6 моделей работают независимо. В индустрии комбинация моделей (ансамбль) почти всегда лучше отдельных моделей.

**Решение — Voting Ensemble:**
```python
# Нормализовать скоры всех моделей в [0, 1]
from sklearn.preprocessing import MinMaxScaler

scores = {
    'IF': score_if,
    'SVM': score_svm,
    'LOF': score_lof,
    'KMeans': score_kmeans,
    'AE': score_ae
}

scaler_mm = MinMaxScaler()
normalized_scores = {}
for name, s in scores.items():
    normalized_scores[name] = scaler_mm.fit_transform(s.reshape(-1, 1)).ravel()

# Средний скор
ensemble_score = np.mean(list(normalized_scores.values()), axis=0)
threshold_ens = np.percentile(ensemble_score, 100 * (1 - contamination))
y_pred_ens = (ensemble_score > threshold_ens).astype(int)

evaluate_predictions("Ensemble (Average)", y_test, y_pred_ens, ensemble_score)
```

---

### 3.4 Subsampling теряет 80–90% обучающих данных

**Проблема:**
```python
X_train_normal_small_svm = sample_rows(X_train_normal, 20000)   # из 102,459
X_train_normal_small_ee  = sample_rows(X_train_normal, 10000)   # из 102,459
```
SVM обучается на ~20% данных, EE на ~10%. Остальные данные **выбрасываются**. Это снижает способность моделей выучить разнообразие нормальных транзакций.

**Решение:**
- Для SVM: увеличить до 50,000 или использовать SGDOneClassSVM (поддерживает большие датасеты)
- Для EE: увеличить до 30,000
- Указать в дипломе как ограничение и показать зависимость качества от размера выборки:
```python
sizes = [5000, 10000, 20000, 50000, 100000]
for size in sizes:
    X_sub = sample_rows(X_train_normal, size)
    model = OneClassSVM(kernel='rbf', nu=contamination, gamma='scale')
    model.fit(X_sub)
    preds = np.where(model.predict(X_test_scaled) == -1, 1, 0)
    print(f"Size={size}, F1={f1_score(y_test, preds):.4f}")
```

---

### 3.5 Нет анализа ошибок (Error Analysis)

**Проблема:**
Пайплайн показывает общие метрики, но не анализирует **какие именно** мошенничества пропускаются. Может быть, все модели пропускают один и тот же тип фрода?

**Решение:**
```python
# Какие фроды пропускает лучшая модель (LOF)?
fraud_test = X_test[y_test == 1].copy()
fraud_test['Predicted'] = y_pred_lof[y_test == 1]
fraud_test['True'] = 1

missed = fraud_test[fraud_test['Predicted'] == 0]
caught = fraud_test[fraud_test['Predicted'] == 1]

print(f"Пойманные фроды: {len(caught)}")
print(f"Пропущенные фроды: {len(missed)}")
print(f"\nСредняя сумма пойманных: {caught['Amount'].mean():.2f}")
print(f"Средняя сумма пропущенных: {missed['Amount'].mean():.2f}")

# Визуализация: распределение суммы пойманных vs пропущенных
plt.figure(figsize=(10, 5))
plt.hist(caught['Amount'], bins=30, alpha=0.5, label='Пойманные')
plt.hist(missed['Amount'], bins=30, alpha=0.5, label='Пропущенные')
plt.legend()
plt.title('Суммы пойманных vs пропущенных мошенничеств')
plt.xlabel('Amount')
plt.show()
```

---

### 3.6 Нет анализа данных перед моделированием (EDA)

**Проблема:**
Отсутствует exploratory data analysis — нет понимания, **чем** фрод-транзакции отличаются от нормальных.

**Решение — добавить блок EDA:**
```python
# Сравнение распределений для фрода и нормальных
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Сумма
axes[0,0].hist(normal['Amount'], bins=50, alpha=0.5, label='Normal', density=True)
axes[0,0].hist(fraud['Amount'], bins=50, alpha=0.5, label='Fraud', density=True)
axes[0,0].set_title('Amount Distribution')
axes[0,0].legend()

# Время
axes[0,1].hist(normal['Time'] % 86400 / 3600, bins=24, alpha=0.5, label='Normal', density=True)
axes[0,1].hist(fraud['Time'] % 86400 / 3600, bins=24, alpha=0.5, label='Fraud', density=True)
axes[0,1].set_title('Hour of Day')
axes[0,1].legend()

# Наиболее различающиеся V-компоненты
for i, (ax, col) in enumerate(zip(axes.flat[2:], ['V1', 'V4', 'V10', 'V14'])):
    ax.hist(normal[col], bins=50, alpha=0.5, label='Normal', density=True)
    ax.hist(fraud[col], bins=50, alpha=0.5, label='Fraud', density=True)
    ax.set_title(f'{col} Distribution')
    ax.legend()

plt.tight_layout()
plt.show()
```

---

## 4. ЧТО СДЕЛАНО ХОРОШО

- **Правильная структура пайплайна:** загрузка → очистка → preprocessing → обучение → оценка — логично и последовательно
- **Обучение на нормальных данных** (novelty detection) — валидный подход для сценариев, где разметка недоступна
- **Contamination на основе реального процента фрода** — правильнее, чем произвольное значение
- **Confusion matrix для каждой модели** — наглядная визуализация ошибок
- **Сравнительная таблица всех моделей** — удобно для анализа
- **LOF как лучший результат** (F1=0.55, Recall=0.78) — разумный выбор для novelty detection
- **StandardScaler** перед моделями — обязательный шаг для SVM, LOF и Autoencoder
- **EarlyStopping** для Autoencoder — правильный подход (хотя параметры нужно подтянуть)
- **Разнообразие моделей** — 6 разных подходов дают полную картину unsupervised методов

---

## 5. ПРИОРИТЕТЫ ИСПРАВЛЕНИЙ

### Обязательный минимум (для защиты диплома)

| # | Проблема | Сложность | Влияние |
|---|----------|-----------|---------|
| 1 | Исправить обработку NaN в фичах | 1 строка кода | Корректность данных |
| 2 | Объяснить почему датасет 128K вместо 284K | Текст в дипломе | Воспроизводимость |
| 3 | Увеличить эпохи Autoencoder до 100, patience до 10 | 2 строки кода | Улучшение Autoencoder |
| 4 | Добавить хотя бы 1 supervised модель (XGBoost) | ~15 строк кода | Ключевое сравнение |
| 5 | Добавить PR AUC метрику | ~5 строк кода | Корректная оценка |
| 6 | Описать причину провала Elliptic Envelope | Текст в дипломе | Научная обоснованность |

### Для отличной оценки

| # | Проблема | Сложность | Влияние |
|---|----------|-----------|---------|
| 7 | Временной split вместо случайного | ~10 строк кода | Реалистичность оценки |
| 8 | Feature engineering (log Amount, Hour) | ~10 строк кода | Улучшение всех моделей |
| 9 | Stratified K-Fold Cross-Validation | ~15 строк кода | Надёжность метрик |
| 10 | Threshold tuning с PR-кривой | ~10 строк кода | Практическая ценность |
| 11 | Error analysis (анализ пропущенных фродов) | ~15 строк кода | Глубина анализа |
| 12 | Ансамбль моделей | ~15 строк кода | Лучший итоговый результат |

### Для выдающейся работы

| # | Проблема | Сложность | Влияние |
|---|----------|-----------|---------|
| 13 | Улучшенная архитектура Autoencoder | ~20 строк кода | Значительное улучшение |
| 14 | EDA блок с визуализацией различий | ~30 строк кода | Понимание данных |
| 15 | Variational Autoencoder (VAE) | ~40 строк кода | Научная новизна |
| 16 | SHAP/LIME интерпретация | ~20 строк кода | Объяснимость моделей |

---

> Все примеры кода в этом документе готовы к вставке в ноутбук.
> При вопросах — обращайтесь.
