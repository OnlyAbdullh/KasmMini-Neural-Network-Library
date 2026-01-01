# 🧠 KasmMiniNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Version](https://img.shields.io/badge/version-0.1.0-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-educational-orange)

**KasmMiniNN** هي مكتبة Python خفيفة لبناء وتدريب شبكات عصبية بسيطة من الصفر،  
موجّهة للتعلّم، الفهم العميق، والاختبار السريع بدون الاعتماد على أطر جاهزة.

> الهدف الأساسي هو فهم ما يحدث داخل الشبكات العصبية (`forward` / `backward`) بدل استخدامها كصندوق أسود.

---

## 📌 الملخّص

| العنصر  | القيمة                               |
|--------|--------------------------------------|
| النسخة | `0.1.0`                              |
| المؤلف | `OnlyOne`                            |
| الغاية | بناء شبكات عصبية يدويًا للتعلّم والتجربة |

---

## ✨ المزايا الرئيسية

- طبقة **`Dense`** مع تجميع الأوزان والتدرجات  
- دوال تنشيط:
  - `Relu()`, `LeakyReLU()`, `Sigmoid()`, `Tanh()`, `Linear()`
- طبقات تنظيم:
  - `Dropout()`
  - `BatchNormalization()`
- دوال خسارة:
  - `SoftmaxCrossEntropy`, `MeanSquaredError`, `BinaryCrossEntropy`
- محسنات:
  - `SGD`, `Momentum`, `AdaGrad`, `Adam`
- كائن `NeuralNetwork` لتجميع الطبقات
- كائن `Trainer` لإدارة التدريب والتقييم
- `HyperparameterTuner` لدعم:
  - Grid Search
  - Random Search
  - K-Fold Cross Validation
- أمثلة جاهزة على **Iris** و **MNIST**
- أدوات رسم تاريخ التدريب (`plotting.py`)

---

## 📦 المتطلبات

- **Python:** 3.8+  
- المكتبات:
  - `numpy`
  - `scikit-learn`
  - `matplotlib` (للرسوم فقط)

### التثبيت
```bash
pip install numpy scikit-learn matplotlib

الاستخدام المحلي من المستودع
git clone <repo-url>
cd <repo-directory>
pip install -e .

🚀 دليل استخدام سريع
1️⃣ تدريب بسيط (Iris)
from KasmMiniNN import (
    Dense, Sigmoid, Relu, BatchNormalization,
    SoftmaxCrossEntropy, NeuralNetwork, SGD, Trainer
)

# افترض أن x_train, x_val, x_test, t_train, t_val, t_test جاهزون ومهيأون
layers = [
    Dense(input_dim, 32),
    Sigmoid(),
    BatchNormalization(32),
    Dense(32, 16),
    Relu(),
    Dense(16, num_classes),
]

net = NeuralNetwork(layers, SoftmaxCrossEntropy())
optimizer = SGD(lr=0.1)

trainer = Trainer(
    network=net,
    optimizer=optimizer,
    x_train=x_train,
    t_train=t_train,
    x_val=x_val,
    t_val=t_val,
    x_test=x_test,
    t_test=t_test,
    epochs=20,
    batch_size=64,
)

history = trainer.fit()

2️⃣ ضبط المعاملات (Hyperparameter Tuning)
from KasmMiniNN import HyperparameterTuner

def build_network_from_config(config):
    # بناء شبكة اعتمادًا على config وإرجاع NeuralNetwork
    ...

tuner = HyperparameterTuner(
    build_network=lambda cfg: build_network_from_config(cfg),
    x_train=x_train, t_train=t_train,
    x_val=x_val, t_val=t_val,
)

results = tuner.grid_search(
    learning_rates=[1e-3, 1e-2],
    batch_sizes=[32, 64],
    hidden_sizes=[64, 128],
    optimizer_types=["adam"],
    dropout_rates=[0.0, 0.3],
    epochs_list=[10],
    num_layers_list=[1, 2],
    activation_types=["relu", "tanh"]
)

best_params = results["best_params"]

📚 مرجع سريع للـ API
NeuralNetwork
NeuralNetwork(layers, loss_layer)


يوفّر الوظائف: forward, predict, loss, accuracy, gradient, init_weight.

Dense
Dense(input_size, output_size, weight_init="he", bias_init=0.)

دوال التفعيل

Relu()

LeakyReLU(alpha)

Sigmoid()

Tanh()

Linear()

Layers التنظيم

Dropout(dropout_ratio)

BatchNormalization(feature_size, momentum=0.9)

دوال الخسارة
الدالة	الاستخدام
SoftmaxCrossEntropy	تصنيف متعدد
MeanSquaredError	انحدار
BinaryCrossEntropy	تصنيف ثنائي
Optimizers

SGD(lr)

Momentum(lr, momentum=0.9)

AdaGrad(lr)

Adam(lr, beta1=0.9, beta2=0.999)

🧩 التصميم الداخلي (بإيجاز)

كل Layer يملك forward و backward.

NeuralNetwork.gradient: forward → loss → backward.

Trainer يدير التدريب والتقييم.

BatchNormalization يدعم وضعيتي training و evaluation.

Dropout يُفعل أثناء التدريب فقط.

🛠️ حلول لمشاكل شائعة

خطأ:

RuntimeError: forward must be called before backward


الحل: تأكد من استدعاء forward أو loss قبل backward.

اختلاف الأبعاد:

تحقق من x.shape[0] == t.shape[0].

تأكد من تنسيق t (labels أو one-hot).

▶️ تشغيل الأمثلة
python example.py


سيُطلب منك:

اختيار البيانات: Iris أو MNIST

وضع التشغيل: train, tune, random, kfold

رسم تاريخ التدريب
from KasmMiniNN.plotting import plot_history
plot_history(history)

🤝 المساهمة

مرحب بالمساهمات:

تحسينات الأداء

توثيق أفضل

إضافة Layers أو Optimizers جديدة

📌 يفضّل إضافة اختبارات وحدة باستخدام pytest.

📄 الترخيص

MIT License — ضع ملف LICENSE في المستودع.

📬 معلومات الاتصال

المؤلف: OnlyOne

الإصدار: 0.1.0

ابدأ الآن — افهم الشبكات العصبية من الداخل بدل الاكتفاء باستخدامها.