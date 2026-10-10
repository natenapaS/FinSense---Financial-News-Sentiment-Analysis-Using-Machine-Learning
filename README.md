# FinSense: Financial News Sentiment Analysis Using Machine Learning

ระบบ AI วิเคราะห์ข้อความข่าวทางการเงินภาษาอังกฤษ และจำแนกเป็น **Positive / Negative / Neutral**
เพื่อช่วยให้นักลงทุนคัดกรองข่าวเบื้องต้นได้รวดเร็วยิ่งขึ้น

## โครงสร้างโปรเจกต์

```
├── FinSense.ipynb              โน้ตบุ๊กหลัก: Problem → Data → EDA → Preprocessing → Training → Evaluation
├── finsense_utils.py           clean_text, stop words, ฟังก์ชันทำนายพร้อมความมั่นใจ (ใช้ร่วมกับ app)
├── app.py                      เว็บเดโม Streamlit
├── data/all-data.csv           Financial PhraseBank (4,846 ประโยค)
├── models/finsense_model.joblib  โมเดลที่ดีที่สุด (สร้างจากโน้ตบุ๊ก)
└── requirements.txt
```

## วิธีใช้งาน

```bash
pip install -r requirements.txt

# 1) รันโน้ตบุ๊ก (เปิดใน Jupyter หรือสั่งรันทั้งไฟล์) — จะสร้าง models/finsense_model.joblib
jupyter notebook FinSense.ipynb
# หรือ
jupyter nbconvert --to notebook --execute --inplace FinSense.ipynb

# 2) เปิดเว็บเดโม
streamlit run app.py
```

> ส่วนเปรียบเทียบ FinBERT จะดาวน์โหลดโมเดล ~440 MB ครั้งแรก ถ้าไม่ต้องการให้ตั้ง `RUN_FINBERT = False` ในเซลล์แรกของโน้ตบุ๊ก

## วิธีการ

- **Preprocessing:** lowercase, ลบ URL, `n't → not`, ตัวเลข → `num`, ลบเครื่องหมาย; เก็บคำบอกทิศทาง (not, up, down, more, less) ไว้
- **Features:** TF-IDF (unigram/bigram) ใน `Pipeline`
- **Data split (stratified):** Train 70% (3,386) / Validation 15% (726) / Test 15% (726)
- **Hyperparameter tuning:** `GridSearchCV` บน Train (5-fold stratified, scoring = macro-F1) เทียบ Naive Bayes, Logistic Regression, Linear SVM, Random Forest
- **Model selection:** เลือกชนิดโมเดลจาก macro-F1 บน Validation set แล้วเทรนใหม่บน Train+Validation
- **Evaluation:** วัดผลบน Test set ครั้งเดียว — learning curve / validation curve, classification report, confusion matrix, น้ำหนักคำ, error analysis, เทียบกับ FinBERT
- **ข่าวยาว (เว็บเดโม):** โมเดลเทรนบนประโยคเดี่ยว จึงแยกข่าวยาวเป็นรายประโยค (`split_sentences`) ทำนายทีละประโยค แล้วสรุปภาพรวมจากค่าเฉลี่ยความน่าจะเป็น

## ผลลัพธ์

| Model | Validation Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---|---|---|
| **Linear SVM** (โมเดลที่เลือก) | **0.745** | 0.777 | **0.736** |
| Logistic Regression | 0.739 | 0.773 | 0.736 |
| Multinomial NB | 0.713 | 0.730 | 0.672 |
| Random Forest | 0.697 | 0.741 | 0.671 |
| FinBERT (ProsusAI, pretrained)* | – | 0.872 | 0.866 |

\* FinBERT ถูก fine-tune บน Financial PhraseBank มาก่อน คะแนนจึงอาจสูงเกินจริง

## ข้อมูลอ้างอิง

- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts.* JASIST. — Financial PhraseBank, license CC BY-NC-SA 3.0
- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* — `ProsusAI/finbert`
