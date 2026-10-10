"""FinSense demo — run with:  streamlit run app.py"""
import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from finsense_utils import LABELS, predict_with_confidence, split_sentences

MODEL_PATH = Path(__file__).parent / "models" / "finsense_model.joblib"
STYLE = {
    "positive": ("#2a78d6", "▲ Positive"),
    "neutral": ("#8a8984", "● Neutral"),
    "negative": ("#e34948", "▼ Negative"),
}
MASCOT = {
    "positive": ("🐂", "fs-bull", "กระทิงมา! ข่าวนี้ออกแนวบวก"),
    "neutral": ("🦀", "fs-crab", "ปูเดินข้าง — ข่าวนี้ไม่ชี้ทิศทาง"),
    "negative": ("🐻", "fs-bear", "ระวังหมี! ข่าวนี้ออกแนวลบ"),
}
EXAMPLES = {
    "กำไรเพิ่ม": "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007.",
    "ขาดทุน": "The company reported a net loss and announced it will lay off 200 employees.",
    "ประชุมผู้ถือหุ้น": "The annual general meeting will be held in Helsinki on 15 March.",
    "ข่าวยาว · ดี": (
        "Nordvik Systems Oyj reported a strong third quarter on Thursday. "
        "Net sales increased by 18% to EUR 245.3 mn from EUR 207.9 mn a year earlier. "
        "Operating profit rose to EUR 31.4 mn, compared with EUR 22.6 mn in the corresponding period. "
        "The company also won a new five-year service contract worth about EUR 40 mn. "
        "Nordvik raised its full-year outlook and now expects operating profit to grow significantly."
    ),
    "ข่าวยาว · ร้าย": (
        "Baltic Paper Corp. said its net sales decreased by 22% to EUR 96.5 mn in the second quarter. "
        "The company swung to an operating loss of EUR 12.8 mn from a profit of EUR 4.1 mn a year ago. "
        "The operating margin fell to 3.1% from 7.4% due to weak demand and higher energy costs. "
        "The company will close its mill in Kotka and cut about 350 jobs. "
        "Baltic Paper lowered its guidance and now expects full-year sales to decline."
    ),
    "ข่าวยาว · ผสม": (
        "Helmi Retail Group's net sales grew by 9% to EUR 512 mn in January-June. "
        "However, the company posted a net loss of EUR 6.2 mn due to restructuring costs. "
        "Helmi plans to reduce its workforce by 120 people in Finland. "
        "Online sales increased by 35% and now account for a quarter of total sales. "
        "The company will publish its interim report for January-September on 24 October."
    ),
}
CSS = """
<style>
[data-testid="stHeader"] {background: transparent;}
[data-testid="stMainBlockContainer"] {max-width: 720px; padding-top: 3rem;}
.fs-title {font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; margin: 0;}
.fs-sub {opacity: 0.6; margin: 0.15rem 0 1.25rem;}
.fs-card {display: flex; gap: 1rem; align-items: flex-start; padding: 1.1rem 1.25rem; border-radius: 10px;
          background: #fbf7ef; border: 1px solid rgba(120,95,60,0.18);}
.fs-body {flex: 1; min-width: 0;}
.fs-head {display: flex; justify-content: space-between; align-items: baseline;}
.fs-label {font-size: 1.5rem; font-weight: 700;}
.fs-conf {opacity: 0.7;}
.fs-tag {font-size: 0.9rem; margin-top: 0.1rem;}
.fs-mascot {font-size: 2.6rem; line-height: 1; padding-top: 0.15rem;}
.fs-bull {animation: fs-bull 0.9s cubic-bezier(.2,.8,.3,1.2) both;}
.fs-bear {animation: fs-bear 1s ease-out both;}
.fs-crab {animation: fs-crab 1.2s ease-in-out both;}
@keyframes fs-bull {
  0% {transform: translateY(28px) scale(0.6); opacity: 0;}
  60% {transform: translateY(-8px) scale(1.1); opacity: 1;}
  100% {transform: translateY(0) scale(1);}
}
@keyframes fs-bear {
  0% {transform: translate(-18px, -26px) rotate(-18deg); opacity: 0;}
  70% {transform: translate(2px, 3px) rotate(6deg); opacity: 1;}
  100% {transform: translate(0, 0) rotate(0);}
}
@keyframes fs-crab {
  0%, 100% {transform: translateX(0);}
  20%, 60% {transform: translateX(-10px);}
  40%, 80% {transform: translateX(10px);}
}
@media (prefers-reduced-motion: reduce) {.fs-mascot {animation: none;}}
.fs-bar {display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin: 0.8rem 0 0.5rem;
         background: rgba(120,95,60,0.12);}
.fs-legend {display: flex; gap: 1rem; font-size: 0.8rem; opacity: 0.75;}
.fs-legend i {display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.35rem;}
.fs-note {font-size: 0.85rem; color: #d9822b; margin-top: 0.6rem;}
.fs-text {line-height: 1.9;}
.fs-sent {padding: 0.15rem 0.2rem; border-radius: 4px; box-decoration-break: clone;
          -webkit-box-decoration-break: clone;}
.fs-meta {font-size: 0.85rem; opacity: 0.7; margin: 0.4rem 0 0.25rem;}
.fs-foot {font-size: 0.8rem; opacity: 0.5; margin-top: 3rem; text-align: center;}
</style>
"""

st.set_page_config(page_title="FinSense", page_icon="📰", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict_df(model, texts):
    labels, proba = predict_with_confidence(model, texts)
    out = pd.DataFrame({"text": texts, "sentiment": labels, "confidence": proba.max(axis=1)})
    for i, c in enumerate(model.classes_):
        out[f"p_{c}"] = proba[:, i]
    return out


def rgba(hex_color, alpha):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha:.2f})"


def show_result(label, proba_by_label, confidence):
    color, name = STYLE[label]
    bar = "".join(f"<div style='width:{float(proba_by_label[lab]):.1%};background:{STYLE[lab][0]}'></div>"
                  for lab in LABELS)
    legend = "".join(f"<span><i style='background:{STYLE[lab][0]}'></i>{lab} {float(proba_by_label[lab]):.0%}</span>"
                     for lab in LABELS)
    note = "<div class='fs-note'>ความมั่นใจต่ำ — ควรอ่านข่าวนี้ด้วยตนเอง</div>" if confidence < 0.5 else ""
    emoji, anim, tag = MASCOT[label]
    st.markdown(
        f"<div class='fs-card'><div class='fs-mascot {anim}'>{emoji}</div><div class='fs-body'>"
        f"<div class='fs-head'><span class='fs-label' style='color:{color}'>{name}</span>"
        f"<span class='fs-conf'>ความมั่นใจ {confidence:.0%}</span></div>"
        f"<div class='fs-tag' style='color:{color}'>{tag}</div>"
        f"<div class='fs-bar'>{bar}</div><div class='fs-legend'>{legend}</div>{note}</div></div>",
        unsafe_allow_html=True,
    )


def show_sentences(out):
    spans = []
    for _, row in out.iterrows():
        color = STYLE[row["sentiment"]][0]
        alpha = 0.08 + 0.22 * row["confidence"]
        tip = f"{row['sentiment']} {row['confidence']:.0%}"
        spans.append(f"<span class='fs-sent' title='{tip}' style='background:{rgba(color, alpha)}'>"
                     f"{html.escape(row['text'])}</span>")
    counts = out["sentiment"].value_counts().reindex(LABELS, fill_value=0)
    summary = " · ".join(f"{STYLE[lab][1].split()[0]} {counts[lab]}" for lab in reversed(LABELS))
    st.markdown(
        f"<div class='fs-meta'>รายประโยค — {summary} จาก {len(out)} ประโยค (ชี้ที่ประโยคเพื่อดูค่า)</div>"
        f"<div class='fs-text'>{' '.join(spans)}</div>",
        unsafe_allow_html=True,
    )
    with st.expander("ดูเป็นตาราง"):
        st.dataframe(
            out[["text", "sentiment", "confidence"]].rename(columns={"text": "sentence"}),
            width="stretch",
            hide_index=True,
            column_config={"confidence": CONFIDENCE_COLUMN},
        )


CONFIDENCE_COLUMN = st.column_config.ProgressColumn("confidence", min_value=0, max_value=1, format="%.2f")

st.markdown("<p class='fs-title'>FinSense</p>"
            "<p class='fs-sub'>วิเคราะห์ sentiment ของข่าวการเงินภาษาอังกฤษ — Positive / Neutral / Negative</p>",
            unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(f"ไม่พบโมเดลที่ `{MODEL_PATH}` — กรุณารัน `FinSense.ipynb` ให้จบก่อนเพื่อสร้างไฟล์โมเดล")
    st.stop()

model = load_model()

tab_single, tab_batch = st.tabs(["วิเคราะห์ข่าว", "หลายข่าว (CSV)"])

with tab_single:
    example = st.pills("ตัวอย่าง", list(EXAMPLES), selection_mode="single", label_visibility="collapsed")
    text = st.text_area("ข้อความข่าว", value=EXAMPLES.get(example, ""), height=180,
                        placeholder="วางข่าวภาษาอังกฤษที่นี่ หรือเลือกตัวอย่างด้านบน …",
                        label_visibility="collapsed")
    if st.button("วิเคราะห์", type="primary", disabled=not text.strip()):
        sentences = split_sentences(text) or [text.strip()]
        out = predict_df(model, sentences)
        if len(out) == 1:
            res = out.iloc[0]
            show_result(res["sentiment"], {lab: res[f"p_{lab}"] for lab in LABELS}, res["confidence"])
        else:
            mean = {lab: out[f"p_{lab}"].mean() for lab in LABELS}
            overall = max(mean, key=mean.get)
            show_result(overall, mean, mean[overall])
            show_sentences(out)

with tab_batch:
    st.caption("วางข่าวบรรทัดละ 1 ข่าว หรืออัปโหลดไฟล์ CSV ที่มีคอลัมน์ข้อความ")
    pasted = st.text_area("ข่าวหลายบรรทัด", height=150, key="batch_text", label_visibility="collapsed",
                          placeholder="ข่าวที่ 1\nข่าวที่ 2\n…")
    uploaded = st.file_uploader("อัปโหลด CSV", type="csv", label_visibility="collapsed")

    texts = [t.strip() for t in pasted.splitlines() if t.strip()]
    if uploaded is not None:
        csv = pd.read_csv(uploaded)
        col = st.selectbox("คอลัมน์ข้อความ", csv.columns)
        texts = csv[col].dropna().astype(str).tolist()

    if texts and st.button("วิเคราะห์ทั้งหมด", type="primary"):
        out = predict_df(model, texts)
        counts = out["sentiment"].value_counts().reindex(LABELS, fill_value=0)
        for c, label in zip(st.columns(3), reversed(LABELS)):
            c.metric(STYLE[label][1], f"{counts[label]:,}", f"{counts[label] / len(out):.0%}", delta_color="off")
        st.dataframe(
            out[["text", "sentiment", "confidence"]],
            width="stretch",
            hide_index=True,
            column_config={"confidence": CONFIDENCE_COLUMN},
        )
        st.download_button("ดาวน์โหลดผล (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                           "finsense_results.csv", "text/csv")

st.markdown(f"<div class='fs-foot'>{type(model.named_steps['model']).__name__} + TF-IDF · Financial PhraseBank"
            f" · ใช้คัดกรองข่าวเบื้องต้น ไม่ใช่คำแนะนำการลงทุน</div>", unsafe_allow_html=True)
