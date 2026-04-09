import re
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


# =========================================================
# 금융 뉴스/리포트 감성 분석 시스템
# =========================================================

st.set_page_config(
    page_title="금융 문서 감성 분석 시스템",
    page_icon="📈",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# 모델 경로
MODEL_PATH_BEST_LOSS = BASE_DIR / "models2" / "main_base_best_val_loss.keras"
MODEL_PATH_BEST_ACC = BASE_DIR / "models2" / "main_base_best_val_accuracy.keras"
TOKENIZER_PATH_MAIN = BASE_DIR / "models2" / "tokenizer_main.pkl"
CONFIG_PATH_MAIN = BASE_DIR / "models2" / "preprocess_config.json"

# 기본값: 시연 체감상 더 자연스러웠던 모델
DEFAULT_MODEL_PATH = str(MODEL_PATH_BEST_LOSS)
DEFAULT_TOKENIZER_PATH = str(TOKENIZER_PATH_MAIN)
DEFAULT_CONFIG_PATH = str(CONFIG_PATH_MAIN)

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_SCORE = {"negative": -1, "neutral": 0, "positive": 1}


# -----------------------------
# 세션 상태 초기화
# -----------------------------
if "model_path_input" not in st.session_state:
    st.session_state["model_path_input"] = DEFAULT_MODEL_PATH

if "tokenizer_path_input" not in st.session_state:
    st.session_state["tokenizer_path_input"] = DEFAULT_TOKENIZER_PATH

if "config_path_input" not in st.session_state:
    st.session_state["config_path_input"] = DEFAULT_CONFIG_PATH

if "active_model_name" not in st.session_state:
    st.session_state["active_model_name"] = "실전 해석형 모델"

if "active_model_desc" not in st.session_state:
    st.session_state["active_model_desc"] = "시연 체감상 더 자연스럽게 보였던 모델"

if "apply_message" not in st.session_state:
    st.session_state["apply_message"] = ""


# -----------------------------
# 유틸 함수
# -----------------------------
def apply_model(model_name: str, model_path: str, desc: str):
    st.session_state["model_path_input"] = model_path
    st.session_state["active_model_name"] = model_name
    st.session_state["active_model_desc"] = desc
    st.session_state["apply_message"] = f"'{model_name}' 적용 완료"


def softmax_to_label(prob: np.ndarray) -> Tuple[str, float]:
    idx = int(np.argmax(prob))
    return LABEL_MAP[idx], float(prob[idx])


def split_text_into_sentences(text: str) -> List[str]:
    if not text:
        return []

    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text).strip()

    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)

    cleaned = []
    for p in parts:
        p = p.strip()
        if len(p) >= 8:
            cleaned.append(p)

    return cleaned


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_all_resources(model_path_str: str, tokenizer_path_str: str, config_path_str: str):
    model_path = Path(model_path_str)
    tokenizer_path = Path(tokenizer_path_str)
    config_path = Path(config_path_str)

    model = load_model(model_path)
    tokenizer = load_pickle(tokenizer_path)

    # 학습 노트북 기준 max_len = 40
    config = {"max_len": 40}
    if config_path.exists():
        try:
            config.update(load_json(config_path))
        except Exception:
            pass

    return model, tokenizer, config


def predict_sentences(sentences: List[str], model, tokenizer, max_len: int) -> pd.DataFrame:
    sequences = tokenizer.texts_to_sequences(sentences)
    x_pad = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    probs = model.predict(x_pad, verbose=0)

    rows = []
    for i, (sent, prob) in enumerate(zip(sentences, probs), start=1):
        label, conf = softmax_to_label(prob)
        rows.append({
            "sentence_no": i,
            "sentence": sent,
            "pred_label": label,
            "confidence": round(conf, 4),
            "negative_prob": round(float(prob[0]), 4),
            "neutral_prob": round(float(prob[1]), 4),
            "positive_prob": round(float(prob[2]), 4),
            "sentiment_score": LABEL_SCORE[label]
        })

    return pd.DataFrame(rows)


def summarize_document(df: pd.DataFrame) -> Dict:
    total = len(df)
    if total == 0:
        return {
            "total_sentences": 0,
            "negative_ratio": 0.0,
            "neutral_ratio": 0.0,
            "positive_ratio": 0.0,
            "overall_label": "분석 불가",
            "overall_reason": "문장이 없어서 분석할 수 없습니다."
        }

    neg = int((df["pred_label"] == "negative").sum())
    neu = int((df["pred_label"] == "neutral").sum())
    pos = int((df["pred_label"] == "positive").sum())

    negative_ratio = neg / total
    neutral_ratio = neu / total
    positive_ratio = pos / total

    if negative_ratio >= 0.45:
        overall_label = "전체적으로 부정적"
        overall_reason = "부정 문장 비중이 높아 투자 심리에 부담을 줄 수 있습니다."
    elif positive_ratio >= 0.45:
        overall_label = "전체적으로 긍정적"
        overall_reason = "긍정 문장 비중이 높아 기대 심리를 자극할 수 있습니다."
    elif neutral_ratio >= 0.50:
        overall_label = "전체적으로 중립적"
        overall_reason = "정보 전달형 문장이 많아 방향성이 강하지 않습니다."
    else:
        overall_label = "혼합적"
        overall_reason = "긍정과 부정이 함께 섞여 있어 해석이 갈릴 수 있습니다."

    return {
        "total_sentences": total,
        "negative_ratio": negative_ratio,
        "neutral_ratio": neutral_ratio,
        "positive_ratio": positive_ratio,
        "overall_label": overall_label,
        "overall_reason": overall_reason
    }


def sentence_level_comment(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "분석 가능한 문장이 없습니다."

    neg_count = int((df["pred_label"] == "negative").sum())
    pos_count = int((df["pred_label"] == "positive").sum())
    neu_count = int((df["pred_label"] == "neutral").sum())

    comments = [
        f"총 {len(df)}개 문장 중 부정 {neg_count}개, 중립 {neu_count}개, 긍정 {pos_count}개입니다."
    ]

    if neg_count > pos_count:
        comments.append("부정 신호가 상대적으로 강합니다.")
    elif pos_count > neg_count:
        comments.append("긍정 신호가 상대적으로 강합니다.")
    else:
        comments.append("긍정과 부정이 비슷한 수준입니다.")

    comments.append("상위 부정 문장과 상위 긍정 문장을 함께 확인하는 것이 좋습니다.")
    return " ".join(comments)


def make_ratio_df(summary: Dict) -> pd.DataFrame:
    return pd.DataFrame({
        "label": ["negative", "neutral", "positive"],
        "ratio": [
            summary["negative_ratio"],
            summary["neutral_ratio"],
            summary["positive_ratio"]
        ]
    })


def build_highlighted_text(df: pd.DataFrame) -> str:
    chunks = []
    for _, row in df.iterrows():
        sent = row["sentence"]
        label = row["pred_label"]
        conf = row["confidence"]

        if label == "negative":
            color = "#ffe1e1"
            border = "#d9534f"
        elif label == "positive":
            color = "#e3ffe5"
            border = "#2e8b57"
        else:
            color = "#eef3ff"
            border = "#4f81bd"

        chunk = f"""
        <div style="background:{color}; border-left:6px solid {border}; padding:10px; margin-bottom:8px; border-radius:8px;">
            <div style="font-size:12px; color:#555;">{label} | confidence={conf:.4f}</div>
            <div style="font-size:15px;">{sent}</div>
        </div>
        """
        chunks.append(chunk)
    return "\n".join(chunks)


def load_text_from_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if suffix == ".csv":
        df = pd.read_csv(uploaded_file)
        if df.shape[1] == 1:
            return "\n".join(df.iloc[:, 0].astype(str).tolist())
        return "\n".join(" ".join(row.astype(str).tolist()) for _, row in df.iterrows())

    return ""


SAMPLE_TEXT = """Samsung Electronics reported stronger-than-expected quarterly earnings and said demand for advanced memory chips improved.
The company also announced that data center customers were increasing orders for high-bandwidth memory products.
However, management warned that smartphone demand could remain weak in the near term.
Some analysts noted that rising capital expenditure may pressure margins in the second half of the year.
Investors responded positively to the stronger server-related outlook despite concerns over near-term volatility.
"""


# -----------------------------
# 상단
# -----------------------------
st.title("📈 금융 뉴스/리포트 감성 분석 시스템")
st.caption("긴 금융 텍스트를 문장 단위로 분해해 감성을 분석하고, 전체 문서 흐름까지 시각화합니다.")

# -----------------------------
# 사이드바
# -----------------------------
with st.sidebar:
    st.header("설정")

    st.markdown("### 현재 적용 모델")
    st.success(f"{st.session_state['active_model_name']}")
    st.caption(st.session_state["active_model_desc"])

    if st.session_state["apply_message"]:
        st.info(st.session_state["apply_message"])

    st.markdown("---")
    st.markdown("### 모델 선택")

    st.markdown("**실전 해석형 모델**")
    st.caption("시연 체감상 문장 해석이 더 자연스러웠던 모델")
    st.code(str(MODEL_PATH_BEST_LOSS), language=None)
    if st.button("실전 해석형 모델 적용", use_container_width=True):
        apply_model(
            "실전 해석형 모델",
            str(MODEL_PATH_BEST_LOSS),
            "시연 체감상 더 자연스럽게 보였던 모델"
        )
        st.rerun()

    st.markdown("**정량 지표 최고 모델**")
    st.caption("accuracy / F1 기준 전체 비교 1위 모델")
    st.code(str(MODEL_PATH_BEST_ACC), language=None)
    if st.button("정량 1위 모델 적용", use_container_width=True):
        apply_model(
            "정량 지표 최고 모델",
            str(MODEL_PATH_BEST_ACC),
            "accuracy / F1 기준 전체 비교 1위 모델"
        )
        st.rerun()

    st.markdown("---")
    st.markdown("### 토크나이저")
    st.caption("현재 모델들과 함께 쓰는 토크나이저")
    st.code(str(TOKENIZER_PATH_MAIN), language=None)

    st.markdown("---")
    st.markdown("### 직접 경로 입력")

    st.text_input(
        "모델 경로",
        key="model_path_input",
        help="위 버튼으로 자동 적용하거나, 직접 경로를 수정할 수 있습니다."
    )

    st.text_input(
        "토크나이저 경로",
        key="tokenizer_path_input",
        help="기본 토크나이저는 위 코드 블록 경로를 사용하세요."
    )

    st.text_input(
        "전처리 설정 경로",
        key="config_path_input"
    )

    st.markdown("---")
    st.markdown("**입력 권장 예시**")
    st.write("- 영어 금융 뉴스 기사")
    st.write("- 영어 실적 기사")
    st.write("- 영어 공시 해설")
    st.write("- 영어 증권 리포트 일부")
    st.write("- A4 한 장 수준 텍스트")


# -----------------------------
# 리소스 로드
# -----------------------------
resource_error = None
try:
    model, tokenizer, config = load_all_resources(
        st.session_state["model_path_input"],
        st.session_state["tokenizer_path_input"],
        st.session_state["config_path_input"]
    )
    max_len = int(config.get("max_len", 40))
except Exception as e:
    resource_error = str(e)
    model, tokenizer, config, max_len = None, None, None, 40

if resource_error:
    st.error("모델 또는 토크나이저를 불러오지 못했습니다.")
    st.code(resource_error)
    st.info("사이드바에서 추천 모델 버튼을 눌러 다시 적용해 보세요.")
    st.stop()


# -----------------------------
# 본문
# -----------------------------
input_col, meta_col = st.columns([3, 1])

with input_col:
    uploaded_file = st.file_uploader("텍스트 파일 업로드 (txt, csv)", type=["txt", "csv"])
    uploaded_text = load_text_from_uploaded_file(uploaded_file)
    default_text = uploaded_text if uploaded_text else SAMPLE_TEXT

    text_input = st.text_area(
        "분석할 금융 문서 입력",
        value=default_text,
        height=320,
        help="영어 금융 뉴스 기사, 실적 기사, 공시 설명 등 긴 텍스트를 붙여넣으세요."
    )

    btn1, btn2 = st.columns(2)
    with btn1:
        use_sample = st.button("샘플 문서 불러오기")
    with btn2:
        run_analysis = st.button("분석 실행", type="primary")

    if use_sample:
        st.session_state["sample_text"] = SAMPLE_TEXT
        st.rerun()

    if "sample_text" in st.session_state:
        text_input = st.session_state["sample_text"]

with meta_col:
    st.subheader("현재 설정")
    st.write(f"최대 길이(max_len): **{max_len}**")
    st.write(f"현재 모델 파일: **{Path(st.session_state['model_path_input']).name}**")
    st.write(f"토크나이저 파일: **{Path(st.session_state['tokenizer_path_input']).name}**")
    st.markdown("---")
    st.subheader("현재 연결 상태")
    st.success(f"{st.session_state['active_model_name']} 연결 중")
    st.caption(st.session_state["active_model_desc"])
    st.markdown("---")
    st.subheader("분석 흐름")
    st.write("1. 긴 문서 입력")
    st.write("2. 문장 자동 분리")
    st.write("3. 문장별 감성 예측")
    st.write("4. 전체 문서 요약")


if run_analysis:
    text_input = (text_input or "").strip()

    if not text_input:
        st.warning("분석할 텍스트를 입력하세요.")
        st.stop()

    sentences = split_text_into_sentences(text_input)

    if len(sentences) == 0:
        st.warning("문장 분리가 되지 않았습니다. 영어 금융 기사 문장을 넣어 보세요.")
        st.stop()

    df_result = predict_sentences(sentences, model, tokenizer, max_len)
    summary = summarize_document(df_result)
    ratio_df = make_ratio_df(summary)

    st.markdown("## 분석 결과")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("전체 문서 감성", summary["overall_label"])
    c2.metric("문장 수", summary["total_sentences"])
    c3.metric("Negative 비율", f"{summary['negative_ratio']*100:.1f}%")
    c4.metric("Neutral 비율", f"{summary['neutral_ratio']*100:.1f}%")
    c5.metric("Positive 비율", f"{summary['positive_ratio']*100:.1f}%")

    st.info(summary["overall_reason"])
    st.write(sentence_level_comment(df_result))

    tab1, tab2, tab3, tab4 = st.tabs([
        "전체 요약",
        "문장별 분석",
        "감성 흐름",
        "원문 하이라이트"
    ])

    with tab1:
        st.subheader("전체 문서 요약")
        st.write(f"- 전체 문서 판단: **{summary['overall_label']}**")
        st.write(f"- 해석: {summary['overall_reason']}")
        st.write(f"- 총 문장 수: {summary['total_sentences']}개")

        st.subheader("감성 비율")
        fig = plt.figure(figsize=(7, 4))
        plt.bar(ratio_df["label"], ratio_df["ratio"])
        plt.ylim(0, 1)
        plt.ylabel("ratio")
        plt.title("문서 내 감성 비율")
        st.pyplot(fig)

        st.subheader("상위 부정 문장")
        neg_top = (
            df_result[df_result["pred_label"] == "negative"]
            .sort_values(["negative_prob", "confidence"], ascending=False)
            .head(5)
        )
        if len(neg_top) > 0:
            st.dataframe(
                neg_top[["sentence_no", "sentence", "pred_label", "confidence", "negative_prob"]],
                use_container_width=True
            )
        else:
            st.write("예측된 negative 문장이 없습니다.")

        st.subheader("상위 긍정 문장")
        pos_top = (
            df_result[df_result["pred_label"] == "positive"]
            .sort_values(["positive_prob", "confidence"], ascending=False)
            .head(5)
        )
        if len(pos_top) > 0:
            st.dataframe(
                pos_top[["sentence_no", "sentence", "pred_label", "confidence", "positive_prob"]],
                use_container_width=True
            )
        else:
            st.write("예측된 positive 문장이 없습니다.")

    with tab2:
        st.subheader("문장별 감성 분석")
        st.dataframe(df_result, use_container_width=True)

        csv = df_result.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "문장별 분석 결과 CSV 다운로드",
            data=csv,
            file_name="financial_sentiment_sentence_analysis.csv",
            mime="text/csv"
        )

    with tab3:
        st.subheader("문장 순서별 감성 흐름")
        flow_df = df_result[["sentence_no", "sentiment_score", "negative_prob", "neutral_prob", "positive_prob"]]

        fig1 = plt.figure(figsize=(10, 4))
        plt.plot(flow_df["sentence_no"], flow_df["sentiment_score"], marker="o")
        plt.title("문장 순서별 감성 점수 흐름")
        plt.xlabel("sentence_no")
        plt.ylabel("sentiment_score (-1: negative, 0: neutral, 1: positive)")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig1)

        fig2 = plt.figure(figsize=(10, 4))
        plt.plot(flow_df["sentence_no"], flow_df["negative_prob"], marker="o", label="negative_prob")
        plt.plot(flow_df["sentence_no"], flow_df["neutral_prob"], marker="o", label="neutral_prob")
        plt.plot(flow_df["sentence_no"], flow_df["positive_prob"], marker="o", label="positive_prob")
        plt.title("문장 순서별 감성 확률 흐름")
        plt.xlabel("sentence_no")
        plt.ylabel("probability")
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig2)

    with tab4:
        st.subheader("원문 하이라이트")
        st.markdown(build_highlighted_text(df_result), unsafe_allow_html=True)

    with st.expander("분석 원문 보기", expanded=False):
        st.text(text_input)

else:
    st.markdown("### 사용 방법")
    st.write("1. 영어 금융 뉴스나 리포트 텍스트를 붙여넣습니다.")
    st.write("2. 원하는 모델 버튼을 눌러 적용합니다.")
    st.write("3. 분석 실행 버튼을 눌러 결과를 확인합니다.")
    st.write("4. 현재 어떤 모델이 연결되어 있는지 우측 상태 패널과 좌측 배지로 확인합니다.")