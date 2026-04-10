from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


st.set_page_config(
    page_title="한국어 독성 댓글 탐지",
    page_icon="🛡️",
    layout="wide",
)

ARTIFACT_ROOT = Path("mk2/light/artifacts")
MODEL_DIR = ARTIFACT_ROOT / "model"
DATA_DIR = ARTIFACT_ROOT / "data"
IMAGE_DIR = ARTIFACT_ROOT / "images"

LABEL_MAP = {0: "normal", 1: "toxic"}
EXAMPLE_TEXTS = [
    "오늘 방송 너무 재밌어요!",
    "은근히 나대네 기분나쁘게",
    "설명 잘해주셔서 이해가 잘 됐어요.",
    "왜 이렇게 공격적으로 말해?",
    "벌려 이뇨나",
    "실수는 있었지만 다음에는 더 잘할 수 있을 것 같아요.",
]


def softmax_np(logits):
    import numpy as np

    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


@st.cache_data
def load_json(path: str):
    file_path = Path(path)
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path: str):
    file_path = Path(path)
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)


@st.cache_resource
def load_model_and_tokenizer(model_dir: str):
    model_path = Path(model_dir)
    if not model_path.exists():
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def predict_text(text: str, tokenizer, model, max_length: int = 64):
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits.cpu().numpy()
    probs = softmax_np(logits)[0]
    pred_id = int(probs.argmax())
    return {
        "label_id": pred_id,
        "label_name": LABEL_MAP[pred_id],
        "prob_normal": float(probs[0]),
        "prob_toxic": float(probs[1]),
    }


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div style="
            border:1px solid #D9D9D9;
            border-radius:16px;
            padding:16px 18px;
            background-color:#FFFFFF;
            box-shadow:0 4px 12px rgba(0,0,0,0.04);
            min-height:96px;
        ">
            <div style="font-size:14px;color:#666;">{label}</div>
            <div style="font-size:28px;font-weight:700;margin-top:8px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


dataset_summary = load_json(DATA_DIR / "dataset_summary.json")
metrics = load_json(DATA_DIR / "metrics.json")
classification_report = load_json(DATA_DIR / "classification_report.json")
pred_df = load_csv(DATA_DIR / "predictions_test.csv")
fp_df = load_csv(DATA_DIR / "false_positives.csv")
fn_df = load_csv(DATA_DIR / "false_negatives.csv")
history_df = load_csv(DATA_DIR / "training_history.csv")

tokenizer, model = load_model_and_tokenizer(MODEL_DIR)

st.title("한국어 독성 댓글 탐지 시스템")
st.caption("KOLD 데이터셋 + 한국어 ELECTRA 기반 이진 분류 / Streamlit Cloud 서비스용 화면 (CPU 경량 학습 아티팩트 호환)")

with st.sidebar:
    st.header("서비스 개요")
    st.write("- 입력 댓글을 normal / toxic으로 분류")
    st.write("- 확률 기반 결과 표시")
    st.write("- 테스트 성능과 시각화 결과 함께 제공")
    st.write("- Streamlit Cloud 배포를 전제로 한 서비스 화면")

    st.divider()
    st.subheader("아티팩트 상태")
    st.write(f"모델 디렉터리 존재: {'예' if MODEL_DIR.exists() else '아니오'}")
    st.write(f"평가 지표 파일 존재: {'예' if (DATA_DIR / 'metrics.json').exists() else '아니오'}")
    st.write(f"시각화 이미지 존재: {'예' if IMAGE_DIR.exists() else '아니오'}")


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "프로젝트 개요",
        "댓글 예측",
        "모델 성능 및 시각화",
        "오분류 사례 분석",
    ]
)

with tab1:
    st.subheader("프로젝트 개요")
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown(
            """
            ### 주제
            **한국어 독성 댓글 탐지 모델 개발 및 시각화**

            ### 데이터셋 요약
            - 데이터셋: **KOLD**
            - 입력: `comment`
            - 라벨: `OFF`
            - 이진 분류 기준  
              - `OFF=False` → `normal`  
              - `OFF=True` → `toxic`

            ### 학습 전략
            - Stratified Train / Validation / Test 분할
            - 한국어 ELECTRA small 계열 모델 사용
            - CPU 환경을 고려한 축소 학습셋 사용
            - Weighted Cross Entropy로 클래스 불균형 보완
            - 학습 결과 기반 모델 저장 및 평가 시각화 제공

            ### 구현 구조
            - 학습 노트북에서 모델/그래프/예측 결과 CSV 저장
            - Streamlit 서비스에서 저장된 모델과 평가 결과를 재사용
            """
        )

    with col2:
        if dataset_summary:
            st.markdown("### 데이터셋 규모")
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("원본 KOLD 전체", f"{dataset_summary.get('full_total_samples', dataset_summary.get('total_samples', 0)):,}")
            with c2:
                metric_card("실학습 사용 샘플", f"{dataset_summary.get('total_samples', 0):,}")
            with c3:
                label_info = dataset_summary.get("label_distribution_named", {})
                toxic_count = label_info.get("toxic", 0)
                metric_card("실학습 독성 샘플 수", f"{toxic_count:,}")

            st.markdown("### 분할 정보")
            used_split = dataset_summary.get("used_split_samples")
            full_split = dataset_summary.get("full_split_samples")
            if used_split and full_split:
                split_df = pd.DataFrame(
                    {
                        "split": ["train", "validation", "test"],
                        "원본 분할 수": [
                            full_split.get("train", 0),
                            full_split.get("validation", 0),
                            full_split.get("test", 0),
                        ],
                        "실학습 사용 수": [
                            used_split.get("train", 0),
                            used_split.get("validation", 0),
                            used_split.get("test", 0),
                        ],
                    }
                )
            else:
                split_df = pd.DataFrame(
                    {
                        "split": ["train", "validation", "test"],
                        "samples": [
                            dataset_summary.get("train_samples", 0),
                            dataset_summary.get("validation_samples", 0),
                            dataset_summary.get("test_samples", 0),
                        ],
                    }
                )
            st.dataframe(split_df, use_container_width=True, hide_index=True)
        else:
            st.info("학습 노트북을 먼저 실행하면 데이터셋 요약 정보가 표시됩니다.")

with tab2:
    st.subheader("댓글 예측")
    if tokenizer is None or model is None:
        st.error("학습된 모델이 없습니다. 먼저 노트북에서 모델을 저장해 주세요.")
    else:
        selected_example = st.selectbox("예시 문장 선택", ["직접 입력"] + EXAMPLE_TEXTS)
        default_text = "" if selected_example == "직접 입력" else selected_example
        user_text = st.text_area("댓글 입력", value=default_text, height=120)

        if st.button("예측 실행", type="primary", use_container_width=True):
            if not user_text.strip():
                st.warning("댓글을 입력해 주세요.")
            else:
                result = predict_text(user_text.strip(), tokenizer, model)

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    metric_card("예측 결과", result["label_name"])
                with col_b:
                    metric_card("normal 확률", f"{result['prob_normal']:.4f}")
                with col_c:
                    metric_card("toxic 확률", f"{result['prob_toxic']:.4f}")

                st.markdown("### 확률 분포")
                prob_df = pd.DataFrame(
                    {
                        "label": ["normal", "toxic"],
                        "probability": [result["prob_normal"], result["prob_toxic"]],
                    }
                )
                st.bar_chart(prob_df.set_index("label"))

                if result["label_name"] == "toxic":
                    st.warning("입력 문장은 독성 댓글로 분류되었습니다.")
                else:
                    st.success("입력 문장은 정상 댓글로 분류되었습니다.")

with tab3:
    st.subheader("모델 성능 및 시각화")

    if metrics is not None:
        test_metrics = metrics.get("test_metrics", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_card("Accuracy", f"{test_metrics.get('eval_accuracy', 0):.4f}")
        with c2:
            metric_card("Precision", f"{test_metrics.get('eval_precision', 0):.4f}")
        with c3:
            metric_card("Recall", f"{test_metrics.get('eval_recall', 0):.4f}")
        with c4:
            metric_card("F1", f"{test_metrics.get('eval_f1', 0):.4f}")
        with c5:
            auc_value = test_metrics.get("eval_roc_auc")
            metric_card(
                "ROC AUC",
                f"{auc_value:.4f}" if isinstance(auc_value, (float, int)) and auc_value is not None else "-",
            )
    else:
        st.info("metrics.json 이 존재하지 않습니다. 학습 노트북을 먼저 실행해 주세요.")

    st.markdown("### 평가 그래프")
    image_pairs = [
        ("Confusion Matrix", IMAGE_DIR / "confusion_matrix.png"),
        ("ROC Curve", IMAGE_DIR / "roc_curve.png"),
        ("Actual vs Predicted", IMAGE_DIR / "actual_vs_predicted.png"),
        ("Training History", IMAGE_DIR / "training_history.png"),
    ]

    for title, path in image_pairs:
        st.markdown(f"#### {title}")
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.info(f"{path.name} 파일이 아직 생성되지 않았습니다.")

    if classification_report is not None:
        st.markdown("### Classification Report")
        report_df = pd.DataFrame(classification_report).transpose()
        st.dataframe(report_df, use_container_width=True)
    else:
        st.info("classification_report.json 파일이 없습니다.")

with tab4:
    st.subheader("오분류 사례 분석")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### False Positive")
        if fp_df is not None and not fp_df.empty:
            st.dataframe(
                fp_df[["text", "actual_label", "predicted_label", "prob_toxic"]].head(20),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("False Positive 사례가 없거나 파일이 없습니다.")

    with col2:
        st.markdown("### False Negative")
        if fn_df is not None and not fn_df.empty:
            st.dataframe(
                fn_df[["text", "actual_label", "predicted_label", "prob_toxic"]].head(20),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("False Negative 사례가 없거나 파일이 없습니다.")

    st.markdown("### 테스트셋 예측 결과 샘플")
    if pred_df is not None and not pred_df.empty:
        view_cols = [
            "text",
            "actual_label",
            "predicted_label",
            "prob_normal",
            "prob_toxic",
            "correct",
        ]
        st.dataframe(pred_df[view_cols].head(30), use_container_width=True, hide_index=True)
    else:
        st.info("predictions_test.csv 파일이 없습니다.")
