"""
MLOps Platform — Production-Grade Machine Learning Lifecycle Management
=======================================================================
Features:
  • Experiment tracking & management
  • Multi-model training with hyperparameter tuning
  • Model registry with staging/production promotion
  • Data drift detection (KS-test + PSI)
  • Automated retraining jobs
  • Monitoring dashboards with Plotly
  • Full audit/pipeline log

Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json, time, logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timezone
from pathlib import Path

from app.database import (
    init_db, get_dashboard_stats,
    create_experiment, get_experiments, get_experiment, delete_experiment,
    create_run, update_run, get_runs, get_all_runs, get_run,
    register_model, promote_model, get_models, get_model,
    save_dataset, get_datasets,
    save_drift_report, get_drift_reports,
    create_retraining_job, update_retraining_job, get_retraining_jobs,
    log_event, get_logs,
)
from app.trainer import (
    get_builtin_dataset, compute_dataset_stats,
    train_model, detect_drift, generate_drifted_data,
    CLASSIFICATION_MODELS, REGRESSION_MODELS, CLUSTERING_MODELS, DEFAULT_PARAMS,
)

# ── Init ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
init_db()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLOps Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:      #0F1117;
    --surface: #1A1D27;
    --surface2:#22263A;
    --border:  #2E3250;
    --accent:  #6C63FF;
    --accent2: #A78BFA;
    --green:   #10B981;
    --red:     #EF4444;
    --orange:  #F59E0B;
    --blue:    #3B82F6;
    --text:    #F1F5F9;
    --muted:   #94A3B8;
}

html,body,[class*="css"]{
    font-family:'Inter',sans-serif!important;
    background:var(--bg)!important;
    color:var(--text)!important;
}

[data-testid="stSidebar"]{
    background:var(--surface)!important;
    border-right:1px solid var(--border)!important;
}
[data-testid="stSidebar"] *{color:var(--text)!important;}

.stApp{background:var(--bg)!important;}
.block-container{padding:1.5rem 2rem!important;max-width:1400px!important;}

h1{font-size:1.7rem!important;font-weight:700!important;color:var(--text)!important;}
h2{font-size:1.1rem!important;font-weight:700!important;color:var(--accent2)!important;
   text-transform:uppercase;letter-spacing:.08em;}
h3{font-size:.95rem!important;font-weight:600!important;color:var(--text)!important;}

/* Metric cards */
.metric-card{
    background:var(--surface);border:1px solid var(--border);
    border-radius:12px;padding:1.2rem 1.4rem;
    box-shadow:0 2px 8px rgba(0,0,0,.3);
}
.metric-val{font-size:2rem;font-weight:800;color:var(--accent2);}
.metric-lbl{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:3px;}
.metric-delta{font-size:.8rem;margin-top:4px;}
.delta-up{color:var(--green);}
.delta-dn{color:var(--red);}

/* Badges */
.badge{display:inline-block;padding:2px 9px;border-radius:20px;
       font-size:.7rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;margin-right:4px;}
.b-green {background:rgba(16,185,129,.15);color:#34D399;border:1px solid rgba(16,185,129,.3);}
.b-red   {background:rgba(239,68,68,.15); color:#F87171;border:1px solid rgba(239,68,68,.3);}
.b-orange{background:rgba(245,158,11,.15);color:#FCD34D;border:1px solid rgba(245,158,11,.3);}
.b-blue  {background:rgba(59,130,246,.15);color:#93C5FD;border:1px solid rgba(59,130,246,.3);}
.b-purple{background:rgba(108,99,255,.15);color:#A78BFA;border:1px solid rgba(108,99,255,.3);}
.b-gray  {background:rgba(148,163,184,.1);color:var(--muted);border:1px solid var(--border);}

/* Tables */
.run-row{
    background:var(--surface);border:1px solid var(--border);
    border-radius:8px;padding:.8rem 1rem;margin-bottom:.5rem;
    display:flex;justify-content:space-between;align-items:center;
}

/* Inputs */
.stTextArea textarea,.stTextInput input,.stNumberInput input{
    background:var(--surface2)!important;border:1px solid var(--border)!important;
    color:var(--text)!important;border-radius:8px!important;
    font-family:'Inter',sans-serif!important;
}
.stTextArea textarea:focus,.stTextInput input:focus{
    border-color:var(--accent)!important;
    box-shadow:0 0 0 2px rgba(108,99,255,.2)!important;
}
label{color:var(--muted)!important;font-size:.82rem!important;font-weight:500!important;}

/* Buttons */
.stButton>button{
    background:linear-gradient(135deg,#6C63FF,#A78BFA)!important;
    color:#fff!important;border:none!important;border-radius:8px!important;
    font-weight:600!important;font-size:.88rem!important;
    padding:.5rem 1.3rem!important;
    box-shadow:0 2px 10px rgba(108,99,255,.3)!important;
    transition:all .18s!important;
}
.stButton>button:hover{
    transform:translateY(-1px)!important;
    box-shadow:0 4px 18px rgba(108,99,255,.45)!important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
    background:var(--surface)!important;border-radius:10px!important;
    border:1px solid var(--border)!important;gap:2px;padding:4px;
}
.stTabs [data-baseweb="tab"]{
    background:transparent!important;color:var(--muted)!important;
    border-radius:7px!important;font-weight:500!important;font-size:.84rem!important;
}
.stTabs [aria-selected="true"]{
    background:var(--accent)!important;color:#fff!important;font-weight:700!important;
}

/* Select */
.stSelectbox [data-baseweb="select"]>div{
    background:var(--surface2)!important;border-color:var(--border)!important;
    color:var(--text)!important;border-radius:8px!important;
}

/* Expander */
[data-testid="stExpander"]{
    border:1px solid var(--border)!important;
    border-radius:10px!important;background:var(--surface)!important;
}

/* Slider */
.stSlider .stSlider{color:var(--accent)!important;}

/* HR */
hr{border-color:var(--border)!important;}

/* Scrollbar */
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}

/* Alert / info boxes */
.stAlert{border-radius:10px!important;}
.info-box{background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.25);
          border-radius:10px;padding:.8rem 1rem;margin-bottom:.8rem;font-size:.83rem;color:#93C5FD;}
.warn-box{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);
          border-radius:10px;padding:.8rem 1rem;margin-bottom:.8rem;font-size:.83rem;color:#FCD34D;}
.success-box{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.25);
             border-radius:10px;padding:.8rem 1rem;margin-bottom:.8rem;font-size:.83rem;color:#34D399;}
.error-box{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);
           border-radius:10px;padding:.8rem 1rem;margin-bottom:.8rem;font-size:.83rem;color:#F87171;}

/* Log row */
.log-row{font-size:.77rem;padding:.35rem .5rem;border-bottom:1px solid var(--border);
         font-family:'JetBrains Mono',monospace;}
.log-INFO{color:#93C5FD;}.log-WARNING{color:#FCD34D;}.log-ERROR{color:#F87171;}.log-SUCCESS{color:#34D399;}

.footer{font-size:.7rem;color:var(--muted);text-align:center;
        margin-top:2rem;border-top:1px solid var(--border);padding-top:1rem;}

/* Progress bar */
.stProgress>div>div{background:var(--accent)!important;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def badge(text, cls):
    return f'<span class="badge b-{cls}">{text}</span>'

def status_badge(s):
    c = {"completed":"green","running":"blue","failed":"red",
         "pending":"gray","cancelled":"gray"}.get(s,"gray")
    return badge(s.upper(), c)

def stage_badge(s):
    c = {"production":"green","staging":"orange","archived":"gray","champion":"purple"}.get(s,"gray")
    return badge(s.upper(), c)

def _metric_card(col, val, label, color="#A78BFA", delta=None, delta_label=""):
    delta_html = ""
    if delta is not None:
        cls = "delta-up" if delta >= 0 else "delta-dn"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f'<div class="metric-delta {cls}">{arrow} {abs(delta):.2f} {delta_label}</div>'
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-val" style="color:{color}">{val}</div>
        <div class="metric-lbl">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def plotly_theme():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#F1F5F9"),
        xaxis=dict(gridcolor="#2E3250", showgrid=True),
        yaxis=dict(gridcolor="#2E3250", showgrid=True),
        margin=dict(l=40, r=20, t=40, b=40),
    )

PALETTE = ["#6C63FF","#10B981","#F59E0B","#EF4444","#3B82F6","#EC4899","#8B5CF6","#14B8A6"]

ALL_DATASETS = [
    "Iris (Classification)",
    "Wine Quality (Classification)",
    "Breast Cancer (Classification)",
    "Diabetes (Regression)",
    "Synthetic Classification",
    "Synthetic Regression",
    "Synthetic Clustering",
]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.2rem;border-bottom:1px solid #2E3250;margin-bottom:1rem">
        <div style="display:flex;align-items:center;gap:10px">
            <div style="font-size:1.8rem">🤖</div>
            <div>
                <div style="font-weight:800;font-size:1.1rem;color:#F1F5F9">MLOps Platform</div>
                <div style="font-size:.68rem;color:#94A3B8;text-transform:uppercase;letter-spacing:.06em">
                    v1.0 · Production Ready
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox("Navigation", [
        "🏠 Dashboard",
        "🧪 Experiments",
        "🚀 Train Model",
        "📦 Model Registry",
        "📊 Data Management",
        "🔍 Drift Detection",
        "🔄 Auto Retraining",
        "📈 Monitoring",
        "📋 Pipeline Logs",
    ], label_visibility="collapsed")

    st.markdown("---")
    stats = get_dashboard_stats()
    st.markdown(f"""
    <div style="font-size:.78rem;color:#94A3B8;line-height:2">
        <div>🧪 Experiments: <b style="color:#A78BFA">{stats['total_experiments']}</b></div>
        <div>🏃 Total Runs: <b style="color:#93C5FD">{stats['total_runs']}</b></div>
        <div>📦 Models: <b style="color:#34D399">{stats['total_models']}</b></div>
        <div>🚀 In Production: <b style="color:#FCD34D">{stats['production_models']}</b></div>
        <div>⚠️ Drift Alerts: <b style="color:#F87171">{stats['drift_alerts']}</b></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:.7rem;color:#94A3B8">Built with Streamlit · scikit-learn · SQLite · Plotly</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("# 🤖 MLOps Platform — Dashboard")
    st.markdown(f'<p style="color:#94A3B8;margin-top:0">Last updated: {datetime.now().strftime("%d %b %Y, %H:%M:%S")}</p>',
                unsafe_allow_html=True)

    # ── KPI tiles ──
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    _metric_card(c1, stats["total_experiments"], "Experiments",      "#A78BFA")
    _metric_card(c2, stats["total_runs"],        "Total Runs",       "#93C5FD")
    _metric_card(c3, stats["successful_runs"],   "Successful Runs",  "#34D399")
    _metric_card(c4, stats["failed_runs"],       "Failed Runs",      "#F87171")
    _metric_card(c5, stats["production_models"], "In Production",    "#FCD34D")
    _metric_card(c6, stats["drift_alerts"],      "Drift Alerts",     "#F87171")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    col_a, col_b = st.columns(2)

    with col_a:
        runs = get_all_runs()
        if runs:
            st.markdown("#### Recent Run Performance")
            df_runs = pd.DataFrame(runs[:30])
            status_counts = df_runs["status"].value_counts().reset_index()
            status_counts.columns = ["status","count"]
            fig = px.pie(status_counts, names="status", values="count",
                         color_discrete_sequence=PALETTE, hole=0.5)
            fig.update_layout(**plotly_theme(), showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=-0.2))
            fig.update_traces(textinfo="label+percent")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("No runs yet. Go to 🚀 Train Model to get started.")

    with col_b:
        if runs:
            st.markdown("#### Model Accuracy / F1 Over Runs")
            df_runs = pd.DataFrame(runs)
            df_runs["accuracy"] = df_runs["metrics"].apply(
                lambda m: m.get("accuracy") or m.get("r2_score") or m.get("silhouette_score") or 0
                if isinstance(m, dict) else 0
            )
            df_runs["run_label"] = df_runs["run_name"].str[:20]
            df_plot = df_runs[df_runs["accuracy"]>0].tail(20)
            if not df_plot.empty:
                fig2 = px.bar(df_plot, x="run_label", y="accuracy",
                              color="model_type", color_discrete_sequence=PALETTE)
                fig2.update_layout(**plotly_theme(), xaxis_tickangle=-35,
                                   yaxis_title="Score", xaxis_title="")
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
            else:
                st.info("Train some models to see metrics here.")
        else:
            st.info("No runs yet.")

    # ── Recent Activity ──
    st.markdown("#### Recent Activity")
    logs = get_logs(20)
    if logs:
        for log in logs[:10]:
            lvl = log.get("level","INFO")
            ts  = log.get("created_at","")[:19].replace("T"," ")
            comp= log.get("component","")
            msg = log.get("message","")
            cls = f"log-{lvl}"
            st.markdown(f'<div class="log-row"><span style="color:#94A3B8">{ts}</span> '
                        f'<span class="{cls}">[{lvl}]</span> '
                        f'<span style="color:#A78BFA">[{comp}]</span> {msg}</div>',
                        unsafe_allow_html=True)
    else:
        st.info("No activity yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Experiments":
    st.markdown("# 🧪 Experiment Tracking")

    tab_list, tab_create = st.tabs(["📋 All Experiments", "➕ New Experiment"])

    with tab_create:
        st.markdown("### Create New Experiment")
        with st.form("new_exp"):
            name  = st.text_input("Experiment Name", placeholder="e.g. fraud-detection-v1")
            desc  = st.text_area("Description", placeholder="What are you trying to solve?", height=80)
            ttype = st.selectbox("Task Type", ["classification","regression","clustering"])
            if st.form_submit_button("✨ Create Experiment", use_container_width=True):
                if name.strip():
                    try:
                        eid = create_experiment(name.strip(), desc.strip(), ttype)
                        log_event("INFO","Experiments",f"Created experiment '{name}'",{"id":eid})
                        st.success(f"✅ Experiment '{name}' created! ID: {eid}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Please enter an experiment name.")

    with tab_list:
        exps = get_experiments()
        if not exps:
            st.info("No experiments yet. Create one above.")
        else:
            for exp in exps:
                with st.expander(
                    f"**{exp['name']}** · {exp['task_type'].upper()} · {exp['run_count']} runs",
                    expanded=False
                ):
                    c1,c2,c3 = st.columns([2,1,1])
                    with c1:
                        st.markdown(f"**Description:** {exp.get('description','—')}")
                        st.markdown(f"**Created:** {exp['created_at'][:16].replace('T',' ')}")
                    with c2:
                        st.markdown(f"**Task:** {badge(exp['task_type'].upper(),'purple')}", unsafe_allow_html=True)
                        st.markdown(f"**Runs:** {exp['run_count']}")
                    with c3:
                        if st.button("🗑 Delete", key=f"del_exp_{exp['id']}"):
                            delete_experiment(exp["id"])
                            log_event("WARNING","Experiments",f"Deleted experiment '{exp['name']}'")
                            st.rerun()

                    # Runs table
                    runs = get_runs(exp["id"])
                    if runs:
                        st.markdown("##### Runs")
                        for r in runs[:10]:
                            m = r.get("metrics",{})
                            score_key = next((k for k in ["accuracy","r2_score","f1_score","silhouette_score"] if k in m), None)
                            score_str = f"{score_key}: {m[score_key]:.4f}" if score_key else "—"
                            st.markdown(
                                f'<div class="run-row">'
                                f'<span><b>{r["run_name"]}</b> · {r["model_type"]}</span>'
                                f'<span>{status_badge(r["status"])} {score_str} '
                                f'· {r.get("duration_sec",0) or 0:.1f}s</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚀 Train Model":
    st.markdown("# 🚀 Train Model")

    exps = get_experiments()
    if not exps:
        st.warning("Create an experiment first in 🧪 Experiments.")
        st.stop()

    col_cfg, col_res = st.columns([1,1.2], gap="large")

    with col_cfg:
        st.markdown("### ⚙️ Configuration")

        # Experiment
        exp_names = {e["name"]: e["id"] for e in exps}
        exp_name  = st.selectbox("Experiment", list(exp_names.keys()))
        exp_id    = exp_names[exp_name]
        exp_info  = get_experiment(exp_id)
        task_type = exp_info["task_type"] if exp_info else "classification"

        st.markdown(f'Task: {badge(task_type.upper(),"purple")}', unsafe_allow_html=True)

        # Dataset
        st.markdown("#### Dataset")
        ds_source = st.radio("Source", ["Built-in", "Upload CSV"], horizontal=True)

        df_data, target_col = None, None
        if ds_source == "Built-in":
            # Filter datasets by task type
            filtered = [d for d in ALL_DATASETS if task_type in d.lower()]
            if not filtered:
                filtered = ALL_DATASETS
            ds_name = st.selectbox("Dataset", filtered)
            if st.button("📂 Load Dataset"):
                with st.spinner("Loading…"):
                    df_data, target_col, detected_task = get_builtin_dataset(ds_name)
                    st.session_state["df_data"]    = df_data
                    st.session_state["target_col"] = target_col
                    st.session_state["ds_name"]    = ds_name
                    # Save dataset record
                    stats_d = compute_dataset_stats(df_data, target_col)
                    save_dataset(ds_name, "v1", task_type,
                                 len(df_data), len(df_data.columns)-1,
                                 target_col,
                                 [c for c in df_data.columns if c!=target_col],
                                 stats_d, "")
                    log_event("INFO","DataManagement",f"Loaded dataset '{ds_name}'",
                              {"samples":len(df_data),"features":len(df_data.columns)-1})
                    st.success(f"✅ Loaded: {df_data.shape[0]} rows × {df_data.shape[1]} cols")
        else:
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded:
                df_data = pd.read_csv(uploaded)
                st.session_state["df_data"] = df_data
                cols = df_data.columns.tolist()
                target_col = st.selectbox("Target Column", cols, index=len(cols)-1)
                st.session_state["target_col"] = target_col
                st.session_state["ds_name"]    = uploaded.name

        # Use cached
        if "df_data" in st.session_state and df_data is None:
            df_data    = st.session_state["df_data"]
            target_col = st.session_state["target_col"]

        if df_data is not None:
            st.markdown(f'`{df_data.shape[0]} rows × {df_data.shape[1]} cols` · target: `{target_col}`')

        # Model selection
        st.markdown("#### Model")
        if task_type == "classification":
            model_options = list(CLASSIFICATION_MODELS.keys())
        elif task_type == "regression":
            model_options = list(REGRESSION_MODELS.keys())
        else:
            model_options = list(CLUSTERING_MODELS.keys())

        model_name = st.selectbox("Algorithm", model_options)

        # Hyperparameters
        st.markdown("#### Hyperparameters")
        defaults = DEFAULT_PARAMS.get(model_name, {})
        params   = {}
        if defaults:
            with st.expander("Edit hyperparameters", expanded=False):
                for k, v in defaults.items():
                    if isinstance(v, bool):
                        params[k] = st.checkbox(k, value=v)
                    elif isinstance(v, int):
                        params[k] = st.number_input(k, value=v, step=1)
                    elif isinstance(v, float):
                        params[k] = st.number_input(k, value=v, format="%.4f")
                    elif v is None:
                        params[k] = None
                    else:
                        params[k] = st.text_input(k, value=str(v))
        else:
            st.info("No configurable hyperparameters for this model.")

        # Training options
        st.markdown("#### Training Options")
        col_o1, col_o2 = st.columns(2)
        test_size  = col_o1.slider("Test Split", 0.1, 0.4, 0.2, 0.05)
        cv_folds   = col_o2.slider("CV Folds", 3, 10, 5)
        tune_hp    = st.checkbox("🔧 Hyperparameter Tuning", value=False)
        if tune_hp:
            tune_method = st.radio("Method", ["grid","random"], horizontal=True)
            n_iter = st.slider("Iterations (random only)", 5, 50, 20) if tune_method=="random" else 20
        else:
            tune_method, n_iter = "grid", 20

        run_name = st.text_input("Run Name", value=f"{model_name}-{datetime.now().strftime('%H%M%S')}")

        register_after = st.checkbox("📦 Register model after training", value=True)
        model_reg_name = st.text_input("Model Registry Name",
                                       value=exp_name.replace(" ","-").lower()) if register_after else ""

        # Train button
        st.markdown("---")
        train_btn = st.button("🚀 Start Training", use_container_width=True,
                              disabled=(df_data is None))

    with col_res:
        st.markdown("### 📊 Training Results")

        if train_btn and df_data is not None:
            run_id = create_run(exp_id, run_name, model_name, params)
            log_event("INFO","Training",f"Started run '{run_name}'",
                      {"model":model_name,"experiment":exp_name,"run_id":run_id})

            progress = st.progress(0, text="Preprocessing data…")
            status_box = st.empty()

            try:
                progress.progress(20, text="Building pipeline…")
                time.sleep(0.3)
                progress.progress(40, text=f"Training {model_name}…")

                result = train_model(
                    experiment_id=exp_id,
                    run_id=run_id,
                    df=df_data,
                    target_col=target_col,
                    task_type=task_type,
                    model_name=model_name,
                    params=params,
                    test_size=test_size,
                    cv_folds=cv_folds,
                    tune_hyperparams=tune_hp,
                    tune_method=tune_method,
                    n_iter=n_iter,
                )

                progress.progress(80, text="Computing metrics…")
                metrics  = result["metrics"]
                artifact = result["artifact_path"]
                duration = result["duration"]
                feat_imp = result["feature_importance"]

                update_run(run_id, "completed", metrics, duration, artifact)
                progress.progress(100, text="Done!")
                log_event("SUCCESS","Training",f"Run '{run_name}' completed",
                          {"metrics":metrics,"duration":duration})

                # Register model
                if register_after and model_reg_name:
                    mid = register_model(model_reg_name, exp_id, run_id,
                                         metrics, params, artifact,
                                         f"Trained by {run_name}")
                    log_event("INFO","ModelRegistry",f"Registered model '{model_reg_name}'",
                              {"model_id":mid,"version":1})
                    status_box.markdown(
                        f'<div class="success-box">✅ Training complete in {duration}s · '
                        f'Model registered as <b>{model_reg_name}</b> (ID: {mid})</div>',
                        unsafe_allow_html=True
                    )
                else:
                    status_box.markdown(
                        f'<div class="success-box">✅ Training complete in {duration}s</div>',
                        unsafe_allow_html=True
                    )

                # Show metrics
                st.markdown("#### Metrics")
                display_metrics = {k:v for k,v in metrics.items()
                                   if k not in ["confusion_matrix","test_size","train_size"]}
                m_cols = st.columns(min(len(display_metrics), 4))
                colors_cycle = ["#A78BFA","#34D399","#93C5FD","#FCD34D","#F87171"]
                for i,(k,v) in enumerate(display_metrics.items()):
                    _metric_card(m_cols[i%4], f"{v:.4f}" if isinstance(v,float) else v,
                                 k.replace("_"," ").title(), colors_cycle[i%5])

                # Confusion matrix
                if "confusion_matrix" in metrics:
                    st.markdown("#### Confusion Matrix")
                    cm = np.array(metrics["confusion_matrix"])
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Viridis",
                                       aspect="auto")
                    fig_cm.update_layout(**plotly_theme(), title="")
                    st.plotly_chart(fig_cm, use_container_width=True,
                                    config={"displayModeBar":False})

                # Feature importance
                if feat_imp:
                    st.markdown("#### Feature Importance")
                    df_fi = pd.DataFrame(list(feat_imp.items()),
                                         columns=["Feature","Importance"]).head(15)
                    fig_fi = px.bar(df_fi.sort_values("Importance"),
                                    x="Importance", y="Feature",
                                    orientation="h", color="Importance",
                                    color_continuous_scale="Viridis")
                    fig_fi.update_layout(**plotly_theme(), yaxis_title="",
                                         showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(fig_fi, use_container_width=True,
                                    config={"displayModeBar":False})

            except Exception as e:
                update_run(run_id, "failed", {}, 0.0)
                log_event("ERROR","Training",f"Run '{run_name}' failed: {str(e)}")
                progress.progress(100, text="Failed")
                st.error(f"Training failed: {e}")

        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem;color:#94A3B8">
                <div style="font-size:3rem;margin-bottom:1rem">🤖</div>
                <div style="font-size:1rem;font-weight:600">Configure & run training on the left</div>
                <div style="font-size:.85rem;margin-top:.5rem">Results will appear here</div>
            </div>
            """, unsafe_allow_html=True)

        # Previous runs for this experiment
        if exps:
            st.markdown("---")
            st.markdown("#### Previous Runs")
            sel_exp = st.selectbox("Show runs for", [e["name"] for e in exps],
                                    key="prev_runs_exp", label_visibility="collapsed")
            sel_id  = next(e["id"] for e in exps if e["name"]==sel_exp)
            prev    = get_runs(sel_id)
            if prev:
                df_prev = pd.DataFrame([{
                    "Run": r["run_name"][:25],
                    "Model": r["model_type"],
                    "Status": r["status"],
                    "Score": next((f"{v:.4f}" for k,v in r.get("metrics",{}).items()
                                   if k in ["accuracy","r2_score","f1_score","silhouette_score"]
                                   and isinstance(v,float)), "—"),
                    "Duration": f"{r.get('duration_sec',0) or 0:.1f}s",
                    "Date": r["created_at"][:16].replace("T"," "),
                } for r in prev[:10]])
                st.dataframe(df_prev, use_container_width=True, hide_index=True)
            else:
                st.info("No runs yet for this experiment.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Model Registry":
    st.markdown("# 📦 Model Registry")

    tab_all, tab_prod = st.tabs(["📋 All Models", "🚀 Production"])

    with tab_all:
        models = get_models()
        if not models:
            st.info("No models registered yet. Train a model and enable registration.")
        else:
            # Filter
            stages = ["All"] + list({m["stage"] for m in models})
            sel_stage = st.selectbox("Filter by Stage", stages)
            if sel_stage != "All":
                models = [m for m in models if m["stage"]==sel_stage]

            for m in models:
                metrics = m.get("metrics",{})
                score_key = next((k for k in ["accuracy","r2_score","f1_score","silhouette_score"] if k in metrics), None)
                score_str = f"{metrics[score_key]:.4f}" if score_key else "—"
                with st.expander(
                    f"**{m['name']}** v{m['version']} · {stage_badge(m['stage'])} · Score: {score_str}",
                    expanded=False
                ):
                    c1,c2,c3 = st.columns([2,1,1])
                    with c1:
                        st.markdown(f"**Description:** {m.get('description','—')}")
                        st.markdown(f"**Registered:** {m['created_at'][:16].replace('T',' ')}")
                        if metrics:
                            st.markdown("**Metrics:**")
                            display = {k:v for k,v in metrics.items()
                                       if k not in ["confusion_matrix","test_size","train_size"]}
                            st.json(display)
                    with c2:
                        st.markdown(f"**Stage:** {stage_badge(m['stage'])}", unsafe_allow_html=True)
                        new_stage = st.selectbox("Promote to",
                                                  ["staging","production","archived","champion"],
                                                  index=["staging","production","archived","champion"].index(m["stage"]),
                                                  key=f"stage_{m['id']}")
                        if st.button("✅ Promote", key=f"promote_{m['id']}"):
                            promote_model(m["id"], new_stage)
                            log_event("INFO","ModelRegistry",
                                      f"Promoted {m['name']} v{m['version']} to {new_stage}",
                                      {"model_id":m["id"]})
                            st.rerun()
                    with c3:
                        st.markdown(f"**Artifact:**")
                        st.code(m.get("artifact_path","—"), language="text")

    with tab_prod:
        prod_models = get_models("production")
        if not prod_models:
            st.info("No models in production yet.")
        else:
            st.markdown(f"**{len(prod_models)} model(s) currently in production**")
            for m in prod_models:
                metrics = m.get("metrics",{})
                col_a, col_b = st.columns([3,1])
                with col_a:
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:.8rem">
                        <div style="display:flex;justify-content:space-between;align-items:baseline">
                            <span style="font-weight:700;font-size:1rem">{m['name']} v{m['version']}</span>
                            {stage_badge('production')}
                        </div>
                        <div style="font-size:.82rem;color:#94A3B8;margin-top:.5rem">
                            Registered: {m['created_at'][:16].replace('T',' ')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    if st.button("⬇️ Rollback to Staging", key=f"rollback_{m['id']}"):
                        promote_model(m["id"], "staging")
                        log_event("WARNING","ModelRegistry",f"Rolled back {m['name']} to staging")
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Management":
    st.markdown("# 📊 Data Management")

    tab_ds, tab_explore = st.tabs(["📁 Datasets", "🔍 Explore"])

    with tab_ds:
        datasets = get_datasets()
        if not datasets:
            st.info("No datasets recorded. Load a dataset in the Train Model page.")
        else:
            for ds in datasets:
                with st.expander(f"**{ds['name']}** · {ds['version']} · {ds['num_samples']} rows", False):
                    c1,c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Task:** {ds['task_type']}")
                        st.markdown(f"**Samples:** {ds['num_samples']:,}")
                        st.markdown(f"**Features:** {ds['num_features']}")
                        st.markdown(f"**Target:** `{ds.get('target_column','—')}`")
                    with c2:
                        st.markdown(f"**Created:** {ds['created_at'][:16].replace('T',' ')}")
                        feat_names = ds.get("feature_names",[])
                        if feat_names:
                            st.markdown(f"**Features:** {', '.join(str(f) for f in feat_names[:8])}{'…' if len(feat_names)>8 else ''}")

    with tab_explore:
        st.markdown("### Explore a Built-in Dataset")
        ds_choice = st.selectbox("Dataset", ALL_DATASETS)
        if st.button("📂 Load & Explore"):
            with st.spinner("Loading…"):
                df, tc, tt = get_builtin_dataset(ds_choice)
                st.session_state["explore_df"] = df
                st.session_state["explore_tc"] = tc

        if "explore_df" in st.session_state:
            df = st.session_state["explore_df"]
            tc = st.session_state["explore_tc"]

            c1,c2,c3,c4 = st.columns(4)
            _metric_card(c1, df.shape[0], "Samples",  "#A78BFA")
            _metric_card(c2, df.shape[1]-1,"Features","#34D399")
            _metric_card(c3, int(df.isnull().sum().sum()), "Missing Values", "#F87171")
            _metric_card(c4, int(df[tc].nunique()) if tc in df.columns else 0, "Classes/Unique", "#FCD34D")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Sample Data")
            st.dataframe(df.head(20), use_container_width=True)

            st.markdown("#### Feature Distributions")
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()[:8]
            if numeric:
                fig = make_subplots(rows=2, cols=4,
                                    subplot_titles=numeric[:8])
                for i, col in enumerate(numeric[:8]):
                    r, c_ = divmod(i, 4)
                    fig.add_trace(go.Histogram(x=df[col], name=col,
                                               marker_color=PALETTE[i%len(PALETTE)],
                                               showlegend=False),
                                  row=r+1, col=c_+1)
                fig.update_layout(**plotly_theme(), height=400)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

            st.markdown("#### Correlation Matrix")
            corr = df[numeric].corr() if numeric else None
            if corr is not None:
                fig_corr = px.imshow(corr, color_continuous_scale="RdBu",
                                     zmin=-1, zmax=1, aspect="auto", text_auto=".2f")
                fig_corr.update_layout(**plotly_theme(), height=400)
                st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar":False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Drift Detection":
    st.markdown("# 🔍 Data Drift Detection")
    st.markdown('<div class="info-box">Uses Kolmogorov-Smirnov test + Population Stability Index (PSI) to detect feature distribution shifts.</div>',
                unsafe_allow_html=True)

    tab_run, tab_history = st.tabs(["▶️ Run Drift Check", "📋 Drift History"])

    with tab_run:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("### Configuration")
            ds_name   = st.selectbox("Reference Dataset", ALL_DATASETS)
            threshold = st.slider("Drift Threshold (KS stat)", 0.05, 0.5, 0.1, 0.01)
            drift_mag = st.slider("Simulate Drift Magnitude", 0.0, 2.0, 0.5, 0.1)

            models_list = get_models()
            if models_list:
                model_names = {f"{m['name']} v{m['version']} ({m['stage']})": m["id"]
                               for m in models_list}
                sel_model_str = st.selectbox("Associate with Model (optional)",
                                              ["None"] + list(model_names.keys()))
                sel_model_id  = model_names.get(sel_model_str)
            else:
                sel_model_id = None
                st.info("No models registered yet.")

            run_drift_btn = st.button("🔍 Run Drift Detection", use_container_width=True)

        with col_r:
            st.markdown("### Results")
            if run_drift_btn:
                with st.spinner("Loading data and computing drift…"):
                    df_ref, tc, tt  = get_builtin_dataset(ds_name)
                    df_cur          = generate_drifted_data(df_ref, drift_mag)
                    drift_result    = detect_drift(df_ref, df_cur, threshold)

                drift_score    = drift_result["drift_score"]
                drift_detected = drift_result["drift_detected"]
                feature_drift  = drift_result["feature_drift"]
                report         = drift_result["report"]

                # Save to DB
                ds_id = None
                dbs   = get_datasets()
                if dbs:
                    ds_id = dbs[0]["id"]
                save_drift_report(sel_model_id, ds_id, drift_score,
                                  drift_detected, feature_drift, report)
                log_event("WARNING" if drift_detected else "INFO",
                          "DriftDetection",
                          f"Drift {'DETECTED' if drift_detected else 'not detected'} — score={drift_score:.4f}",
                          {"threshold":threshold,"n_drifted":report.get("n_features_drifted")})

                # Show results
                alert_cls = "error-box" if drift_detected else "success-box"
                alert_msg = (f"⚠️ DRIFT DETECTED — Score: {drift_score:.4f} (threshold: {threshold})"
                             if drift_detected
                             else f"✅ No significant drift — Score: {drift_score:.4f}")
                st.markdown(f'<div class="{alert_cls}">{alert_msg}</div>', unsafe_allow_html=True)

                c1,c2,c3,c4 = st.columns(4)
                _metric_card(c1, f"{drift_score:.4f}","Overall Score","#F87171" if drift_detected else "#34D399")
                _metric_card(c2, report["n_features_checked"],"Features Checked","#93C5FD")
                _metric_card(c3, report["n_features_drifted"],"Features Drifted","#FCD34D")
                _metric_card(c4, f"{report['drift_pct']}%","Drift %","#F87171" if drift_detected else "#34D399")

                st.markdown("<br>", unsafe_allow_html=True)

                # Per-feature drift chart
                if feature_drift:
                    st.markdown("#### Per-Feature Drift (KS Statistic)")
                    df_fd = pd.DataFrame([
                        {"Feature": k,
                         "KS Stat": v["ks_statistic"],
                         "PSI":     v["psi"],
                         "Drifted": "Yes" if v["drifted"] else "No"}
                        for k,v in feature_drift.items()
                    ]).sort_values("KS Stat", ascending=False).head(15)

                    fig_drift = px.bar(df_fd, x="Feature", y="KS Stat",
                                       color="Drifted",
                                       color_discrete_map={"Yes":"#EF4444","No":"#10B981"})
                    fig_drift.add_hline(y=threshold, line_dash="dash",
                                        line_color="#F59E0B",
                                        annotation_text=f"Threshold ({threshold})")
                    fig_drift.update_layout(**plotly_theme(), xaxis_tickangle=-35)
                    st.plotly_chart(fig_drift, use_container_width=True, config={"displayModeBar":False})

                    # Distribution comparison for top drifted feature
                    top_feat = df_fd.iloc[0]["Feature"] if not df_fd.empty else None
                    if top_feat and top_feat in df_ref.columns:
                        st.markdown(f"#### Distribution Shift: `{top_feat}`")
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Histogram(x=df_ref[top_feat], name="Reference",
                                                         opacity=0.7, marker_color="#6C63FF"))
                        fig_dist.add_trace(go.Histogram(x=df_cur[top_feat], name="Current",
                                                         opacity=0.7, marker_color="#EF4444"))
                        fig_dist.update_layout(**plotly_theme(), barmode="overlay",
                                               legend=dict(orientation="h"))
                        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar":False})

    with tab_history:
        reports = get_drift_reports()
        if not reports:
            st.info("No drift reports yet.")
        else:
            df_rep = pd.DataFrame([{
                "ID":           r["id"],
                "Drift Score":  round(r.get("drift_score",0),4),
                "Drift?":       "⚠️ YES" if r.get("drift_detected") else "✅ NO",
                "Created":      r["created_at"][:16].replace("T"," "),
            } for r in reports])
            st.dataframe(df_rep, use_container_width=True, hide_index=True)

            # Trend chart
            fig_trend = px.line(df_rep, x="Created", y="Drift Score",
                                markers=True, color_discrete_sequence=["#6C63FF"])
            fig_trend.add_hline(y=0.1, line_dash="dash", line_color="#F59E0B",
                                annotation_text="Threshold")
            fig_trend.update_layout(**plotly_theme())
            st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar":False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AUTO RETRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Auto Retraining":
    st.markdown("# 🔄 Automated Retraining")

    tab_trigger, tab_jobs = st.tabs(["▶️ Trigger Job", "📋 Job History"])

    with tab_trigger:
        st.markdown('<div class="info-box">Schedule or manually trigger retraining jobs when drift is detected or performance degrades.</div>',
                    unsafe_allow_html=True)

        models_list = get_models()
        if not models_list:
            st.warning("No models registered. Train and register a model first.")
        else:
            col_cfg, col_run = st.columns(2)
            with col_cfg:
                model_names = {f"{m['name']} v{m['version']} ({m['stage']})": m for m in models_list}
                sel_str     = st.selectbox("Select Model to Retrain", list(model_names.keys()))
                sel_model   = model_names[sel_str]

                trigger = st.selectbox("Trigger Reason", [
                    "Manual trigger",
                    "Drift detected",
                    "Performance degradation",
                    "Scheduled weekly",
                    "Data update",
                ])
                st.markdown(f"**Model ID:** `{sel_model['id']}`")
                st.markdown(f"**Current Stage:** {stage_badge(sel_model['stage'])}", unsafe_allow_html=True)

            with col_run:
                # Find experiment for this model
                exps_list = get_experiments()
                if exps_list and st.button("🔄 Trigger Retraining Job", use_container_width=True):
                    job_id = create_retraining_job(sel_model["id"], trigger)
                    update_retraining_job(job_id, "running")
                    log_event("INFO","AutoRetraining",f"Retraining job started for model {sel_model['name']}",
                              {"job_id":job_id,"trigger":trigger})

                    with st.spinner("Retraining in progress…"):
                        try:
                            # Find original experiment
                            exp_id_for_retrain = sel_model.get("experiment_id") or exps_list[0]["id"]
                            exp_info = get_experiment(exp_id_for_retrain)
                            if exp_info:
                                task_type = exp_info["task_type"]
                                ds_name   = [d for d in ALL_DATASETS if task_type in d.lower()][0]
                                df_rt, tc_rt, _ = get_builtin_dataset(ds_name)

                                old_params = sel_model.get("params",{})
                                model_key  = next((k for k in (list(CLASSIFICATION_MODELS.keys())
                                                               +list(REGRESSION_MODELS.keys())
                                                               +list(CLUSTERING_MODELS.keys()))
                                                   if k.lower() in sel_model["name"].lower()),
                                                  "Random Forest")

                                new_run_id = create_run(exp_id_for_retrain,
                                                         f"retrain-{sel_model['name']}-{datetime.now().strftime('%H%M%S')}",
                                                         model_key, old_params)
                                result = train_model(exp_id_for_retrain, new_run_id,
                                                     df_rt, tc_rt, task_type,
                                                     model_key, old_params)
                                update_run(new_run_id, "completed",
                                           result["metrics"], result["duration"],
                                           result["artifact_path"])

                                # Register new version
                                new_mid = register_model(sel_model["name"],
                                                          exp_id_for_retrain, new_run_id,
                                                          result["metrics"], old_params,
                                                          result["artifact_path"],
                                                          f"Auto-retrained: {trigger}")
                                update_retraining_job(job_id, "completed", new_run_id)
                                log_event("SUCCESS","AutoRetraining",
                                          f"Retraining complete — new model v{new_mid}",
                                          {"job_id":job_id,"new_model_id":new_mid})
                                st.markdown(f'<div class="success-box">✅ Retraining complete! New model registered as v{new_mid}</div>',
                                            unsafe_allow_html=True)
                                metrics_d = result["metrics"]
                                st.json({k:v for k,v in metrics_d.items() if k not in ["confusion_matrix"]})
                        except Exception as e:
                            update_retraining_job(job_id, "failed")
                            log_event("ERROR","AutoRetraining",f"Retraining failed: {e}")
                            st.error(f"Retraining failed: {e}")

    with tab_jobs:
        jobs = get_retraining_jobs()
        if not jobs:
            st.info("No retraining jobs yet.")
        else:
            df_jobs = pd.DataFrame([{
                "Job ID":   j["id"],
                "Model ID": j.get("model_id","—"),
                "Trigger":  j.get("trigger","—"),
                "Status":   j.get("status","—"),
                "Started":  (j.get("started_at") or "—")[:16],
                "Completed":(j.get("completed_at") or "—")[:16],
            } for j in jobs])
            st.dataframe(df_jobs, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MONITORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Monitoring":
    st.markdown("# 📈 Monitoring Dashboard")

    runs = get_all_runs()
    models = get_models()
    drift_reports = get_drift_reports()

    if not runs:
        st.info("No runs to monitor yet. Train some models first.")
    else:
        df_runs = pd.DataFrame(runs)

        # ── Row 1: KPIs ──
        c1,c2,c3,c4,c5 = st.columns(5)
        succ_rate = (len([r for r in runs if r["status"]=="completed"]) / max(len(runs),1) * 100)
        avg_dur   = np.mean([r.get("duration_sec",0) or 0 for r in runs if r.get("duration_sec")])
        _metric_card(c1, len(runs),            "Total Runs",         "#A78BFA")
        _metric_card(c2, f"{succ_rate:.1f}%",  "Success Rate",       "#34D399")
        _metric_card(c3, f"{avg_dur:.1f}s",    "Avg Train Time",     "#93C5FD")
        _metric_card(c4, len(models),           "Registered Models",  "#FCD34D")
        _metric_card(c5, len(drift_reports),    "Drift Reports",      "#F87171")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 2: Run timeline + Model type distribution ──
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Run Timeline")
            df_timeline = df_runs.copy()
            df_timeline["date"] = pd.to_datetime(df_timeline["created_at"]).dt.date
            daily = df_timeline.groupby(["date","status"]).size().reset_index(name="count")
            fig_tl = px.bar(daily, x="date", y="count", color="status",
                            color_discrete_map={"completed":"#10B981","failed":"#EF4444",
                                                "running":"#3B82F6","pending":"#94A3B8"})
            fig_tl.update_layout(**plotly_theme())
            st.plotly_chart(fig_tl, use_container_width=True, config={"displayModeBar":False})

        with col_b:
            st.markdown("#### Model Type Distribution")
            model_dist = df_runs["model_type"].value_counts().reset_index()
            model_dist.columns = ["model_type","count"]
            fig_md = px.bar(model_dist, x="model_type", y="count",
                            color="model_type", color_discrete_sequence=PALETTE)
            fig_md.update_layout(**plotly_theme(), showlegend=False,
                                  xaxis_tickangle=-20)
            st.plotly_chart(fig_md, use_container_width=True, config={"displayModeBar":False})

        # ── Row 3: Metric trends ──
        st.markdown("#### Metric Trends Over Runs")
        df_metric = df_runs.copy()
        df_metric["score"] = df_metric["metrics"].apply(
            lambda m: next((v for k,v in m.items()
                            if k in ["accuracy","r2_score","f1_score","silhouette_score"]
                            and isinstance(v,float)), None)
            if isinstance(m,dict) else None
        )
        df_metric = df_metric[df_metric["score"].notna()].tail(30)
        if not df_metric.empty:
            fig_mt = px.scatter(df_metric, x="created_at", y="score",
                                color="model_type", size="score",
                                hover_data=["run_name","experiment_name"],
                                color_discrete_sequence=PALETTE)
            fig_mt.add_hline(y=df_metric["score"].mean(), line_dash="dash",
                             line_color="#F59E0B", annotation_text="Mean")
            fig_mt.update_layout(**plotly_theme())
            st.plotly_chart(fig_mt, use_container_width=True, config={"displayModeBar":False})

        # ── Row 4: Model stage distribution ──
        if models:
            st.markdown("#### Model Stage Distribution")
            stage_dist = pd.DataFrame(models)["stage"].value_counts().reset_index()
            stage_dist.columns = ["stage","count"]
            fig_stage = px.pie(stage_dist, names="stage", values="count",
                               color_discrete_sequence=PALETTE, hole=0.5)
            fig_stage.update_layout(**plotly_theme())
            col_p1, col_p2 = st.columns([1,2])
            with col_p1:
                st.plotly_chart(fig_stage, use_container_width=True, config={"displayModeBar":False})
            with col_p2:
                df_models_tbl = pd.DataFrame([{
                    "Name":    m["name"],
                    "Version": m["version"],
                    "Stage":   m["stage"],
                    "Score":   next((f"{v:.4f}" for k,v in m.get("metrics",{}).items()
                                     if k in ["accuracy","r2_score","f1_score"] and isinstance(v,float)), "—"),
                    "Registered": m["created_at"][:10],
                } for m in models[:15]])
                st.dataframe(df_models_tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PIPELINE LOGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Pipeline Logs":
    st.markdown("# 📋 Pipeline Logs")

    col_f1, col_f2, col_f3 = st.columns(3)
    log_limit = col_f1.slider("Max logs", 20, 500, 100)
    comp_filter = col_f2.selectbox("Component",
                                    ["All","Training","Experiments","ModelRegistry",
                                     "DriftDetection","AutoRetraining","DataManagement"])
    lvl_filter  = col_f3.selectbox("Level", ["All","INFO","WARNING","ERROR","SUCCESS"])

    logs = get_logs(log_limit)

    if comp_filter != "All":
        logs = [l for l in logs if l.get("component")==comp_filter]
    if lvl_filter != "All":
        logs = [l for l in logs if l.get("level")==lvl_filter]

    if not logs:
        st.info("No log entries found.")
    else:
        st.markdown(f"**{len(logs)} entries**")
        for log in logs:
            lvl  = log.get("level","INFO")
            ts   = log.get("created_at","")[:19].replace("T"," ")
            comp = log.get("component","")
            msg  = log.get("message","")
            cls  = f"log-{lvl}"
            color= {"INFO":"#93C5FD","WARNING":"#FCD34D","ERROR":"#F87171",
                    "SUCCESS":"#34D399"}.get(lvl,"#94A3B8")
            st.markdown(
                f'<div class="log-row">'
                f'<span style="color:#94A3B8;min-width:140px;display:inline-block">{ts}</span>'
                f'<span style="color:{color};min-width:80px;display:inline-block;font-weight:700">[{lvl}]</span>'
                f'<span style="color:#A78BFA;min-width:130px;display:inline-block">[{comp}]</span>'
                f'<span>{msg}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Download logs
        log_text = "\n".join(
            f"{l['created_at'][:19]} [{l['level']}] [{l['component']}] {l['message']}"
            for l in logs
        )
        st.download_button("⬇️ Download Logs",
                           data=log_text,
                           file_name=f"mlops_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                           mime="text/plain")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    MLOps Platform v1.0 &nbsp;·&nbsp;
    Streamlit · scikit-learn · SQLite · Plotly &nbsp;·&nbsp;
    Built by Shaunak Mirajgaonkar
</div>
""", unsafe_allow_html=True)
