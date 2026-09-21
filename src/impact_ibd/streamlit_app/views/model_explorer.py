"""Model explorer page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from impact_ibd.streamlit_app.components import charts
from impact_ibd.streamlit_app.components.layout import (
    card,
    card_title,
    clear_model_results,
    page_header,
    render_sidebar,
    section_gap,
    show_errors,
    show_warnings,
)
from impact_ibd.streamlit_app.components.neighbor_graph import build_neighbor_figure, plotly_config
from impact_ibd.streamlit_app.constants import (
    DISTANCE_METRICS,
    MAX_K,
    PLOTLY_CONFIG,
    TASK_LABELS,
    TSS_HELP,
)
from impact_ibd.streamlit_app.services import model_service
from impact_ibd.streamlit_app.services.validation import (
    ValidationError,
    expected_model_features,
    parse_csv_bytes,
    validate_abundance,
    validate_metadata,
)
from impact_ibd.streamlit_app.state import SS


def _sync_task_defaults() -> str:
    task = st.session_state.get(SS.model_task, "ibd_control")
    if st.session_state.get(SS.last_model_task) != task:
        clear_model_results()
        defaults = model_service.model_defaults(task)
        st.session_state[SS.model_threshold] = float(defaults["threshold"])
        st.session_state[SS.model_k] = int(defaults["k"])
        st.session_state[SS.model_metric] = defaults["metric"]
        st.session_state[SS.model_weight_his] = float(defaults["weights"]["His"])
        st.session_state[SS.model_weight_pts] = float(defaults["weights"]["PTS"])
        st.session_state[SS.model_weight_fla] = float(defaults["weights"]["Fla"])
        st.session_state[SS.last_model_task] = task
    return task


def render() -> None:
    render_sidebar()
    page_header(
        "Model explorer",
        "Upload a sample abundance matrix, choose whether to apply TSS normalization, then run prediction or evaluation.",
        eyebrow="Predictive models",
    )

    with card("card_model_choice"):
        card_title("1. Choose a model")
        st.selectbox(
            "Select model",
            options=list(TASK_LABELS.keys()),
            format_func=lambda key: TASK_LABELS[key],
            key=SS.model_task,
        )
    task = _sync_task_defaults()
    defaults = model_service.model_defaults(task)
    prototype = model_service.get_cached_model(task)

    with card("card_model_summary"):
        card_title("Model summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Target column", str(defaults["target_col"]))
        c2.metric("Positive class", str(defaults["pos_label"]))
        c3.metric("Classes", " / ".join(map(str, defaults["classes"])))
        c4, c5, c6 = st.columns(3)
        c4.metric("Default threshold", f"{defaults['threshold']:.4f}")
        c5.metric("Default k", defaults["k"])
        c6.metric("Default metric", defaults["metric"])
        st.caption(
            "Module weights His / PTS / Fla = "
            f"{defaults['weights']['His']:.4f} / {defaults['weights']['PTS']:.4f} / {defaults['weights']['Fla']:.4f}"
        )

    section_gap()
    with card("card_model_upload"):
        card_title("2. Upload and validate data")
        abundance_file = st.file_uploader(
            "Upload abundance CSV (required)",
            type=["csv"],
            key=SS.abundance_upload,
            help="The first column is the unique sample ID. One sample per row, one feature per column.",
        )
        metadata_file = st.file_uploader(
            "Upload metadata CSV (optional, for evaluation)",
            type=["csv"],
            key=SS.metadata_upload,
        )

    X_raw = None
    meta_df = None
    abd_summary = None
    abundance_bytes = b""
    metadata_bytes = b""

    if abundance_file is not None:
        abundance_bytes = abundance_file.getvalue()
        try:
            parsed = parse_csv_bytes(abundance_bytes)
            abd_summary = validate_abundance(parsed, expected_model_features(prototype))
            X_raw = abd_summary.df
        except ValidationError as exc:
            st.error(str(exc))

    if metadata_file is not None:
        metadata_bytes = metadata_file.getvalue()
        try:
            meta_df = parse_csv_bytes(metadata_bytes)
        except ValidationError as exc:
            st.error(str(exc))
            meta_df = None

    import hashlib

    upload_sig = hashlib.sha256(task.encode() + b"|" + abundance_bytes + b"|" + metadata_bytes).hexdigest()
    if st.session_state.get(SS.last_upload_signature) not in (None, upload_sig):
        clear_model_results()
    st.session_state[SS.last_upload_signature] = upload_sig

    if abd_summary is not None:
        with card("card_model_validation"):
            card_title("Input validation")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Input samples", abd_summary.n_samples)
            m2.metric("Input features", abd_summary.n_features)
            m3.metric("Overlapping features", abd_summary.n_overlap)
            m4.metric("Missing features (filled with 0)", abd_summary.n_missing)
            m5.metric("Extra features (dropped)", abd_summary.n_extra)
            st.caption("Validation compares feature name sets only. The matrix is not trimmed or reordered before TSS normalization.")
            show_errors(abd_summary.errors)
            if not st.session_state.get(SS.model_apply_tss, False):
                show_warnings(abd_summary.warnings)
            with st.expander("Row sums by sample", expanded=False):
                sums = abd_summary.row_sums.rename("row_sum").reset_index()
                sums.columns = ["Sample ID", "row_sum"]
                st.dataframe(sums, width="stretch", hide_index=True, height=220)

    target_options = list(meta_df.columns) if meta_df is not None else []
    if meta_df is not None and X_raw is not None:
        default_target = defaults["target_col"]
        if default_target in target_options:
            st.session_state.setdefault(SS.metadata_target_col, default_target)
            if st.session_state.get(SS.metadata_target_col) not in target_options:
                st.session_state[SS.metadata_target_col] = default_target

    section_gap()
    with card("card_model_run"):
        card_title("3. Configure and run")
        with st.form("model_run_form"):
            st.checkbox(
                "Apply TSS normalization to the uploaded data",
                key=SS.model_apply_tss,
                help=TSS_HELP,
            )
            if target_options:
                st.selectbox(
                    "Metadata target column",
                    options=target_options,
                    key=SS.metadata_target_col,
                )
            else:
                st.caption("Without metadata, the model runs prediction and does not compute evaluation metrics.")

            with st.expander("Advanced parameters", expanded=False):
                st.caption("Defaults are the saved model values. These parameters apply only to a copy used for this run.")
                st.number_input(
                    "decision threshold",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.0001,
                    format="%.4f",
                    key=SS.model_threshold,
                )
                st.number_input(
                    "k",
                    min_value=1,
                    max_value=MAX_K,
                    step=1,
                    key=SS.model_k,
                )
                st.selectbox(
                    "distance metric",
                    options=list(DISTANCE_METRICS),
                    key=SS.model_metric,
                )
                w1, w2, w3 = st.columns(3)
                with w1:
                    st.number_input("His weight", min_value=0.0, step=0.05, key=SS.model_weight_his)
                with w2:
                    st.number_input("PTS weight", min_value=0.0, step=0.05, key=SS.model_weight_pts)
                with w3:
                    st.number_input("Fla weight", min_value=0.0, step=0.05, key=SS.model_weight_fla)

            submitted = st.form_submit_button("Run prediction / evaluation", type="primary")

    if submitted:
        _execute_run(
            task=task,
            abundance_file=abundance_file,
            metadata_file=metadata_file,
            abundance_bytes=abundance_bytes,
            metadata_bytes=metadata_bytes,
            X_raw=X_raw,
            meta_df=meta_df,
            abd_summary=abd_summary,
        )

    _render_results(task)


def _execute_run(
    *,
    task,
    abundance_file,
    metadata_file,
    abundance_bytes,
    metadata_bytes,
    X_raw,
    meta_df,
    abd_summary,
) -> None:
    if abundance_file is None or X_raw is None or abd_summary is None:
        st.error("Upload a validated abundance CSV first.")
        return
    if abd_summary.errors:
        show_errors(abd_summary.errors)
        return

    apply_tss = bool(st.session_state.get(SS.model_apply_tss, False))
    if apply_tss and abd_summary.zero_sum_ids:
        st.error(
            "TSS normalization cannot be applied because these samples have a row sum of 0 across all uploaded features: "
            + ", ".join(abd_summary.zero_sum_ids)
        )
        return

    target_col = st.session_state.get(SS.metadata_target_col) if meta_df is not None else None
    if meta_df is not None:
        prototype = model_service.get_cached_model(task)
        meta_summary = validate_metadata(
            meta_df,
            X_raw.index,
            allowed_labels=model_service.model_defaults(task)["classes"],
            target_col=target_col or prototype.target_col_train_,
        )
        show_warnings(meta_summary.warnings)
        if meta_summary.errors:
            show_errors(meta_summary.errors)
            return
        meta_df = meta_summary.df

    signature = model_service.run_signature(
        abundance_bytes,
        metadata_bytes if metadata_file is not None else None,
        task,
        apply_tss,
        int(st.session_state[SS.model_k]),
        str(st.session_state[SS.model_metric]),
        (
            float(st.session_state[SS.model_weight_his]),
            float(st.session_state[SS.model_weight_pts]),
            float(st.session_state[SS.model_weight_fla]),
        ),
        float(st.session_state[SS.model_threshold]),
    )
    if signature != st.session_state.get(SS.model_run_signature):
        # Keep previous success until the new run completes.
        pass

    try:
        with st.spinner("Running the model…"):
            result = model_service.run_model(
                task=task,
                X_raw=X_raw,
                meta=meta_df,
                target_col=target_col,
                apply_tss=apply_tss,
                k=int(st.session_state[SS.model_k]),
                metric=str(st.session_state[SS.model_metric]),
                weight_his=float(st.session_state[SS.model_weight_his]),
                weight_pts=float(st.session_state[SS.model_weight_pts]),
                weight_fla=float(st.session_state[SS.model_weight_fla]),
                threshold=float(st.session_state[SS.model_threshold]),
            )
    except ValidationError as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error("The model run failed. Check the inputs and try again.")
        return

    st.session_state[SS.model_run_signature] = signature
    st.session_state[SS.model_prediction_df] = result.prediction_df
    st.session_state[SS.model_metrics] = result.metrics
    st.session_state[SS.model_neighbors_df] = result.neighbors_df
    st.session_state[SS.model_applied_tss] = result.applied_tss
    st.session_state[SS.model_true_labels] = result.true_labels
    st.session_state[SS.model_mode] = result.mode
    st.session_state[SS.model_pos_label] = result.pos_label
    st.session_state[SS.model_class_order] = result.class_order
    st.session_state[SS.model_confusion] = result.confusion
    st.session_state[SS.model_roc] = result.roc
    first_sample = result.prediction_df["Sample_ID"].iloc[0]
    st.session_state[SS.neighbor_query_sample] = first_sample
    st.success("Run complete.")


def _render_results(task: str) -> None:
    pred = st.session_state.get(SS.model_prediction_df)
    if pred is None:
        st.info("The model has not been run yet. Upload an abundance CSV, then click the button.")
        return

    applied = bool(st.session_state.get(SS.model_applied_tss, False))
    if applied:
        st.success("TSS normalization was applied to all uploaded features. Prediction, evaluation, and nearest-neighbor lookup use the same matrix.")
    else:
        st.info("This run used the uploaded values without TSS normalization.")

    display = pred.copy()
    rename = {"Sample_ID": "Sample ID", "Predicted_Class": "Predicted Label"}
    display = display.rename(columns=rename)
    mode = st.session_state.get(SS.model_mode)
    if mode == "evaluate":
        true_labels = st.session_state.get(SS.model_true_labels)
        if true_labels is not None:
            display.insert(
                list(display.columns).index("Predicted Label") + 1,
                "True Label",
                display["Sample ID"].map(true_labels),
            )
    with card("card_prediction_results"):
        card_title("4. Prediction results")
        st.dataframe(display, width="stretch", hide_index=True)

        counts = pred["Predicted_Class"].value_counts(dropna=False).rename_axis("Predicted Label").reset_index(name="Samples")
        st.dataframe(counts, width="stretch", hide_index=True)

    metrics = st.session_state.get(SS.model_metrics) or {}
    if mode == "evaluate":
        with card("card_evaluation_results"):
            card_title("Evaluation")
            cols = st.columns(6)
            keys = [
                ("accuracy", "Accuracy"),
                ("f1_score", "F1-score"),
                ("auc_roc", "AUC-ROC"),
                ("mcc", "MCC"),
                ("sensitivity", "Sensitivity"),
                ("specificity", "Specificity"),
            ]
            for col, (key, label) in zip(cols, keys):
                value = metrics.get(key)
                col.metric(label, f"{value:.4f}" if isinstance(value, float) else "—")

        confusion = st.session_state.get(SS.model_confusion)
        labels = st.session_state.get(SS.model_class_order) or []
        if confusion is not None:
            with card("card_evaluation_confusion"):
                card_title("Confusion matrix")
                with st.container(key="eval_confusion_plot"):
                    st.plotly_chart(
                        charts.confusion_heatmap(confusion, labels),
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                        key=f"cm_{st.session_state.get(SS.model_run_signature)}",
                    )

        roc = st.session_state.get(SS.model_roc) or {}
        with card("card_evaluation_roc"):
            card_title("ROC curve")
            if roc.get("single_class"):
                st.warning("True labels contain only one class, so a ROC curve cannot be drawn.")
            elif "fpr" in roc:
                with st.container(key="eval_roc_plot"):
                    st.plotly_chart(
                        charts.roc_curve_figure(roc["fpr"], roc["tpr"], roc.get("auc")),
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                        key=f"roc_{st.session_state.get(SS.model_run_signature)}",
                    )

    neighbors = st.session_state[SS.model_neighbors_df]
    samples = list(pred["Sample_ID"])
    if st.session_state.get(SS.neighbor_query_sample) not in samples:
        st.session_state[SS.neighbor_query_sample] = samples[0]
    with card("card_neighbors"):
        card_title("5. Nearest neighbors")
        st.selectbox("Select test sample", options=samples, key=SS.neighbor_query_sample)
        query = st.session_state[SS.neighbor_query_sample]
        row = pred.loc[pred["Sample_ID"] == query].iloc[0]
        true_labels = st.session_state.get(SS.model_true_labels)
        true_label = None
        if true_labels is not None and query in true_labels.index:
            true_label = true_labels.loc[query]
            st.caption(f"Predicted label = {row['Predicted_Class']}; True label = {true_label}")
        fig = build_neighbor_figure(
            neighbors,
            query,
            task=task,
            predicted_label=row["Predicted_Class"],
            confidence=float(row["Confidence"]),
            true_label=true_label,
            k=int(st.session_state.get(SS.model_k) or neighbors.loc[neighbors["Query_Sample"] == query].shape[0]),
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config=plotly_config(),
            key=f"neighbors_{st.session_state.get(SS.model_run_signature)}_{query}",
        )
        card_title("Nearest neighbor details")
        st.dataframe(
            neighbors.loc[neighbors["Query_Sample"] == query],
            width="stretch",
            hide_index=True,
        )
