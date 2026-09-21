"""Central session-state keys for the Streamlit app."""

from __future__ import annotations


class SS:
    data_profile_name = "data_profile_name"
    data_page_size = "data_page_size"
    data_page_number = "data_page_number"
    host_atlas_color_col = "host_atlas_color_col"
    model_task = "model_task"
    abundance_upload = "abundance_upload"
    metadata_upload = "metadata_upload"
    metadata_target_col = "metadata_target_col"
    model_apply_tss = "model_apply_tss"
    model_threshold = "model_threshold"
    model_k = "model_k"
    model_metric = "model_metric"
    model_weight_his = "model_weight_his"
    model_weight_pts = "model_weight_pts"
    model_weight_fla = "model_weight_fla"
    model_run_signature = "model_run_signature"
    model_prediction_df = "model_prediction_df"
    model_metrics = "model_metrics"
    model_neighbors_df = "model_neighbors_df"
    neighbor_query_sample = "neighbor_query_sample"
    model_applied_tss = "model_applied_tss"
    model_true_labels = "model_true_labels"
    model_mode = "model_mode"
    model_pos_label = "model_pos_label"
    model_class_order = "model_class_order"
    model_confusion = "model_confusion"
    model_roc = "model_roc"
    last_model_task = "_last_model_task"
    last_upload_signature = "_last_upload_signature"


MODEL_RESULT_KEYS = (
    SS.model_run_signature,
    SS.model_prediction_df,
    SS.model_metrics,
    SS.model_neighbors_df,
    SS.neighbor_query_sample,
    SS.model_applied_tss,
    SS.model_true_labels,
    SS.model_mode,
    SS.model_pos_label,
    SS.model_class_order,
    SS.model_confusion,
    SS.model_roc,
)
