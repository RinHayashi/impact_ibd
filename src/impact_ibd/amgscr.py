import joblib
import importlib.resources
from typing import Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix, matthews_corrcoef
)

# Global Pseudocount
EPS = 1e-9

def compute_distance(Zq, Zr, metric="cosine"):
    """
    Compute distances between scaled abundance matrices Zq and Zr.

    This module-level function is used by WeightedDistanceKNN.
    """
    if metric == "cosine":
        return cosine_distances(Zq, Zr)
    elif metric == "euclidean":
        return euclidean_distances(Zq, Zr)
    elif metric == "braycurtis":
        return cdist(Zq, Zr, metric="braycurtis")
    elif metric == "correlation":
        # 1 - Pearson
        Zq_c = Zq - Zq.mean(axis=1, keepdims=True)
        Zr_c = Zr - Zr.mean(axis=1, keepdims=True)
        num = Zq_c @ Zr_c.T
        den = (np.linalg.norm(Zq_c, axis=1, keepdims=True) *
               np.linalg.norm(Zr_c, axis=1))
        corr = num / (den + 1e-9)
        return 1 - corr
    else:
        raise ValueError(metric)


class WeightedDistanceKNN:
    def __init__(self, 
                 genes_aa, genes_carb, genes_flag, # Gene lists for each AMG module
                 metric='cosine', # Distance metric
                 w_aa=0.5, w_carb=0.3, w_flag=0.2, # AMG module weights
                 k=15, # Number of matched neighbors
                 use_log1p=True, method='log1p+z-score',
                 threshold=0.5): # Initial classification threshold
        self.genes_aa = list(genes_aa)
        self.genes_carb = list(genes_carb)
        self.genes_flag = list(genes_flag)
        self.metric = metric
        self.w = np.array([w_aa, w_carb, w_flag], dtype=float)
        self.k = k
        self.use_log1p = use_log1p
        self.method = method
        
        # Threshold attributes; underscored values are updated and saved during training.
        self.threshold = threshold
        self.threshold_ = threshold
        self.pos_label_ = None
        self.target_col_train_ = None
        
        self.scaler_aa = StandardScaler()
        self.scaler_carb = StandardScaler()
        self.scaler_flag = StandardScaler()
        
    def get_params(self, deep=True):
        return dict(
            genes_aa=self.genes_aa,
            genes_carb=self.genes_carb,
            genes_flag=self.genes_flag,
            metric=self.metric,
            w_aa=float(self.w[0]),
            w_carb=float(self.w[1]),
            w_flag=float(self.w[2]),
            k=self.k,
            use_log1p=self.use_log1p,
            method=self.method,
            threshold=self.threshold
        )
    
    def _prep_block(self, X, genes, scaler, fit=False):
        """Prepare a feature block for internal preprocessing."""
        Xb = X.reindex(columns=genes, fill_value=0.0)
        
        if self.method == 'CLR': 
            X_clr = np.log(Xb + EPS) - np.log(Xb + EPS).mean(axis=0)
            return X_clr.values
        else:
            if self.use_log1p:
                Xb = np.log1p(Xb)
            if fit:
                Z = scaler.fit_transform(Xb) 
            else:
                Z = scaler.transform(Xb) 
            return Z

    def fit(self, X_ref: pd.DataFrame, meta_ref: pd.DataFrame, 
            optimize_threshold=False, target_col=None, pos_label=None, n_splits=5):
        """
        Fit scalers on X_ref and store the reference dataset.
        
        Args:
            X_ref: Training abundance matrix.
            meta_ref: Training clinical data or metadata.
            optimize_threshold: Whether to optimize the Youden index threshold
                for binary classification.
            target_col: Target column name; required for threshold optimization.
            pos_label: Positive class label. If None during optimization, the
                second sorted label is used.
            n_splits: Number of cross-validation folds.
        """
        self.X_ref = X_ref.copy()
        self.ref_ids_ = self.X_ref.index.to_numpy()
        self.meta_ref = meta_ref.copy()
        self.target_col_train_ = target_col

        self.Z_aa = self._prep_block(self.X_ref, self.genes_aa, self.scaler_aa, fit=True)
        self.Z_carb = self._prep_block(self.X_ref, self.genes_carb, self.scaler_carb, fit=True)
        self.Z_flag = self._prep_block(self.X_ref, self.genes_flag, self.scaler_flag, fit=True)
        
        # Optionally optimize the Youden index threshold.
        if optimize_threshold:
            if target_col is None:
                raise ValueError("Optimizing threshold requires specifying 'target_col'.")
            self._optimize_threshold_cv(X_ref, meta_ref, target_col, pos_label, n_splits)
            
        return self 

    def _optimize_threshold_cv(self, X_ref, meta_ref, target_col, pos_label, n_splits):
        """Find the optimal Youden threshold using stratified cross-validation."""
        y_ref = meta_ref[target_col].to_numpy()
        unique_labels = np.unique(y_ref)
        
        if len(unique_labels) != 2:
            raise ValueError(f"Youden index optimization is only applicable for binary classification. Found labels: {unique_labels}")
        
        if pos_label is None:
            # Use the second sorted label as the positive class by default.
            pos_label = unique_labels[1]
        
        self.pos_label_ = pos_label
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        all_y_true = []
        all_y_proba = []
        
        # Run cross-validation.
        for train_idx, val_idx in skf.split(X_ref, y_ref):
            X_train, X_val = X_ref.iloc[train_idx], X_ref.iloc[val_idx]
            meta_train, meta_val = meta_ref.iloc[train_idx], meta_ref.iloc[val_idx]
            
            # Use a temporary model to preserve the current fitted state.
            fold_model = WeightedDistanceKNN(
                genes_aa=self.genes_aa, genes_carb=self.genes_carb, genes_flag=self.genes_flag,
                metric=self.metric, w_aa=self.w[0], w_carb=self.w[1], w_flag=self.w[2],
                k=self.k, use_log1p=self.use_log1p, method=self.method, threshold=self.threshold
            )
            fold_model.fit(X_train, meta_train, optimize_threshold=False)
            
            # Predict validation-set probabilities.
            _, _, probas = fold_model.predict_meta(X_val, target_col=target_col, task="clf")
            
            # Extract positive-class probabilities.
            fold_probas = [p.get(pos_label, 0.0) for p in probas]
            
            all_y_proba.extend(fold_probas)
            all_y_true.extend(meta_val[target_col].values)
            
        all_y_true = np.array(all_y_true)
        all_y_proba = np.array(all_y_proba)
        
        # Compute the ROC curve and Youden index.
        fpr, tpr, thresholds = roc_curve(all_y_true, all_y_proba, pos_label=pos_label)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        
        # Clamp boundary threshold placeholders from scikit-learn to [0, 1].
        best_threshold = float(thresholds[best_idx])
        self.threshold_ = max(0.0, min(1.0, best_threshold))

    def _distance_to_ref(self, X_query: pd.DataFrame):
        """Compute the weighted distance matrix."""
        Zq_aa = self._prep_block(X_query, self.genes_aa, self.scaler_aa, fit=False) 
        Zq_carb = self._prep_block(X_query, self.genes_carb, self.scaler_carb, fit=False)
        Zq_flag = self._prep_block(X_query, self.genes_flag, self.scaler_flag, fit=False)

        d_aa = compute_distance(Zq_aa, self.Z_aa, metric=self.metric)
        d_carb = compute_distance(Zq_carb, self.Z_carb, metric=self.metric)
        d_flag = compute_distance(Zq_flag, self.Z_flag, metric=self.metric)

        D = self.w[0]*d_aa + self.w[1]*d_carb + self.w[2]*d_flag
        return D  

    def search(self, X_query: pd.DataFrame, topk=None, tau=None):
        """Run a weighted KNN search against reference samples for each query."""
        if topk is None:
            topk = self.k 

        D = self._distance_to_ref(X_query) 
        idx = np.argsort(D, axis=1)[:, :topk] 
        dist = np.take_along_axis(D, idx, axis=1)
        
        if tau is None:
            tau = np.median(D) + EPS 
            
        sim = np.exp(-dist / tau)
        hit_ids = self.ref_ids_[idx]

        return hit_ids, dist, sim

    def get_nearest_neighbors(
        self,
        X_query: pd.DataFrame,
        topk=None,
        target_col: Optional[str] = None,
        tau: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Return labeled reference neighbors for each query sample.

        This is a presentation-oriented wrapper around :meth:`search`; it does
        not alter the fitted model or the prediction path. Distance and
        similarity therefore retain exactly the same definitions as
        :meth:`search`, including its existing default ``tau`` behavior.

        Args:
            X_query: Query abundance matrix with samples as rows and features
                as columns. Its index is used as the query sample name.
            topk: Number of reference neighbors to return. Defaults to the
                model's fitted ``k`` value.
            target_col: Reference metadata column shown as the neighbor label.
                Defaults to the target column saved during model fitting.
            tau: Optional similarity scale passed directly to :meth:`search`.
                When omitted, the existing search default is used.

        Returns:
            A long-format DataFrame with query sample, rank, reference sample,
            reference label, distance, and similarity columns.
        """
        if not isinstance(X_query, pd.DataFrame):
            raise TypeError("X_query must be a pandas DataFrame.")
        if X_query.empty:
            raise ValueError("X_query must contain at least one sample.")

        effective_topk = self.k if topk is None else topk
        if (
            not isinstance(effective_topk, (int, np.integer))
            or isinstance(effective_topk, (bool, np.bool_))
            or effective_topk <= 0
        ):
            raise ValueError("topk must be a positive integer.")

        if tau is not None:
            if not np.isscalar(tau) or not np.isfinite(tau) or tau <= 0:
                raise ValueError("tau must be a positive finite scalar.")

        effective_target_col = (
            target_col
            if target_col is not None
            else getattr(self, "target_col_train_", None)
        )
        if effective_target_col is None:
            raise ValueError(
                "No target column is stored in the model; specify target_col."
            )
        if not hasattr(self, "meta_ref"):
            raise AttributeError("The model does not contain reference metadata.")
        if effective_target_col not in self.meta_ref.columns:
            raise KeyError(
                f"Reference metadata does not contain column "
                f"'{effective_target_col}'."
            )

        hit_ids, dist, sim = self.search(
            X_query,
            topk=int(effective_topk),
            tau=tau,
        )

        records = []
        for query_pos, query_id in enumerate(X_query.index):
            neighbor_ids = hit_ids[query_pos]
            neighbor_labels = self.meta_ref.loc[
                neighbor_ids, effective_target_col
            ].to_numpy()

            for rank, (ref_id, label, distance, similarity) in enumerate(
                zip(
                    neighbor_ids,
                    neighbor_labels,
                    dist[query_pos],
                    sim[query_pos],
                ),
                start=1,
            ):
                records.append(
                    {
                        "Query_Sample": query_id,
                        "Rank": rank,
                        "Reference_Sample": ref_id,
                        "Reference_Label": label,
                        "Distance": float(distance),
                        "Similarity": float(similarity),
                    }
                )

        return pd.DataFrame.from_records(
            records,
            columns=[
                "Query_Sample",
                "Rank",
                "Reference_Sample",
                "Reference_Label",
                "Distance",
                "Similarity",
            ],
        )

    def predict_meta(self, X_query: pd.DataFrame, target_col: str, task="auto"):
        """Predict each query by weighting the neighbors returned by search()."""
        hit_ids, dist, sim = self.search(X_query, topk=self.k)
        y_ref = self.meta_ref[target_col]
        
        preds = []
        confs = []
        probas = []

        if task == "auto":
            is_numeric = pd.api.types.is_numeric_dtype(y_ref)
            task_eff = "reg" if is_numeric else "clf"
        else:
            task_eff = task

        for q in range(hit_ids.shape[0]): 
            neigh_ids = hit_ids[q]
            weights = sim[q] 
            y = y_ref.loc[neigh_ids] 
            
            if task_eff == "clf":
                score = {} 
                for label, w in zip(y.values, weights): 
                    score[label] = score.get(label, 0.0) + float(w) 
                
                total_w = sum(score.values()) + EPS
                proba = {k: v / total_w for k, v in score.items()}
                
                # Apply the saved threshold for binary classification.
                if self.pos_label_ is not None and len(proba) <= 2:
                    pos_prob = proba.get(self.pos_label_, 0.0)
                    if pos_prob >= self.threshold_:
                        pred = self.pos_label_
                    else:
                        # Select the other label as the negative class.
                        other_labels = [k for k in score.keys() if k != self.pos_label_]
                        pred = other_labels[0] if other_labels else self.pos_label_
                else:
                    # Otherwise, select the class with the highest score.
                    pred = max(score, key=score.get)
                
                # Use the predicted class probability as confidence.
                conf = proba.get(pred, 0.0)
                
                preds.append(pred) 
                confs.append(conf) 
                probas.append(proba) 

            else:  # reg
                yv = y.astype(float).values 
                pred = float(np.sum(weights * yv) / (np.sum(weights) + EPS))
                conf = float(1.0 / (np.std(yv) + 1e-6))
                preds.append(pred)
                confs.append(conf)
        
        if task_eff == "clf":
            return np.array(preds), np.array(confs), probas
        else:
            return np.array(preds), np.array(confs)

    def evaluate(self, X_test: pd.DataFrame, target_col: str = None, meta_test: pd.DataFrame = None, 
                 pos_label = None, threshold = None, save_csv: str = None, save_pdf: str = None) -> dict:
        """
        Run predictions and optionally evaluate their performance.
        
        Args:
            X_test (pd.DataFrame): Test or external validation abundance matrix.
            target_col (str, optional): Target column in meta_test. Defaults to
                self.target_col_train_ when meta_test is provided.
            meta_test (pd.DataFrame, optional): Ground-truth test metadata.
            pos_label: Positive class label. Defaults to self.pos_label_.
            threshold (float, optional): Classification threshold. Defaults to
                self.threshold_.
            save_csv (str, optional): Path for detailed prediction results.
            save_pdf (str, optional): Path for evaluation plots.
            
        Returns:
            dict: Core evaluation metrics or run status.
        """
        
        # Configure Matplotlib.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['font.sans-serif'] = ['Arial']
        matplotlib.rcParams['axes.unicode_minus'] = False

        # ----- 0. Validate the training target column. -----
        if not hasattr(self, 'target_col_train_') or self.target_col_train_ is None:
            raise AttributeError("模型内部未检测到训练集目标列名 'target_col_train_'。请确保在 fit() 阶段将目标列名保存至 self.target_col_train_ 。")

        # ----- 0. Align features with the model's AMG gene lists. -----
        # Combine all modules, preserving order and removing duplicates.
        expected_features = list(dict.fromkeys(self.genes_aa + self.genes_carb + self.genes_flag))
        
        original_cols = set(X_test.columns)
        expected_cols = set(expected_features)
        
        intersect_feats = original_cols.intersection(expected_cols)
        missing_feats = expected_cols - original_cols
        extra_feats = original_cols - expected_cols

        # Print the feature alignment report.
        print("=" * 60)
        print("Feature Dimension Alignment Report:")
        print(f"  - Expected features (AA+Carb+Flag): {len(expected_cols)}")
        print(f"  - Provided features (in input data): {len(original_cols)}")
        print(f"  - Overlapping features (retained):   {len(intersect_feats)}")
        print(f"  - Missing features (filled with 0):  {len(missing_feats)}")
        print(f"  - Extra features (discarded):        {len(extra_feats)}")
        print("=" * 60)

        # Fill missing features with 0.0 and discard extra features.
        X_test_aligned = X_test.reindex(columns=expected_features, fill_value=0.0)

        # 2. Resolve the positive label and decision threshold.
        eff_pos_label = pos_label if pos_label is not None else self.pos_label_
        eff_threshold = threshold if threshold is not None else self.threshold_

        # 3. Run inference using the stored training target column.
        _, confs, probas = self.predict_meta(X_test_aligned, target_col=self.target_col_train_, task="clf")
        
        # Collect all class labels observed in this task.
        all_labels = set()
        for p in probas:
            all_labels.update(p.keys())
        all_labels = sorted(list(all_labels))
        
        if eff_pos_label is None and len(all_labels) == 2:
            eff_pos_label = all_labels[1]
        
        # 4. Apply the selected decision threshold.
        if eff_pos_label in all_labels and len(all_labels) == 2:
            y_probs = np.array([p.get(eff_pos_label, 0.0) for p in probas])
            eff_neg_label = [l for l in all_labels if l != eff_pos_label][0]
            preds_opt = np.where(y_probs >= eff_threshold, eff_pos_label, eff_neg_label)
        else:
            preds_opt, _, _ = self.predict_meta(X_test_aligned, target_col=self.target_col_train_, task="clf")
            y_probs = np.array([p.get(preds_opt[i], 0.0) for i, p in enumerate(probas)])

        # 5. Build and optionally save predictions for all samples.
        res_df = pd.DataFrame({
            "Sample_ID": X_test.index,
            "Predicted_Class": preds_opt,
            "Confidence": confs
        })
        for label in all_labels:
            res_df[f"Probability_{label}"] = [p.get(label, 0.0) for p in probas]
            
        if save_csv:
            res_df.to_csv(save_csv, index=False)
            print(f"[Success] Prediction results successfully exported to: {save_csv}")

        metrics = {"used_threshold": eff_threshold, "pos_label": eff_pos_label}
        
        # 6. Evaluate performance when ground truth is provided.
        if meta_test is not None:
            # Resolve the target column used for the test set.
            eff_target_col_test = target_col if target_col is not None else self.target_col_train_
            if eff_target_col_test not in meta_test.columns:
                raise KeyError(f"测试集元数据 'meta_test' 中未找到指定的列: '{eff_target_col_test}'")
                
            y_true = meta_test.loc[X_test.index, eff_target_col_test].values
            y_true_bin = (y_true == eff_pos_label).astype(int)
            
            acc = accuracy_score(y_true, preds_opt)
            f1 = f1_score(y_true, preds_opt, pos_label=eff_pos_label, average='binary' if len(all_labels)==2 else 'macro')
            mcc = matthews_corrcoef(y_true, preds_opt)
            auc_val = roc_auc_score(y_true_bin, y_probs) if len(np.unique(y_true_bin)) == 2 else 0.0
            
            labels_order = [eff_neg_label, eff_pos_label] if (eff_pos_label in all_labels and len(all_labels)==2) else all_labels
            cm = confusion_matrix(y_true, preds_opt, labels=labels_order)
            
            if len(cm.ravel()) == 4:
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                sensitivity, specificity = 0.0, 0.0

            metrics.update({
                "accuracy": acc, "f1_score": f1, "auc_roc": auc_val,
                "mcc": mcc, "sensitivity": sensitivity, "specificity": specificity
            })

            print("=" * 60)
            print(f"Machine Learning Model Evaluation Report (Target: {eff_target_col_test})")
            print(f"Applied Decision Threshold:           {eff_threshold:.4f}")
            print(f"Designated Positive Class:            {eff_pos_label}")
            print("-" * 60)
            print(f"Accuracy:                             {acc:.4f}")
            print(f"F1-Score:                             {f1:.4f}")
            print(f"AUC-ROC:                              {auc_val:.4f}")
            print(f"Matthews Correlation Coefficient:     {mcc:.4f}")
            print(f"Sensitivity (TPR):                    {sensitivity:.4f}")
            print(f"Specificity (TNR):                    {specificity:.4f}")
            print("-" * 60)
            print("\nDetailed Classification Report:")
            print(classification_report(y_true, preds_opt))
            print("=" * 60)

            # 7. Render evaluation plots.
            if save_pdf:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                            xticklabels=labels_order, yticklabels=labels_order)
                ax1.set_title(f"Confusion Matrix\n(Threshold: {eff_threshold:.3f})", fontsize=13)
                ax1.set_ylabel("Actual Class", fontsize=11)
                ax1.set_xlabel("Predicted Class", fontsize=11)
                
                if len(np.unique(y_true_bin)) == 2:
                    fpr, tpr, _ = roc_curve(y_true_bin, y_probs)
                    ax2.plot([0, 1], [0, 1], ls="--", lw=1.5, color="r")
                    ax2.plot(fpr, tpr, color="orange", lw=2.5, label=f"Test Set (AUC = {auc_val:.2f})")
                    ax2.set_xlim([-0.02, 1.02])
                    ax2.set_ylim([-0.02, 1.02])
                    ax2.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
                    ax2.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
                    ax2.set_title("Receiver Operating Characteristic (ROC)", fontsize=13)
                    ax2.legend(loc="lower right", frameon=False, fontsize=11)
                else:
                    ax2.text(0.5, 0.5, "ROC requires both classes in test set", 
                             ha='center', va='center', fontsize=12)
                
                plt.tight_layout()
                try:
                    plt.savefig(save_pdf, format='pdf', bbox_inches='tight', transparent=True)
                    print(f"[Success] Graphical metrics chart exported to PDF: {save_pdf}")
                except Exception as e:
                    print(f"[Error] Failed to write PDF output: {e}")
                plt.close(fig)
        else:
            print("=" * 60)
            print(f"Prediction Mode Active: No ground truth 'meta_test' provided.")
            print(f"Applied Decision Threshold:           {eff_threshold:.4f}")
            print(f"Designated Positive Class:            {eff_pos_label}")
            print("=" * 60)
            
        return metrics
    
    def export_module_scores(self, train: bool = False, X_train: pd.DataFrame = None, 
                             X_test: pd.DataFrame = None, output_path: str = "module_scores.csv") -> pd.DataFrame:
        """
        Compute and export sample module scores as mean Z-scores.
        
        Args:
            train (bool): If True, refit scalers on X_train and optionally
                process X_test. If False, use the fitted scalers on X_test.
            X_train (pd.DataFrame, optional): Training abundance matrix;
                required when train=True.
            X_test (pd.DataFrame, optional): Test abundance matrix; required
                when train=False.
            output_path (str): Output CSV path.
            
        Returns:
            pd.DataFrame: SampleID, dataset source, and module scores.
        """
        import pandas as pd
        import numpy as np
        
        all_results = []

        if train:
            # Mode 1: fit parameters and compute X_train scores.
            if X_train is None:
                raise ValueError("当 train=True 时，必须传入 'X_train' 参数。")
            
            # Fit scalers on X_train.
            Z_aa_tr = self._prep_block(X_train, self.genes_aa, self.scaler_aa, fit=True)
            Z_carb_tr = self._prep_block(X_train, self.genes_carb, self.scaler_carb, fit=True)
            Z_flag_tr = self._prep_block(X_train, self.genes_flag, self.scaler_flag, fit=True)
            
            df_train = pd.DataFrame({
                "SampleID": X_train.index,
                "Dataset": "Train",
                "His_Score": np.mean(Z_aa_tr, axis=1),
                "PTS_Score": np.mean(Z_carb_tr, axis=1),
                "Fla_Score": np.mean(Z_flag_tr, axis=1)
            })
            all_results.append(df_train)
            
            # If provided, transform X_test with the scalers fitted on X_train.
            if X_test is not None:
                Z_aa_te = self._prep_block(X_test, self.genes_aa, self.scaler_aa, fit=False)
                Z_carb_te = self._prep_block(X_test, self.genes_carb, self.scaler_carb, fit=False)
                Z_flag_te = self._prep_block(X_test, self.genes_flag, self.scaler_flag, fit=False)
                
                df_test = pd.DataFrame({
                    "SampleID": X_test.index,
                    "Dataset": "Test",
                    "His_Score": np.mean(Z_aa_te, axis=1),
                    "PTS_Score": np.mean(Z_carb_te, axis=1),
                    "Fla_Score": np.mean(Z_flag_te, axis=1)
                })
                all_results.append(df_test)
                
        else:
            # Mode 2: compute X_test scores with the fitted parameters.
            if X_test is None:
                raise ValueError("当 train=False 时，必须传入 'X_test' 参数以计算得分。")
            if X_train is not None:
                print("[Warning] 在 train=False 模式下传入了 X_train，该数据将被忽略。")
                
            # Transform X_test with the fitted scalers.
            Z_aa_te = self._prep_block(X_test, self.genes_aa, self.scaler_aa, fit=False)
            Z_carb_te = self._prep_block(X_test, self.genes_carb, self.scaler_carb, fit=False)
            Z_flag_te = self._prep_block(X_test, self.genes_flag, self.scaler_flag, fit=False)
            
            df_test = pd.DataFrame({
                "SampleID": X_test.index,
                "Dataset": "Test",
                "His_Score": np.mean(Z_aa_te, axis=1),
                "PTS_Score": np.mean(Z_carb_te, axis=1),
                "Fla_Score": np.mean(Z_flag_te, axis=1)
            })
            all_results.append(df_test)

        # Combine and save the results.
        final_df = pd.concat(all_results, axis=0).reset_index(drop=True)
        final_df.to_csv(output_path, index=False)
        print(f"[Success] Module scores exported to: {output_path}")
        
        return final_df


# ————————— load pretrained model ———————————

def load_pretrained_model(
    task: Optional[str] = "ibd_control", 
    model_path: Optional[Union[str, Path]] = None
) -> WeightedDistanceKNN:
    """
    Load a bundled pretrained model or a custom local scoring model.

    Args:
        task (str, optional): Pretrained task name. Available values:
            - 'ibd_control': optimized IBD-versus-control classifier (default)
            - 'cd_uc': Crohn's disease versus ulcerative colitis classifier
            Ignored when `model_path` is specified.
        model_path (str or Path, optional): Path to a custom .joblib model.

    Returns:
        WeightedDistanceKNN: Loaded model with its full training state.
    """
    # Route 1: Prefer a custom local model when a path is provided.
    if model_path is not None:
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"未找到用户自定义模型文件: {model_path}")
        
        try:
            model = joblib.load(model_path_obj)
            # Validate the loaded object.
            if not hasattr(model, "predict_meta"):
                raise TypeError("加载的文件不是有效的 WeightedDistanceKNN 模型实例。")
            return model
        except Exception as e:
            raise RuntimeError(f"解析用户自定义模型失败: {e}")

    # Route 2: Load a bundled pretrained model.
    # Map task names to bundled model files.
    TASK_MODEL_MAPPING = {
        "ibd_control": "ibd_control_optimal.joblib",
        "cd_uc": "cd_uc_optimal.joblib",
    }

    if task not in TASK_MODEL_MAPPING:
        raise ValueError(
            f"未知的内置任务名称: '{task}'。当前支持的内置任务: {list(TASK_MODEL_MAPPING.keys())}。\n"
            f"如果您想加载自己的模型，请指定 'model_path' 参数。"
        )

    filename = TASK_MODEL_MAPPING[task]

    # Safely access package data with importlib.resources.
    try:
        # Use the files() API on Python 3.9+.
        ref = importlib.resources.files("impact_ibd.model").joinpath(filename)
        with importlib.resources.as_file(ref) as path:
            return joblib.load(path)
    except AttributeError:
        # Fall back to path() on older Python versions.
        with importlib.resources.path("impact_ibd.model", filename) as path:
            return joblib.load(path)
