# evaluation/__init__.py

from evaluation.metrics import validate_projection_head
from evaluation.heatmap import (
    compute_multiscale_heatmap,
    visualize_heatmap,
    plot_heatmap,
    compute_heatmap_stats,
    compute_localization_metrics,
    extract_dinov2_features,
    prepare_multiscale_targets
)
from evaluation.anomaly_scorer import (
    AnomalyScorer,
    evaluate_anomaly_detector
)

__all__ = [
    'validate_projection_head',
    'compute_multiscale_heatmap',
    'visualize_heatmap',
    'plot_heatmap',
    'compute_heatmap_stats',
    'compute_localization_metrics',
    'extract_dinov2_features',
    'prepare_multiscale_targets',
    'AnomalyScorer',
    'evaluate_anomaly_detector'
]
