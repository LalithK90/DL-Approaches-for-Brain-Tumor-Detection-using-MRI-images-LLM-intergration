import React from 'react';
import { IonCard, IonCardContent, IonCardHeader, IonCardTitle, IonGrid, IonRow, IonCol, IonText } from '@ionic/react';
import '../theme/metrics.css';

interface MetricValue {
  value: number;
  interpretation: string;
  level: string;
  explanation: string;
}

interface MetricsComparisonProps {
  dice: MetricValue;
  iou: MetricValue;
  comprehensiveness: MetricValue;
  sufficiency: MetricValue;
  deletion_auc: MetricValue;
  insertion_auc: MetricValue;
}

const MetricsComparison: React.FC<MetricsComparisonProps> = ({
  dice,
  iou,
  comprehensiveness,
  sufficiency,
  deletion_auc,
  insertion_auc,
}) => {
  const MetricRow = ({ label, metric }: { label: string; metric: MetricValue }) => (
    <IonRow className="comparison-row">
      <IonCol size="6" className="comparison-label">
        <strong>{label}</strong>
      </IonCol>
      <IonCol size="3" className="comparison-value">
        <span className={`value-badge badge-${metric.level}`}>
          {metric.value.toFixed(4)}
        </span>
      </IonCol>
      <IonCol size="3" className="comparison-interpretation">
        <IonText className={`interpretation-text interpretation-${metric.level}`}>
          {metric.interpretation}
        </IonText>
      </IonCol>
    </IonRow>
  );

  return (
    <div className="metrics-section">
      <IonCard className="comparison-card">
        <IonCardHeader>
          <IonCardTitle className="section-title">
            Complete Metrics Overview
          </IonCardTitle>
        </IonCardHeader>
        <IonCardContent className="comparison-content">
          <div className="comparison-description">
            <p>
              Quick reference table comparing all explanation quality metrics. 
              Lower values indicate better explanation quality for some metrics, while higher indicates better for others.
            </p>
          </div>

          <IonGrid className="comparison-grid">
            <IonRow className="comparison-header">
              <IonCol size="6"><strong>Metric</strong></IonCol>
              <IonCol size="3"><strong>Score</strong></IonCol>
              <IonCol size="3"><strong>Status</strong></IonCol>
            </IonRow>

            <div className="metric-category">
              <div className="category-title">Agreement Metrics</div>
              <MetricRow label="Dice Coefficient" metric={dice} />
              <MetricRow label="Intersection over Union" metric={iou} />
            </div>

            <div className="metric-category">
              <div className="category-title">Faithfulness Metrics</div>
              <MetricRow label="Comprehensiveness" metric={comprehensiveness} />
              <MetricRow label="Sufficiency" metric={sufficiency} />
            </div>

            <div className="metric-category">
              <div className="category-title">AUC-Based Metrics</div>
              <MetricRow label="Deletion AUC" metric={deletion_auc} />
              <MetricRow label="Insertion AUC" metric={insertion_auc} />
            </div>
          </IonGrid>

          <div className="metrics-legend">
            <h4>Legend</h4>
            <div className="legend-items">
              <div className="legend-item">
                <span className="legend-badge badge-good"></span>
                <span>Good - High Quality</span>
              </div>
              <div className="legend-item">
                <span className="legend-badge badge-warning"></span>
                <span>Warning - Moderate Quality</span>
              </div>
              <div className="legend-item">
                <span className="legend-badge badge-bad"></span>
                <span>Bad - Low Quality</span>
              </div>
            </div>
          </div>
        </IonCardContent>
      </IonCard>
    </div>
  );
};

export default MetricsComparison;
