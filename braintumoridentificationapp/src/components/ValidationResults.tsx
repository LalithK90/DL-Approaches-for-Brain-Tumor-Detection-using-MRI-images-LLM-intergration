import React from 'react';
import { IonCard, IonCardContent, IonCardHeader, IonCardTitle } from '@ionic/react';
import ExpandableMetric from './ExpandableMetric';
import { analytics } from 'ionicons/icons';
import '../theme/metrics.css';

interface MetricValue {
  value: number;
  interpretation: string;
  level: string;
  explanation: string;
}

interface ValidationResultsProps {
  deletion_auc: MetricValue;
  insertion_auc: MetricValue;
  randomized_weights_corr: MetricValue;
}

const ValidationResults: React.FC<ValidationResultsProps> = ({
  deletion_auc,
  insertion_auc,
  randomized_weights_corr,
}) => {
  return (
    <div className="metrics-section">
      <IonCard className="section-header-card">
        <IonCardHeader>
          <IonCardTitle className="section-title">
            Validation & Robustness Metrics
          </IonCardTitle>
        </IonCardHeader>
        <IonCardContent className="section-description">
          <p>
            These metrics validate the explanation's reliability and measure how sensitive the model is to important regions.
            AUC metrics rank feature importance, while the randomized weights test ensures explanations depend on learned features.
          </p>
        </IonCardContent>
      </IonCard>

      <ExpandableMetric
        title="Deletion AUC"
        metric={deletion_auc}
        unit=""
        icon={analytics}
      />
      
      <ExpandableMetric
        title="Insertion AUC"
        metric={insertion_auc}
        unit=""
        icon={analytics}
      />

      <ExpandableMetric
        title="Randomized Weights Test"
        metric={randomized_weights_corr}
        unit=""
        icon={analytics}
      />

      <div className="metrics-info-box">
        <h4>What These Metrics Mean</h4>
        <ul>
          <li><strong>Deletion AUC:</strong> How quickly does confidence drop as important regions are progressively removed? Higher is better.</li>
          <li><strong>Insertion AUC:</strong> How quickly does confidence increase as important regions are progressively inserted? Higher is better.</li>
          <li><strong>Randomized Weights Test:</strong> Sanity check that explanations depend on learned model features, not random patterns. Lower correlation is better.</li>
        </ul>
      </div>
    </div>
  );
};

export default ValidationResults;
