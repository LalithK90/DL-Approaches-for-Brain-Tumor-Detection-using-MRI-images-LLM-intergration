import React from 'react';
import { IonCard, IonCardContent, IonCardHeader, IonCardTitle } from '@ionic/react';
import ExpandableMetric from './ExpandableMetric';
import { flask } from 'ionicons/icons';
import '../theme/metrics.css';

interface MetricValue {
  value: number;
  interpretation: string;
  level: string;
  explanation: string;
}

interface FaithfulnessMetricsProps {
  comprehensiveness: MetricValue;
  sufficiency: MetricValue;
}

const FaithfulnessMetrics: React.FC<FaithfulnessMetricsProps> = ({
  comprehensiveness,
  sufficiency,
}) => {
  return (
    <div className="metrics-section">
      <IonCard className="section-header-card">
        <IonCardHeader>
          <IonCardTitle className="section-title">
            Explanation Faithfulness Metrics
          </IonCardTitle>
        </IonCardHeader>
        <IonCardContent className="section-description">
          <p>
            These metrics assess how well the highlighted regions actually explain the model's decision-making process.
            Faithfulness indicates that removing important regions decreases model confidence (comprehensiveness),
            and that keeping only important regions maintains confidence (sufficiency).
          </p>
        </IonCardContent>
      </IonCard>

      <ExpandableMetric
        title="Comprehensiveness"
        metric={comprehensiveness}
        unit=""
        icon={flask}
      />
      
      <ExpandableMetric
        title="Sufficiency"
        metric={sufficiency}
        unit=""
        icon={flask}
      />

      <div className="metrics-info-box">
        <h4>What These Metrics Mean</h4>
        <ul>
          <li><strong>Comprehensiveness:</strong> How much does model confidence drop when important regions are removed?</li>
          <li><strong>Sufficiency:</strong> How much of the model's confidence can be maintained with only important regions?</li>
        </ul>
      </div>
    </div>
  );
};

export default FaithfulnessMetrics;
