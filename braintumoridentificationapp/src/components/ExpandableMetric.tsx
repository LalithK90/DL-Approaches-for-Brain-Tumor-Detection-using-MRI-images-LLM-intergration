import React, { useState } from 'react';
import { IonCard, IonCardContent, IonIcon, IonText } from '@ionic/react';
import { chevronDown, chevronUp, informationCircleOutline } from 'ionicons/icons';
import '../theme/metrics.css';

interface MetricValue {
  value: number;
  interpretation: string;
  level: string;
  explanation: string;
}

interface ExpandableMetricProps {
  title: string;
  metric: MetricValue;
  unit?: string;
  icon?: string;
}

const ExpandableMetric: React.FC<ExpandableMetricProps> = ({ 
  title, 
  metric, 
  unit = '',
  icon = informationCircleOutline 
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'good':
        return '#2dd36f';
      case 'warning':
        return '#ffc409';
      case 'bad':
        return '#eb445a';
      default:
        return '#888888';
    }
  };

  return (
    <IonCard className="metric-card" style={{ borderLeftColor: getLevelColor(metric.level) }}>
      <IonCardContent 
        className="metric-content"
        onClick={() => setIsExpanded(!isExpanded)}
        style={{ cursor: 'pointer' }}
      >
        <div className="metric-header">
          <div className="metric-title-section">
            <h3 className="metric-title">{title}</h3>
            <span className={`metric-level metric-level-${metric.level}`}>
              {metric.level.toUpperCase()}
            </span>
          </div>
          <IonIcon 
            icon={isExpanded ? chevronUp : chevronDown} 
            className="metric-toggle-icon"
          />
        </div>

        <div className="metric-value-section">
          <div className="metric-value">
            <span className="value-number">{metric.value.toFixed(4)}</span>
            {unit && <span className="value-unit">{unit}</span>}
          </div>
          <div className="metric-interpretation">
            {metric.interpretation}
          </div>
        </div>

        {isExpanded && (
          <div className="metric-explanation">
            <div className="explanation-header">
              <IonIcon icon={icon} className="explanation-icon" />
              <span>Explanation</span>
            </div>
            <IonText className="explanation-text">
              {metric.explanation}
            </IonText>
          </div>
        )}
      </IonCardContent>
    </IonCard>
  );
};

export default ExpandableMetric;
