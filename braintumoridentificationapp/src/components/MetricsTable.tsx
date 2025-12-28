import { IonCard, IonCardContent, IonCardHeader, IonCardTitle, IonBadge } from '@ionic/react';
import './MetricsTable.css';

interface MetricValue {
  value: number;
  interpretation: string;
  level: string;
  explanation: string;
}

interface MetricsTableProps {
  metrics: {
    [key: string]: MetricValue;
  };
  title?: string;
}

const MetricsTable: React.FC<MetricsTableProps> = ({ metrics, title = "Explainability Metrics" }) => {
  const getBadgeColor = (level: string) => {
    switch (level) {
      case 'good': return 'success';
      case 'warning': return 'warning';
      case 'bad': return 'danger';
      default: return 'medium';
    }
  };

  const metricEntries = Object.entries(metrics).map(([key, metric]) => {
    // Convert snake_case to Title Case
    const displayName = key
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');

    return { key, displayName, ...metric };
  });

  return (
    <IonCard className="metrics-table-card">
      <IonCardHeader>
        <IonCardTitle>{title}</IonCardTitle>
      </IonCardHeader>
      <IonCardContent>
        <div className="table-wrapper">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Status</th>
                <th>Interpretation</th>
              </tr>
            </thead>
            <tbody>
              {metricEntries.map((metric) => (
                <tr key={metric.key} className={`metric-row metric-row-${metric.level}`}>
                  <td className="metric-name">
                    <strong>{metric.displayName}</strong>
                    <p className="metric-explanation">{metric.explanation}</p>
                  </td>
                  <td className="metric-value">
                    <code>{typeof metric.value === 'number' ? metric.value.toFixed(4) : metric.value}</code>
                  </td>
                  <td className="metric-status">
                    <IonBadge color={getBadgeColor(metric.level)}>
                      {metric.level.toUpperCase()}
                    </IonBadge>
                  </td>
                  <td className="metric-interpretation">
                    {metric.interpretation}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </IonCardContent>
    </IonCard>
  );
};

export default MetricsTable;
