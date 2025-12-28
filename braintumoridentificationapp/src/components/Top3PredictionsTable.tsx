import { IonCard, IonCardContent, IonCardHeader, IonCardTitle, IonBadge } from '@ionic/react';
import './Top3PredictionsTable.css';

interface Prediction {
  label: string;
  probability: number;
}

interface Top3PredictionsTableProps {
  predictions: Prediction[];
  title?: string;
}

const Top3PredictionsTable: React.FC<Top3PredictionsTableProps> = ({ 
  predictions, 
  title = "Top 3 Predictions" 
}) => {
  const getColor = (index: number) => {
    switch (index) {
      case 0: return 'gold';
      case 1: return 'medium';
      case 2: return 'danger';
      default: return 'primary';
    }
  };

  const getRankMedal = (index: number) => {
    switch (index) {
      case 0: return '🥇';
      case 1: return '🥈';
      case 2: return '🥉';
      default: return `#${index + 1}`;
    }
  };

  return (
    <IonCard className="top3-predictions-card">
      <IonCardHeader>
        <IonCardTitle>{title}</IonCardTitle>
      </IonCardHeader>
      <IonCardContent>
        <div className="predictions-list">
          {predictions.slice(0, 3).map((pred, index) => (
            <div key={index} className={`prediction-item prediction-rank-${index}`}>
              <div className="prediction-rank">
                <span className="rank-medal">{getRankMedal(index)}</span>
              </div>
              
              <div className="prediction-info">
                <div className="prediction-label">
                  <h4>{pred.label}</h4>
                </div>
                
                <div className="prediction-bar-container">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ 
                        width: `${(pred.probability * 100)}%`,
                        background: `linear-gradient(90deg, #667eea 0%, #764ba2 100%)`
                      }}
                    />
                  </div>
                </div>
              </div>
              
              <div className="prediction-percentage">
                <strong>{(pred.probability * 100).toFixed(2)}%</strong>
                <IonBadge color={getColor(index)}>
                  {(pred.probability * 100).toFixed(1)}
                </IonBadge>
              </div>
            </div>
          ))}
        </div>

        {/* Summary stats */}
        <div className="predictions-summary" style={{ marginTop: '16px', paddingTop: '16px', borderTop: '1px solid #e0e0e0' }}>
          <div className="summary-stat">
            <span className="stat-label">Top Prediction Confidence:</span>
            <strong className="stat-value">{(predictions[0]?.probability * 100).toFixed(2)}%</strong>
          </div>
          <div className="summary-stat">
            <span className="stat-label">Prediction Margin (1st vs 2nd):</span>
            <strong className="stat-value">
              {predictions[0] && predictions[1] 
                ? ((predictions[0].probability - predictions[1].probability) * 100).toFixed(2) + '%'
                : 'N/A'
              }
            </strong>
          </div>
        </div>
      </IonCardContent>
    </IonCard>
  );
};

export default Top3PredictionsTable;
