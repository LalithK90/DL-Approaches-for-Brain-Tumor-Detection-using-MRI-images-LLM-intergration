import {
  IonApp,
  IonContent,
  IonPage,
  IonRouterOutlet,
  setupIonicReact,
  IonHeader,
  IonToolbar,
  IonTitle,
  IonButton,
  IonItem,
  IonLabel,
  IonCard,
  IonCardContent,
  IonCardHeader,
  IonCardTitle,
  IonGrid,
  IonRow,
  IonCol,
  IonImg,
  IonTextarea,
  IonList,
  IonSelect,
  IonSelectOption,
  IonLoading,
  IonAlert,
  IonIcon,
  IonFab,
  IonFabButton,
  IonModal,
  IonButtons,
  IonChip,
  IonBadge,
  IonRange
} from '@ionic/react';
import { IonReactRouter } from '@ionic/react-router';
import { Route } from 'react-router-dom';
import { useState, useRef, useEffect } from 'react';
import { cloudUpload, chatbubbles, close, send, medkit, analytics, logOut } from 'ionicons/icons';
import Login from './pages/Login';
import Footer from './components/Footer';
import ReactMarkdown from 'react-markdown';

/* Core CSS required for Ionic components to work properly */
import '@ionic/react/css/core.css';
import '@ionic/react/css/normalize.css';
import '@ionic/react/css/structure.css';
import '@ionic/react/css/typography.css';
import '@ionic/react/css/padding.css';
import '@ionic/react/css/float-elements.css';
import '@ionic/react/css/text-alignment.css';
import '@ionic/react/css/text-transformation.css';
import '@ionic/react/css/flex-utils.css';
import '@ionic/react/css/display.css';
import '@ionic/react/css/palettes/dark.system.css';
import './theme/variables.css';

setupIonicReact();

// Prefer Vite proxy in development to avoid CORS and allow cookies
const API_BASE_URL = import.meta.env.DEV ? '' : (import.meta.env.VITE_API_BASE_URL || '');
// status, logout and other fetch calls now use `${API_BASE_URL}`

interface PredictionResult {
  original: string;
  gradcam: string;
  saliency: string;
  lime: string;
  gradcam_analysis: string;
  gradcam_heatmap: string;
  prediction: string;
  confidence: {
    value: number;
    interpretation: string;
    level: string;
  };
  patient_info: {
    patient_id: string;
    patient_description: object;
    symptoms: Array<object>;
  };
  top3: Array<{ label: string; probability: number }>;
  entropy: { value: number; interpretation: string; level: string; explanation: string };
  margin: { value: number; interpretation: string; level: string; explanation: string };
  brier: { value: number; interpretation: string; level: string; explanation: string };
  dice: { value: number; interpretation: string; level: string; explanation: string };
  iou: { value: number; interpretation: string; level: string; explanation: string };
  mc_variance: { value: number; interpretation: string; level: string; explanation: string };
  gradcam_explanation: string;
  lime_explanation: string;
  metrics_explanation: string;
  saliency_explanation: string;

  final_report: string | null;
  // Optional fields provided by backend (align with web app)
  activation_ratio?: number;
  mc_confidence_interval?: [number, number];
  center_distance?: number;
}

interface ChatMessage {
  text: string;
  isUser: boolean;
  timestamp: Date;
}

interface User {
  id: string;
  username: string;
  roles: string[];
}

const BrainTumorApp: React.FC<{ user: User; onLogout: () => void }> = ({ user, onLogout }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('propose_balance');
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showAlert, setShowAlert] = useState(false);
  const [alertMessage, setAlertMessage] = useState('');
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [imageSize, setImageSize] = useState<number>(100);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const models = [
    { value: 'propose_balance', label: 'Proposed Model (Balanced)' },
    { value: 'propose_inbalance', label: 'Proposed Model (Imbalanced)' },
    { value: 'ResNet50_balance', label: 'ResNet50 (Balanced)' },
    { value: 'ResNet50_inbalance', label: 'ResNet50 (Imbalanced)' },
    { value: 'vgg16_balance', label: 'VGG16 (Balanced)' },
    { value: 'vgg16_inbalance', label: 'VGG16 (Imbalanced)' },
    { value: 'vgg19_balance', label: 'VGG19 (Balanced)' },
    { value: 'vgg19_inbalance', label: 'VGG19 (Imbalanced)' },
    { value: 'GoogleLeNet_balance', label: 'GoogleLeNet (Balanced)' },
    { value: 'GoogleLeNet_inbalance', label: 'GoogleLeNet (Imbalanced)' },
    { value: 'MobileVNet_balance', label: 'MobileVNet (Balanced)' },
    { value: 'MobileVNet_inbalance', label: 'MobileVNet (Imbalanced)' }
  ];

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setAlertMessage('Please select an image file first.');
      setShowAlert(true);
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    // remove spaces from original filename and persist for later use (e.g. chat endpoint)
    const sanitizedFilename = selectedFile.name.replace(/\s+/g, '');
    localStorage.setItem('uploadedFileName', sanitizedFilename);
    // include the sanitized filename when appending the file so backend and session use the same name
    formData.append('file', selectedFile, sanitizedFilename);
    formData.append('model_name', selectedModel);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      // Transform top3 from array of arrays to array of objects
      result.top3 = result.top3.map(([label, probability]: [string, number]) => ({ label, probability }));
      setPrediction(result as PredictionResult);
  
      if (!response.ok) {
  const err = await response.json();
  throw new Error(err.error ?? `HTTP error! status: ${response.status}`);
}
    } catch (error) {
      console.error('Error uploading file:', error);
      setAlertMessage('Error uploading file. Please make sure the Flask server is running and you are logged in.');
      setShowAlert(true);
    } finally {
      setIsLoading(false);
    }
  };

  const sendChatMessage = async () => {
    if (!chatInput.trim() || !prediction) return;

    const userMessage: ChatMessage = {
      text: chatInput,
      isUser: true,
      timestamp: new Date()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setIsChatLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: chatInput,
          image: localStorage.getItem('uploadedFileName')
         }),
        credentials: 'include'
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      const aiMessage: ChatMessage = {
        text: result.response,
        isUser: false,
        timestamp: new Date()
      };

      setChatMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error sending chat message:', error);
      const errorMessage: ChatMessage = {
        text: 'Sorry, I encountered an error. Please try again.',
        isUser: false,
        timestamp: new Date()
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const getConfidenceColor = (level: string) => {
    switch (level) {
      case 'good': return 'success';
      case 'warning': return 'warning';
      case 'bad': return 'danger';
      default: return 'medium';
    }
  };

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Brain Tumor Identification</IonTitle>
          <IonButtons slot="end">
            <IonButton fill="clear" onClick={onLogout}>
              <IonIcon icon={logOut} />
              <IonLabel style={{ marginLeft: '8px' }}>Logout ({user.username})</IonLabel>
            </IonButton>
          </IonButtons>
        </IonToolbar>
      </IonHeader>
      
      <IonContent fullscreen>
        <div style={{ padding: '16px' }}>
          {/* File Upload Section */}
          <IonCard>
            <IonCardHeader>
              <IonCardTitle>
                <IonIcon icon={cloudUpload} style={{ marginRight: '8px' }} />
                Upload MRI Image
              </IonCardTitle>
            </IonCardHeader>
            <IonCardContent>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                ref={fileInputRef}
                style={{ display: 'none' }}
              />
              
              <IonButton
                expand="block"
                fill="outline"
                onClick={() => fileInputRef.current?.click()}
              >
                {selectedFile ? selectedFile.name : 'Choose Image File'}
              </IonButton>
              
              <IonItem>
                <IonLabel>Select Model:</IonLabel>
                <IonSelect
                  value={selectedModel}
                  onIonChange={(e) => setSelectedModel(e.detail.value as string)}
                >
                  {models.map(model => (
                    <IonSelectOption key={model.value} value={model.value}>
                      {model.label}
                    </IonSelectOption>
                  ))}
                </IonSelect>
              </IonItem>
              
              <IonButton
                expand="block"
                onClick={handleUpload}
                disabled={!selectedFile || isLoading}
                style={{ marginTop: '16px' }}
              >
                <IonIcon icon={analytics} slot="start" />
                Analyze Image
              </IonButton>
            </IonCardContent>
          </IonCard>

          {/* Results Section */}
          {prediction && (
            <>
              {/* Prediction Results */}
              <IonCard>
                <IonCardHeader>
                  <IonCardTitle>
                    <IonIcon icon={medkit} style={{ marginRight: '8px' }} />
                    Diagnosis Results
                  </IonCardTitle>
                </IonCardHeader>
                <IonCardContent>
                  <IonGrid>
                    <IonRow>
                      <IonCol size="12">
                        <div style={{ textAlign: 'center', marginBottom: '16px' }}>
                          <h2 style={{ margin: '8px 0', color: 'var(--ion-color-primary)' }}>
                            {prediction.prediction}
                          </h2>
                          <IonChip color={getConfidenceColor(prediction.confidence.level)}>
                            Confidence: {(prediction.confidence.value * 100).toFixed(1)}% 
                            ({prediction.confidence.interpretation})
                          </IonChip>
                        </div>
                      </IonCol>
                    </IonRow>
                    
                    <IonItem>
                      <IonLabel>Image Size</IonLabel>
                      <IonRange min={50} max={200} value={imageSize} onIonChange={e => setImageSize(e.detail.value as number)} />
                    </IonItem>
                    <IonRow>
                      <IonCol size="6">
                        <div style={{ textAlign: 'center' }}>
                          <h4>Original Image</h4>
                          <IonImg src={prediction.original} alt="Original" style={{ width: `${imageSize}%` }} />
                        </div>
                      </IonCol>
                      <IonCol size="6">
                        <div style={{ textAlign: 'center' }}>
                          <h4>Grad-CAM (Overlay)</h4>
                          <IonImg src={prediction.gradcam} alt="Grad-CAM" style={{ width: `${imageSize}%` }} />
                        
                        </div>
                      </IonCol>
                    </IonRow>
                    {/* Additional XAI Images if available */}
                    {(prediction.gradcam_analysis || prediction.gradcam_heatmap) && (
                      <IonRow>
                        {prediction.gradcam_analysis && (
                          <IonCol size="6">
                            <div style={{ textAlign: 'center' }}>
                              <h4>Grad-CAM Analysis</h4>
                              <IonImg src={prediction.gradcam_analysis} alt="Grad-CAM Analysis" style={{ width: `${imageSize}%` }} />
                            </div>
                          </IonCol>
                        )}
                        {prediction.gradcam_heatmap && (
                          <IonCol size="6">
                            <div style={{ textAlign: 'center' }}>
                              <h4>Grad-CAM Heatmap</h4>
                              <IonImg src={prediction.gradcam_heatmap} alt="Grad-CAM Heatmap" style={{ width: `${imageSize}%` }} />
                            </div>
                          </IonCol>
                        )}
                      </IonRow>
                    )}
                    
                    <IonRow>
                      <IonCol size="6">
                        <div style={{ textAlign: 'center' }}>
                          <h4>Saliency Map</h4>
                          <IonImg src={prediction.saliency} alt="Saliency" style={{ width: `${imageSize}%` }} />
                        
                        </div>
                      </IonCol>
                      <IonCol size="6">
                        <div style={{ textAlign: 'center' }}>
                          <h4>LIME Explanation</h4>
                          <IonImg src={prediction.lime} alt="LIME" style={{ width: `${imageSize}%` }} />
                          
                        </div>
                      </IonCol>
                    </IonRow>
                  </IonGrid>
                  
                  {/* Top 3 Predictions */}
                  <div style={{ marginTop: '16px' }}>
                    <h4>Top 3 Predictions:</h4>
                    <IonList>
                      {prediction.top3.map((item, index) => (
                        <IonItem key={index}>
                          <IonLabel>
                            <h3>{item.label}</h3>
                            <p>{(item.probability * 100).toFixed(2)}%</p>
                          </IonLabel>
                          <IonBadge slot="end">{index + 1}</IonBadge>
                        </IonItem>
                      ))}
                    </IonList>
                  </div>

                  {/* Additional Metrics */}
                  <div style={{ marginTop: '16px' }}>
                    <h4>Confidence Metrics</h4>
                    <IonList>
                      <IonItem>
                        <IonLabel>Brier Score</IonLabel>
                        <IonBadge color={prediction.brier.level}> Value: {prediction.brier.value.toFixed(2)} <br/>  Status: {prediction.brier.interpretation}</IonBadge>
                      </IonItem>
                      <IonItem>
                        <IonLabel>Entropy</IonLabel>
                        <IonBadge color={prediction.entropy.level}> Value: {prediction.entropy.value.toFixed(2)} <br/> Status: {prediction.entropy.interpretation}</IonBadge>
                      </IonItem>
                      <IonItem>
                        <IonLabel>Margin</IonLabel>
                        <IonBadge color={prediction.margin.level}> Value: {prediction.margin.value.toFixed(2)} <br/> Status: {prediction.margin.interpretation}</IonBadge>
                      </IonItem>
                      <IonItem>
                        <IonLabel>MC Variance</IonLabel>
                        <IonBadge color={prediction.mc_variance.level}> Value: {prediction.mc_variance.value.toFixed(2)} <br/> Status: {prediction.mc_variance.interpretation}</IonBadge>
                      </IonItem>
                      <IonItem>
                        <IonLabel>Dice Agreement</IonLabel>
                        <IonBadge color={prediction.dice.level}> Value: {prediction.dice.value.toFixed(2)} <br/> Status: {prediction.dice.interpretation}</IonBadge>
                      </IonItem>
                      <IonItem>
                        <IonLabel>IoU Agreement</IonLabel>
                        <IonBadge color={prediction.iou.level}> Value: {prediction.iou.value.toFixed(2)} <br/> Status: {prediction.iou.interpretation}</IonBadge>
                      </IonItem>
                      {typeof prediction.activation_ratio === 'number' && (
                        <IonItem>
                          <IonLabel>Activation Ratio</IonLabel>
                          <IonBadge> {(prediction.activation_ratio * 100).toFixed(2)}% </IonBadge>
                        </IonItem>
                      )}
                      {Array.isArray(prediction.mc_confidence_interval) && prediction.mc_confidence_interval.length === 2 && (
                        <IonItem>
                          <IonLabel>MC Confidence Interval</IonLabel>
                          <IonBadge> [{prediction.mc_confidence_interval[0].toFixed(2)}, {prediction.mc_confidence_interval[1].toFixed(2)}] </IonBadge>
                        </IonItem>
                      )}
                      {typeof prediction.center_distance === 'number' && (
                        <IonItem>
                          <IonLabel>Center Distance</IonLabel>
                          <IonBadge> {prediction.center_distance.toFixed(2)} </IonBadge>
                        </IonItem>
                      )}
                    </IonList>
                  </div>
                </IonCardContent>
              </IonCard>

              {/* AI Report */}
              {prediction.final_report && (
                <IonCard>
                  <IonCardHeader>
                    <IonCardTitle>AI Medical Report</IonCardTitle>
                  </IonCardHeader>
                  <IonCardContent>
                    {(() => {
                      if (!prediction.final_report) return <></>;
                      const startIndex = prediction.final_report.indexOf('<think>');
                      const endIndex = prediction.final_report.indexOf('</think>');
                      let thinkContent = '';
                      let reportContent = prediction.final_report;
                      if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
                        thinkContent = prediction.final_report.substring(startIndex + 7, endIndex).trim();
                        reportContent = (prediction.final_report.substring(0, startIndex) + prediction.final_report.substring(endIndex + 8)).trim();
                      }
                      return (
                        <>
                          {thinkContent && (
                            <div style={{ marginBottom: '16px', padding: '8px', backgroundColor: 'var(--ion-color-light)', borderRadius: '4px' }}>
                              <h5>AI Thinking Process:</h5>
                              <p>{thinkContent}</p>
                            </div>
                          )}
                          <ReactMarkdown>{reportContent}</ReactMarkdown>
                        <br/>
                        <br/>
                        <br/>
                        <br/>
                        <br/>
                        <br/>
                        <br/>
                        <br/>
                        <br/>
                        <br/>
                        </>
                      );
                    })()}

                  </IonCardContent>
                  <IonCardContent>
                {/* Footer */}
              <Footer></Footer>
                  </IonCardContent>
                </IonCard>
              )}
              
              
              {/* Chat FAB */}
              <IonFab style={{ position: 'fixed', bottom: '20px', right: '20px', zIndex: 1000 }}>
                <IonFabButton onClick={() => setIsChatOpen(true)}>
                  <IonIcon icon={chatbubbles} />
                </IonFabButton>
              </IonFab>
            </>
          )}
        </div>

        {/* Chat Modal */}
        <IonModal isOpen={isChatOpen} onDidDismiss={() => setIsChatOpen(false)}>
          <IonHeader>
            <IonToolbar>
              <IonTitle>AI Assistant</IonTitle>
              <IonButtons slot="end">
                <IonButton onClick={() => setIsChatOpen(false)}>
                  <IonIcon icon={close} />
                </IonButton>
              </IonButtons>
            </IonToolbar>
          </IonHeader>
          
          <IonContent>
            <div style={{ padding: '16px', height: '100%', display: 'flex', flexDirection: 'column' }}>
              <div style={{ flex: 1, overflowY: 'auto', marginBottom: '16px' }}>
                {chatMessages.map((msg, index) => (
                  <div
                    key={index}
                    style={{
                      marginBottom: '12px',
                      textAlign: msg.isUser ? 'right' : 'left'
                    }}
                  >
                    <div
                      style={{
                        display: 'inline-block',
                        padding: '8px 12px',
                        borderRadius: '12px',
                        backgroundColor: msg.isUser ? 'var(--ion-color-primary)' : 'var(--ion-color-light)',
                        color: msg.isUser ? 'white' : 'var(--ion-color-dark)',
                        maxWidth: '80%',
                        whiteSpace: 'pre-wrap'
                      }}
                    >
                      {msg.isUser ? msg.text : <ReactMarkdown>{msg.text}</ReactMarkdown>}
                    </div>
                  </div>
                ))}
                {isChatLoading && (
                  <div style={{ textAlign: 'left', marginBottom: '12px' }}>
                    <div
                      style={{
                        display: 'inline-block',
                        padding: '8px 12px',
                        borderRadius: '12px',
                        backgroundColor: 'var(--ion-color-light)',
                        color: 'var(--ion-color-dark)'
                      }}
                    >
                      AI is typing...
                    </div>
                  </div>
                )}
              </div>
              
              <div style={{ display: 'flex', gap: '8px' }}>
                <IonTextarea
                  value={chatInput}
                  onIonInput={(e) => setChatInput(e.detail.value!)}
                  placeholder="Ask about the diagnosis..."
                  rows={2}
                  style={{ flex: 1 }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      sendChatMessage();
                    }
                  }}
                />
                <IonButton
                  onClick={sendChatMessage}
                  disabled={!chatInput.trim() || isChatLoading}
                >
                  <IonIcon icon={send} />
                </IonButton>
              </div>
            </div>
          </IonContent>
        </IonModal>

        <IonLoading isOpen={isLoading} message="Analyzing image..." />
        
        <IonAlert
          isOpen={showAlert}
          onDidDismiss={() => setShowAlert(false)}
          header="Alert"
          message={alertMessage}
          buttons={['OK']}
        />
        
      </IonContent>
    </IonPage>
  );
};

const App = () => {
  const [user, setUser] = useState<User | null>(null);
  const [isCheckingAuth, setIsCheckingAuth] = useState<boolean>(true);

  useEffect(() => {
    const checkAuthStatus = async () => {
      try {
        // Check authentication status with Flask backend
        const response = await fetch(API_BASE_URL+'/login/api/status', {
          method: 'GET',
          credentials: 'include'
        });
        
        if (response.ok) {
          const result = await response.json();
          if (result.authenticated && result.user) {
            setUser(result.user);
            localStorage.setItem('user', JSON.stringify(result.user));
          } else {
            // User not authenticated, clear any stored data
            setUser(null);
            localStorage.removeItem('user');
          }
        } else {
          // If status check fails, fall back to localStorage but user will need to re-authenticate
          const storedUser = localStorage.getItem('user');
          if (storedUser) {
            try {
              setUser(JSON.parse(storedUser));
            } catch (error) {
              console.error('Error parsing stored user:', error);
              localStorage.removeItem('user');
            }
          }
        }
      } catch (error) {
        console.error('Error checking auth status:', error);
        // If Flask server is not available, clear authentication
        setUser(null);
        localStorage.removeItem('user');
      } finally {
        setIsCheckingAuth(false);
      }
    };
    
    checkAuthStatus();
  }, []);

  const handleLoginSuccess = (userData: User) => {
    setUser(userData);
    // Store user in localStorage for persistence
    localStorage.setItem('user', JSON.stringify(userData));
  };

  const handleLogout = async () => {
    try {
      // Try to call the logout API if available
      await fetch(API_BASE_URL+'/login/api/logout', {
        method: 'POST',
        credentials: 'include'
      }).catch(err => console.log('Logout API not available:', err));
      
      // Always clear local state regardless of API success
      setUser(null);
      localStorage.removeItem('user');
    } catch (error) {
      console.error('Error logging out:', error);
    }
  };

  if (isCheckingAuth) {
    return (
      <IonApp>
        <IonLoading isOpen={true} message="Checking authentication..." />
      </IonApp>
    );
  }

  return (
    <IonApp>
      <IonReactRouter>
        <IonRouterOutlet>
          <Route exact path="/">
            {user ? (
              <BrainTumorApp user={user} onLogout={handleLogout} />
            ) : (
              <Login onLoginSuccess={handleLoginSuccess} />
            )}
          </Route>
        </IonRouterOutlet>
      </IonReactRouter>
    </IonApp>
  );
};

export default App;
