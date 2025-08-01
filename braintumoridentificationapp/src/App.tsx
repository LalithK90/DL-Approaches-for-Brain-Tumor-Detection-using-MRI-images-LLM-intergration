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
  IonBadge
} from '@ionic/react';
import { IonReactRouter } from '@ionic/react-router';
import { Route } from 'react-router-dom';
import { useState, useRef, useEffect } from 'react';
import { cloudUpload, chatbubbles, close, send, medkit, analytics, logOut } from 'ionicons/icons';
import Login from './pages/Login';
import Footer from './components/Footer';

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

const API_BASE_URL = "http://127.0.0.1:5000";

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
  patient_info: unknown;
  top3: Array<{ label: string; probability: number }>;
  entropy: unknown;
  margin: unknown;
  brier: unknown;
  final_report: string;
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
    formData.append('file', selectedFile);
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

      const result: PredictionResult = await response.json();
      setPrediction(result);
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
        body: JSON.stringify({ message: chatInput }),
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
                    
                    <IonRow>
                      <IonCol size="6">
                        <div style={{ textAlign: 'center' }}>
                          <h4>Original Image</h4>
                          <IonImg src={prediction.original} alt="Original" />
                        </div>
                      </IonCol>
                      <IonCol size="6">
                        <div style={{ textAlign: 'center' }}>
                          <h4>Grad-CAM Heatmap</h4>
                          <IonImg src={prediction.gradcam} alt="Grad-CAM" />
                        </div>
                      </IonCol>
                    </IonRow>
                    
                    <IonRow>
                      <IonCol size="6">
                        <div style={{ textAlign: 'center' }}>
                          <h4>Saliency Map</h4>
                          <IonImg src={prediction.saliency} alt="Saliency" />
                        </div>
                      </IonCol>
                      <IonCol size="6">
                        <div style={{ textAlign: 'center' }}>
                          <h4>LIME Explanation</h4>
                          <IonImg src={prediction.lime} alt="LIME" />
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
                </IonCardContent>
              </IonCard>

              {/* AI Report */}
              {prediction.final_report && (
                <IonCard>
                  <IonCardHeader>
                    <IonCardTitle>AI Medical Report</IonCardTitle>
                  </IonCardHeader>
                  <IonCardContent>
                    <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
                      {prediction.final_report}
                    </div>
                  </IonCardContent>
                </IonCard>
              )}

              {/* Chat FAB */}
              <IonFab vertical="bottom" horizontal="end" slot="fixed">
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
                      {msg.text}
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
        
        <Footer />
      </IonContent>
    </IonPage>
  );
};

const App = () => {
  const [user, setUser] = useState<User | null>(null);
  const [isCheckingAuth, setIsCheckingAuth] = useState<boolean>(false);

  useEffect(() => {
    
    // Check if we have a user in localStorage from previous session
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser));
      } catch (error) {
        console.error('Error parsing stored user:', error);
        localStorage.removeItem('user');
      }
    }
  }, []);

  const handleLoginSuccess = (userData: User) => {
    setUser(userData);
    // Store user in localStorage for persistence
    localStorage.setItem('user', JSON.stringify(userData));
  };

  const handleLogout = async () => {
    try {
      // Try to call the logout API if available
      await fetch('http://localhost:5000/login/api/logout', {
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
