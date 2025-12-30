import { IonApp, IonContent, IonPage, IonRouterOutlet, setupIonicReact, IonHeader, IonToolbar, IonTitle, IonButton, IonItem, IonLabel, IonCard, IonCardContent, IonCardHeader, IonCardTitle, IonGrid, IonRow, IonCol, IonImg, IonTextarea, IonList, IonSelect, IonSelectOption, IonLoading, IonAlert, IonIcon, IonButtons, IonChip, IonBadge, IonRange, IonFooter } from '@ionic/react';
import { IonReactRouter } from '@ionic/react-router';
import { Route } from 'react-router-dom';
import { useState, useRef, useEffect } from 'react';
import { cloudUpload, send, medkit, analytics, logOut } from 'ionicons/icons';
import Login from './pages/Login';
import Footer from './components/Footer';
// Removed segmented views; showing a single unified metrics table
import MetricsTable from './components/MetricsTable';
import Top3PredictionsTable from './components/Top3PredictionsTable';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

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
import './theme/variables.css';
import './CompactMarkdown.css';

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
    explanation: string;
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
  // NEW: Faithfulness metrics
  comprehensiveness: { value: number; interpretation: string; level: string; explanation: string };
  sufficiency: { value: number; interpretation: string; level: string; explanation: string };
  // NEW: AUC metrics
  deletion_auc: { value: number; interpretation: string; level: string; explanation: string };
  insertion_auc: { value: number; interpretation: string; level: string; explanation: string };
  // NEW: Validation test
  randomized_weights_corr: { value: number; interpretation: string; level: string; explanation: string };
  // Educational notes
  xai_educational_notes?: {
    gradcam_explanation: string;
    saliency_explanation: string;
    lime_explanation: string;
    metrics_explanation: string;
  };
  gradcam_explanation?: string;
  lime_explanation?: string;
  metrics_explanation?: string;
  saliency_explanation?: string;

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
  response_type?: 'user' | 'system';
  id?: string;
  thinking_part?: string;
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
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [imageSize, setImageSize] = useState<number>(100);
  const [expandedThinks, setExpandedThinks] = useState<Record<number, boolean>>({});
  const [isMetricsExpanded, setIsMetricsExpanded] = useState<boolean>(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  // Segmented tabs removed; using a single unified metrics table
  const fileInputRef = useRef<HTMLInputElement>(null);

  const models = [
    { value: 'propose_balance', label: 'Proposed Model (Balanced)' },
    { value: 'propose_imbalanced', label: 'Proposed Model (Imbalanced)' },
    { value: 'ResNet50_balance', label: 'ResNet50 (Balanced)' },
    { value: 'ResNet50_imbalanced', label: 'ResNet50 (Imbalanced)' },
    { value: 'vgg16_balance', label: 'VGG16 (Balanced)' },
    { value: 'vgg16_imbalanced', label: 'VGG16 (Imbalanced)' },
    { value: 'vgg19_balance', label: 'VGG19 (Balanced)' },
    { value: 'vgg19_imbalanced', label: 'VGG19 (Imbalanced)' },
    { value: 'GoogleLeNet_balance', label: 'GoogleLeNet (Balanced)' },
    { value: 'GoogleLeNet_imbalanced', label: 'GoogleLeNet (Imbalanced)' },
    { value: 'MobileVNet_balance', label: 'MobileVNet (Balanced)' },
    { value: 'MobileVNet_imbalanced', label: 'MobileVNet (Imbalanced)' }
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
      timestamp: new Date(),
      response_type: 'user',
      id: `${Date.now()}-user`
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
      
      // Extract thinking part from system response
      let messageText = result.response;
      let thinkingPart = '';
      
      const thinkStartIndex = messageText.indexOf('<think>');
      const thinkEndIndex = messageText.indexOf('</think>');
      
      if (thinkStartIndex !== -1 && thinkEndIndex !== -1 && thinkEndIndex > thinkStartIndex) {
        thinkingPart = messageText.substring(thinkStartIndex + 7, thinkEndIndex).trim();
        messageText = (messageText.substring(0, thinkStartIndex) + messageText.substring(thinkEndIndex + 8)).trim();
      }
      
      const aiMessage: ChatMessage = {
        text: messageText,
        isUser: false,
        timestamp: new Date(),
        response_type: 'system',
        id: `${Date.now()}-system`,
        thinking_part: thinkingPart
      };

      setChatMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error sending chat message:', error);
      const errorMessage: ChatMessage = {
        text: 'Sorry, I encountered an error. Please try again.',
        isUser: false,
        timestamp: new Date(),
        response_type: 'system',
        id: `${Date.now()}-error`
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsChatLoading(false);
    }
  };

  useEffect(() => {
    console.log('chatMessages', chatMessages);
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [chatMessages, isChatLoading]);

  useEffect(() => {
    if (prediction && prediction.final_report) {
      let reportContent = prediction.final_report;
      const msgId = `${Date.now()}-initial`;
      
      // Extract thinking part from initial report
      let thinkingPart = '';
      const thinkStartIndex = reportContent.indexOf('<think>');
      const thinkEndIndex = reportContent.indexOf('</think>');
      
      if (thinkStartIndex !== -1 && thinkEndIndex !== -1 && thinkEndIndex > thinkStartIndex) {
        thinkingPart = reportContent.substring(thinkStartIndex + 7, thinkEndIndex).trim();
        reportContent = (reportContent.substring(0, thinkStartIndex) + reportContent.substring(thinkEndIndex + 8)).trim();
      }
      
      const initialMessage: ChatMessage = {
        text: reportContent,
        isUser: false,
        timestamp: new Date(),
        response_type: 'system',
        id: msgId,
        thinking_part: thinkingPart
      };
      
      setChatMessages([initialMessage]);
    }
  }, [prediction]);

  const getConfidenceColor = (level: string) => {
    switch (level) {
      case 'good': return 'success';
      case 'warning': return 'warning';
      case 'bad': return 'danger';
      default: return 'medium';
    }
  };



  const toggleThink = (idx: number) => {
    setExpandedThinks(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Brain Tumor Support AI</IonTitle>
          <IonButtons slot="end">
            <IonButton fill="clear" onClick={onLogout}>
              <IonIcon icon={logOut} />
              <IonLabel style={{ marginLeft: '8px' }}>Logout ({user.username})</IonLabel>
            </IonButton>
          </IonButtons>
        </IonToolbar>
      </IonHeader>
      
      <IonContent fullscreen>
        <div style={{ padding: '16px', paddingBottom: '200px' }}>
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
                    
                    <div style={{ textAlign: 'center', marginBottom: '12px', marginTop: '8px' }}>
                      <p style={{ margin: '0', fontSize: '14px', color: 'var(--ion-color-primary)', fontWeight: '600' }}>
                        Current Size: {imageSize}%
                      </p>
                    </div>
                    
                    {/* Unified responsive image array: 3 per row on desktop, horizontal scroll on mobile */}
                    {(() => {
                      const imageItems = [
                        { title: 'Original Image', src: prediction.original, alt: 'Original' },
                        // { title: 'Grad-CAM (Overlay)', src: prediction.gradcam, alt: 'Grad-CAM' },
                        { title: 'Grad-CAM Analysis', src: prediction.gradcam_analysis, alt: 'Grad-CAM Analysis' },
                        { title: 'Grad-CAM Heatmap', src: prediction.gradcam_heatmap, alt: 'Grad-CAM Heatmap' },
                        { title: 'Saliency Map', src: prediction.saliency, alt: 'Saliency' },
                        { title: 'LIME Explanation', src: prediction.lime, alt: 'LIME' },
                      ].filter(item => !!item.src);

                      // Columns: 1 for large images (full width), 2 for medium, 3 for small (space efficient)
                      const columns = imageSize >= 150 ? 1 : imageSize >= 100 ? 2 : 3;
                      const containerStyle: React.CSSProperties = {
                        display: 'grid',
                        gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
                        gap: '12px',
                        marginBottom: '5px'
                      };

                      // Mobile card basis adapts with slider; at max show full-width cards
                      const mobileBasis = imageSize >= 180 ? '100%' : imageSize >= 120 ? '80%' : '60%';

                      const imgStyle: React.CSSProperties = {
                        width: '100%',
                        height: 'auto',
                        objectFit: 'contain'
                      };

                      return (
                        <div className="image-array" style={containerStyle}>
                          {imageItems.map((item, idx) => (
                            <div className="image-card" key={`${item.title}-${idx}`} style={{ flex: `0 0 ${mobileBasis}`, maxWidth: `${imageSize}%` }}>
                              <h4 style={{ textAlign: 'center' }}>{item.title}</h4>
                              <IonImg src={item.src as string} alt={item.alt} style={imgStyle} />
                            </div>
                          ))}
                        </div>
                      );
                    })()}
                  </IonGrid>
                  
                  {/* Top 3 Predictions - First Display */}
                  <Top3PredictionsTable 
                    predictions={prediction.top3} 
                    title="Top 3 Predictions"
                  />

                  {/* Unified Metrics Table (no tabs) */}
                  <div style={{ marginTop: '16px' }}>
                    <button
                      onClick={() => setIsMetricsExpanded(!isMetricsExpanded)}
                      style={{
                        width: '100%',
                        padding: '12px',
                        backgroundColor: 'var(--ion-color-primary)',
                        color: 'white',
                        border: 'none',
                        borderRadius: '8px',
                        cursor: 'pointer',
                        fontWeight: 600,
                        fontSize: '16px',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: isMetricsExpanded ? '12px' : '0'
                      }}
                    >
                      <span>All Explainability Metrics</span>
                      <span>{isMetricsExpanded ? '▲' : '▼'}</span>
                    </button>
                    {isMetricsExpanded && (
                      <MetricsTable 
                        metrics={{
                          brier: prediction.brier,
                          entropy: prediction.entropy,
                          margin: prediction.margin,
                          mc_variance: prediction.mc_variance,
                          comprehensiveness: prediction.comprehensiveness,
                          sufficiency: prediction.sufficiency,
                          deletion_auc: prediction.deletion_auc,
                          insertion_auc: prediction.insertion_auc,
                          randomized_weights_corr: prediction.randomized_weights_corr,
                          dice: prediction.dice,
                          iou: prediction.iou
                        }}
                        title=""
                      />
                    )}
                  </div>
                </IonCardContent>
              </IonCard>

              

              {prediction.final_report && (
                <IonCard>
                  <IonCardHeader>
                    <IonCardTitle>AI Medical Assistant</IonCardTitle>
                  </IonCardHeader>
                  <IonCardContent>
                    <div
                      ref={chatContainerRef}
                      style={{
                        width: '100%',
                        overflowX: 'hidden',
                        padding: '12px',
                        backgroundColor: 'var(--ion-color-light)',
                        borderRadius: '12px',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '12px',
                        border: '1px solid var(--ion-color-primary)',
                        marginBottom: '12px'
                      }}
                    >
                      {chatMessages.map((msg, index) => {
                        const isExpanded = expandedThinks[index] ?? false;
                        
                        if (msg.isUser) {
                          // User message - right aligned
                          return (
                            <div
                              key={msg.id || index}
                              style={{
                                display: 'flex',
                                justifyContent: 'flex-end'
                              }}
                            >
                              <div
                                style={{
                                  padding: '10px 14px',
                                  borderRadius: '14px',
                                  backgroundColor: '#f8d7e2',
                                  color: 'var(--ion-color-dark)',
                                  maxWidth: '80%',
                                  boxShadow: '0 2px 6px rgba(0,0,0,0.08)',
                                  whiteSpace: 'pre-wrap',
                                  wordBreak: 'break-word',
                                  border: '1px solid #f08cb3'
                                }}
                              >
                                {msg.text}
                              </div>
                            </div>
                          );
                        }
                        
                        // System message - left aligned with thinking section
                        return (
                          <div
                            key={msg.id || index}
                            style={{
                              display: 'flex',
                              justifyContent: 'flex-start',
                              width: '100%'
                            }}
                          >
                            <div style={{ maxWidth: '80%', width: '100%' }}>
                              {msg.thinking_part && (
                                <div style={{ marginBottom: '8px' }}>
                                  <button
                                    onClick={() => toggleThink(index)}
                                    style={{
                                      width: '100%',
                                      padding: '8px',
                                      backgroundColor: 'var(--ion-color-primary)',
                                      color: 'white',
                                      border: 'none',
                                      borderRadius: '8px',
                                      cursor: 'pointer',
                                      fontWeight: 600,
                                      display: 'flex',
                                      justifyContent: 'space-between',
                                      alignItems: 'center'
                                    }}
                                  >
                                    <span>AI Thinking Process</span>
                                    <span>{isExpanded ? '▲' : '▼'}</span>
                                  </button>
                                  {isExpanded && (
                                    <div style={{ marginTop: '8px', padding: '10px', backgroundColor: '#eef3ff', borderRadius: '8px', border: '1px solid #d0ddff', whiteSpace: 'pre-wrap' }}>
                                      {msg.thinking_part}
                                    </div>
                                  )}
                                </div>
                              )}
                              <div
                                className="compact-markdown"
                                style={{
                                  padding: '10px 14px',
                                  borderRadius: '14px',
                                  backgroundColor: '#d7f8d7',
                                  color: 'var(--ion-color-dark)',
                                  boxShadow: '0 2px 6px rgba(0,0,0,0.08)',
                                  wordBreak: 'break-word',
                                  border: '1px solid #63c77a'
                                }}
                              >
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                      {isChatLoading && (
                        <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
                          <div
                            style={{
                              padding: '10px 14px',
                              borderRadius: '14px',
                              backgroundColor: '#d7f8d7',
                              color: 'var(--ion-color-dark)',
                              boxShadow: '0 2px 6px rgba(0,0,0,0.08)',
                              border: '1px solid #63c77a'
                            }}
                          >
                            AI is typing...
                          </div>
                        </div>
                      )}
                    </div>

                    <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end' }}>
                      <IonTextarea
                        value={chatInput}
                        onIonInput={(e) => setChatInput(e.detail.value!)}
                        placeholder="Ask anything..."
                        rows={3}
                        style={{ flex: 1, minHeight: '80px', backgroundColor: '#bdb1b5ff', border: '1px solid #f08cb3', borderRadius: '10px' }}
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
                        style={{ alignSelf: 'stretch', minWidth: '56px', backgroundColor: '#f08cb3' }}
                      >
                        <IonIcon icon={send} />
                      </IonButton>
                    </div>
                  </IonCardContent>
                </IonCard>

              )}

            </>
          )}
        </div>
        <IonLoading isOpen={isLoading} message="Analyzing image..." />
        
        <IonAlert
          isOpen={showAlert}
          onDidDismiss={() => setShowAlert(false)}
          header="Alert"
          message={alertMessage}
          buttons={['OK']}
        />
      </IonContent>
      <IonFooter>
        <div style={{ padding: '4px', margin: '0' }}>
          <Footer />
        </div>
      </IonFooter>
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
