import React, { useState, useEffect } from 'react';
import {
  IonContent,
  IonHeader,
  IonPage,
  IonTitle,
  IonToolbar,
  IonCard,
  IonCardContent,
  IonItem,
  IonLabel,
  IonInput,
  IonButton,
  IonAlert,
  IonIcon,
  IonText,
  IonSpinner
} from '@ionic/react';
import { lockClosedOutline, personOutline } from 'ionicons/icons';
import './Login.css';
import appInfo from '../config/appInfo';

interface LoginProps {
  onLoginSuccess: (user: { id: string; username: string; roles: string[] }) => void;
}

interface LoginResponse {
  success: boolean;
  message: string;
  user?: {
    id: string;
    username: string;
    roles: string[];
  };
}

const API_BASE_URL = import.meta.env.DEV ? '' : (import.meta.env.VITE_API_BASE_URL || '');

const Login: React.FC<LoginProps> = ({ onLoginSuccess }) => {
  const [username, setUsername] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [showAlert, setShowAlert] = useState<boolean>(false);
  const [alertMessage, setAlertMessage] = useState<string>('');
  const [isDatasetOpen, setIsDatasetOpen] = useState<boolean>(false);
  const [datasetText, setDatasetText] = useState<string>('');

  const handleLogin = async () => {
    if (!username.trim() || !password.trim()) {
      setAlertMessage('Please enter both username and password.');
      setShowAlert(true);
      return;
    }

    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/login/api`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, password }),
        credentials: 'include'
      });

      // Check if response is OK before parsing JSON
      if (!response.ok) {
        const responseText = await response.text();
        console.error('Login API error:', response.status, response.statusText, responseText);
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      // Check content type to ensure it's JSON
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const responseText = await response.text();
        console.error('Non-JSON response:', responseText);
        throw new Error('Server returned non-JSON response');
      }

      const result: LoginResponse = await response.json();

      if (result.success && result.user) {
        onLoginSuccess(result.user);
      } else {
        setAlertMessage(result.message || 'Login failed. Please try again.');
        setShowAlert(true);
      }
    } catch (error: unknown) {
      console.error('Login error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Network error. Please check your connection.';
      setAlertMessage(errorMessage);
      setShowAlert(true);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      handleLogin();
    }
  };

  const openDatasetModal = async () => {
    try {
      const res = await fetch('/dataset_details.txt');
      const text = await res.text();
      setDatasetText(text);
    } catch (e) {
      setDatasetText('Unable to load dataset details.');
    }
    setIsDatasetOpen(true);
  };

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar color="primary">
          <IonTitle>Brain Tumor Support AI</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent className="login-content">
        <div className="login-layout">
          {/* Left: Login form */}
          <div className="login-left">
            <div className="login-container">
              <div className="login-header">
                <IonText>
                  <p>Brain Tumor Support AI</p>
                </IonText>
              </div>

              <IonCard className="login-card">
                <IonCardContent>
                  <IonItem className="login-item">
                    <IonIcon icon={personOutline} slot="start" />
                    <IonLabel position="stacked">Username</IonLabel>
                    <IonInput
                      type="text"
                      value={username}
                      onIonInput={(e) => setUsername(e.detail.value!)}
                      onKeyPress={handleKeyPress}
                      placeholder="Enter your username"
                      disabled={isLoading}
                    />
                  </IonItem>

                  <IonItem className="login-item">
                    <IonIcon icon={lockClosedOutline} slot="start" />
                    <IonLabel position="stacked">Password</IonLabel>
                    <IonInput
                      type="password"
                      value={password}
                      onIonInput={(e) => setPassword(e.detail.value!)}
                      onKeyPress={handleKeyPress}
                      placeholder="Enter your password"
                      disabled={isLoading}
                    />
                  </IonItem>

                  <IonButton
                    expand="block"
                    onClick={handleLogin}
                    disabled={isLoading}
                    className="login-button"
                    color="primary"
                  >
                    {isLoading ? (
                      <>
                        <IonSpinner name="crescent" />
                        &nbsp; Logging in...
                      </>
                    ) : (
                      'Login'
                    )}
                  </IonButton>

                  <div className="demo-credentials">
                    <IonText color="medium">
                      <h3>Demo Credentials:</h3>
                      <p><strong>Admin:</strong> admin / admin</p>
                      <p><strong>Doctor:</strong> doctor / doctor</p>
                      <p><strong>Radiologist:</strong> radiologist / radiologist</p>
                    </IonText>
                  </div>
                  <div className="p-5"></div>
                </IonCardContent>
              </IonCard>
            </div>
          </div>

          {/* Right: Research information panel */}
          <div className="login-right">
            <IonCard className="research-card">
              <IonCardContent>
                <IonText>
                  <h2 className="research-title">{appInfo.researchTitle}</h2>
                  <p className="research-institute">{appInfo.institute}</p>
                  <p style={{ marginTop: 4, marginBottom: 4, fontWeight: 600 }}>{appInfo.thesisPresentedByLabel}</p>
                  <p><strong>Researcher:</strong> {appInfo.researcherName}</p>
                  <p><strong>Registration No:</strong> {appInfo.registrationNumber}</p>
                  <p><strong>Board of Study:</strong> {appInfo.boardOfStudy}</p>
                  <div className="research-supervisors">
                    <strong>Supervisor(s):</strong>
                    <ul>
                      {appInfo.supervisors.map((name, idx) => (
                        <li key={idx}>{name}</li>
                      ))}
                    </ul>
                  </div>
                  <p><strong>Degree:</strong> {appInfo.degreeName}</p>
                  <p><strong>University:</strong> {appInfo.university}</p>
                  <p><strong>Country:</strong> {appInfo.country}</p>
                  <p><strong>Year:</strong> {appInfo.year}</p>
                  <p className="research-disclaimer">{appInfo.disclaimer}</p>
                  <p className="research-note">{appInfo.note}</p>
                  <p style={{ marginTop: 12 }}>
                    <strong>Dataset:</strong> {appInfo.dataset.name} ({appInfo.dataset.source}) —
                    <a href={appInfo.dataset.url} target="_blank" rel="noopener noreferrer" style={{ marginLeft: 6 }}>
                      View on Kaggle
                    </a>
                  </p>
                  <p style={{ fontSize: '12px', color: 'var(--ion-color-medium)' }}>{appInfo.dataset.licenseNote}</p>
                </IonText>
               
              </IonCardContent>
            </IonCard>
          </div>
        </div>

        <IonAlert
          isOpen={showAlert}
          onDidDismiss={() => setShowAlert(false)}
          header="Login Error"
          message={alertMessage}
          buttons={['OK']}
        />



      </IonContent>

    </IonPage>
  );
};

export default Login;