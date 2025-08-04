import React, { useState } from 'react';
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
import { medicalOutline, lockClosedOutline, personOutline } from 'ionicons/icons';
import Footer from '../components/Footer';
import './Login.css';

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

const Login: React.FC<LoginProps> = ({ onLoginSuccess }) => {
  const [username, setUsername] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [showAlert, setShowAlert] = useState<boolean>(false);
  const [alertMessage, setAlertMessage] = useState<string>('');

  const handleLogin = async () => {
    if (!username.trim() || !password.trim()) {
      setAlertMessage('Please enter both username and password.');
      setShowAlert(true);
      return;
    }

    setIsLoading(true);

    try {
      // First, check if we're using demo credentials for local testing
      // if ((username === 'admin' && password === 'admin') ||
      //     (username === 'doctor' && password === 'doctor') ||
      //     (username === 'radiologist' && password === 'radiologist')) {

      //   // Simulate successful login with demo credentials
      //   setTimeout(() => {
      //     const demoUser = {
      //       id: '123',
      //       username: username,
      //       roles: [username] // Use username as role for demo
      //     };
      //     onLoginSuccess(demoUser);
      //   }, 1000); // Simulate network delay

      //   return;
      // }

      // Otherwise, proceed with actual API call
      const response = await fetch('http://127.0.0.1:5000/login/api', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, password }),
        credentials: 'include'
      });

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

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar color="primary">
          <IonTitle>Medical Image XAI</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent className="login-content">
        <div className="login-container">
          <div className="login-header">
            <IonText>
              <p>Brain Tumor Identification System</p>
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