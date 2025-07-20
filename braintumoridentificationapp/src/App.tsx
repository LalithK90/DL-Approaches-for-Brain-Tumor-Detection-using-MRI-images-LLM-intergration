import {
  IonApp,
  IonContent,
  IonPage,
  IonRouterOutlet,
  setupIonicReact,
  useIonViewWillEnter,
} from '@ionic/react';
import { IonReactRouter } from '@ionic/react-router';
import { Route } from 'react-router-dom';
import { useRef } from 'react';

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

// It's a good practice to store configuration values like URLs in a central place
// or in environment variables (e.g., using .env files).
const FLASK_APP_URL = "http://127.0.0.1:5000/";

const BrowserPage = () => {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useIonViewWillEnter(() => {
    // This logic forces the iframe to reload its content every time the page is viewed.
    // In an SPA, the component might stay in the DOM, and without this, the iframe
    // could show stale content when navigating back to this page.
    if (iframeRef.current) {
      iframeRef.current.src = "about:blank";
      setTimeout(() => {
        if (iframeRef.current) {
          iframeRef.current.src = FLASK_APP_URL;
        }
      }, 10);
    }
  });

  return (
    <IonPage>
      <IonContent fullscreen>
        <iframe
          ref={iframeRef}
          src={FLASK_APP_URL}
          style={{ width: '100%', height: '100%', border: 'none' }}
        />
      </IonContent>
    </IonPage>
  );
};

const App = () => {
  return (
    <IonApp>
      <IonReactRouter>
        <IonRouterOutlet>
          <Route path="/" exact={true}>
            <BrowserPage />
          </Route>
        </IonRouterOutlet>
      </IonReactRouter>
    </IonApp>
  );
};

export default App;
