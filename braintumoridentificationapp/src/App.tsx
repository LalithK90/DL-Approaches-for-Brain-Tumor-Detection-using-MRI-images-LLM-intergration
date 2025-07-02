import {
  IonApp,
  IonContent,
  IonHeader,
  IonPage,
  IonRouterOutlet,
  IonTitle,
  IonToolbar,
  setupIonicReact,
} from '@ionic/react';
import { IonReactRouter } from '@ionic/react-router';
import { Route } from 'react-router-dom';

/* Core CSS required for Ionic components to work properly */
import '@ionic/react/css/core.css';

/* Basic CSS for apps built with Ionic */
import '@ionic/react/css/normalize.css';
import '@ionic/react/css/structure.css';
import '@ionic/react/css/typography.css';

/* Optional CSS utils that can be commented out */
import '@ionic/react/css/padding.css';
import '@ionic/react/css/float-elements.css';
import '@ionic/react/css/text-alignment.css';
import '@ionic/react/css/text-transformation.css';
import '@ionic/react/css/flex-utils.css';
import '@ionic/react/css/display.css';

/**
 * Ionic Dark Mode
 * -----------------------------------------------------
 * For more info, please see:
 * https://ionicframework.com/docs/theming/dark-mode
 */

/* import '@ionic/react/css/palettes/dark.always.css'; */
/* import '@ionic/react/css/palettes/dark.class.css'; */
import '@ionic/react/css/palettes/dark.system.css';

/* Theme variables */
import './theme/variables.css';

setupIonicReact();

const BrowserPage: React.FC = () => {
  return (
    <IonPage>
    {/* //   <IonHeader>
    //     <IonToolbar>
    //       <IonTitle>Localhost Browser</IonTitle>
    //     </IonToolbar>
    //   </IonHeader> */}
      <IonContent fullscreen>
        <iframe
          src="http://127.0.0.1:5000/"
          style={{ width: '100%', height: '100%', border: 'none' }}
        />
      </IonContent>
    </IonPage>
  );
};

const App: React.FC = () => {
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
