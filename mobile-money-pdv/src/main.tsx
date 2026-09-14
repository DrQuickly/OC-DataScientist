import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { demarrerSyncAuto } from './lib/sync';
import './index.css';

// Démarre la synchronisation automatique (retour réseau + filet périodique).
demarrerSyncAuto();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
