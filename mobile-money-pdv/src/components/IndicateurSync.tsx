import { useEffect, useState } from 'react';
import { nombreEnAttente } from '../lib/db';
import { surSync, synchroniser } from '../lib/sync';

/**
 * Indicateur visible de l'état de synchronisation :
 *   « Synchronisé » (vert) ou « En attente (n) » (orange).
 * Affiche aussi l'état hors ligne.
 */
export function IndicateurSync() {
  const [enAttente, setEnAttente] = useState(0);
  const [enCours, setEnCours] = useState(false);
  const [enLigne, setEnLigne] = useState(navigator.onLine);

  useEffect(() => {
    nombreEnAttente().then(setEnAttente);
    const off = surSync(({ enCours, enAttente }) => {
      setEnCours(enCours);
      setEnAttente(enAttente);
    });
    const maj = () => setEnLigne(navigator.onLine);
    window.addEventListener('online', maj);
    window.addEventListener('offline', maj);
    return () => {
      off();
      window.removeEventListener('online', maj);
      window.removeEventListener('offline', maj);
    };
  }, []);

  let libelle: string;
  let couleur: string;
  if (!enLigne) {
    libelle = 'Hors ligne';
    couleur = 'bg-gray-200 text-gray-700';
  } else if (enCours) {
    libelle = 'Synchronisation…';
    couleur = 'bg-amber-100 text-amber-800';
  } else if (enAttente > 0) {
    libelle = `En attente (${enAttente})`;
    couleur = 'bg-amber-100 text-amber-800';
  } else {
    libelle = 'Synchronisé';
    couleur = 'bg-emerald-100 text-emerald-800';
  }

  return (
    <button
      type="button"
      onClick={() => void synchroniser()}
      className={`rounded-full px-3 py-1 text-sm font-medium ${couleur}`}
      title="Toucher pour synchroniser"
    >
      <span aria-live="polite">{libelle}</span>
    </button>
  );
}
