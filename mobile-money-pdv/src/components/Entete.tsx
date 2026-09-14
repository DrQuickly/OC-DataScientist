import type { ReactNode } from 'react';
import { useAuth } from '../auth/AuthContext';
import { IndicateurSync } from './IndicateurSync';

/** En-tête commun : titre, indicateur de synchro, déconnexion. */
export function Entete({ titre, action }: { titre: string; action?: ReactNode }) {
  const { profil, deconnexion } = useAuth();
  return (
    <header className="sticky top-0 z-10 border-b border-gray-200 bg-white">
      <div className="mx-auto flex max-w-lg items-center justify-between gap-2 px-4 py-3">
        <div className="min-w-0">
          <h1 className="truncate text-lg font-bold text-marque">{titre}</h1>
          {profil && <p className="truncate text-xs text-gray-500">{profil.nom_complet}</p>}
        </div>
        <div className="flex items-center gap-2">
          <IndicateurSync />
          {action}
          <button
            type="button"
            onClick={() => void deconnexion()}
            className="rounded-lg px-2 py-1 text-sm text-gray-500"
          >
            Quitter
          </button>
        </div>
      </div>
    </header>
  );
}
