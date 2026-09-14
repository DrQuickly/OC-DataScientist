import { BrowserRouter, Navigate, Route, Routes, NavLink } from 'react-router-dom';
import { FournisseurAuth, useAuth } from './auth/AuthContext';
import { Connexion } from './pages/Connexion';
import { Cloture } from './pages/agent/Cloture';
import { CloturesDuJour } from './pages/admin/CloturesDuJour';
import { Comptes } from './pages/admin/Comptes';

function Routage() {
  const { session, profil, chargement } = useAuth();

  if (!session) return <Connexion />;
  if (chargement) return <Chargement />;
  if (!profil) return <CompteInvalide />;

  if (profil.role === 'ADMIN') {
    return (
      <div className="pb-16">
        <Routes>
          <Route path="/admin/clotures" element={<CloturesDuJour />} />
          <Route path="/admin/comptes" element={<Comptes />} />
          <Route path="*" element={<Navigate to="/admin/clotures" replace />} />
        </Routes>
        <NavAdmin />
      </div>
    );
  }

  // AGENT
  return (
    <Routes>
      <Route path="/cloture" element={<Cloture />} />
      <Route path="*" element={<Navigate to="/cloture" replace />} />
    </Routes>
  );
}

function NavAdmin() {
  const lien = 'flex-1 py-3 text-center text-sm font-medium';
  const actif = ({ isActive }: { isActive: boolean }) =>
    `${lien} ${isActive ? 'text-marque' : 'text-gray-500'}`;
  return (
    <nav className="fixed inset-x-0 bottom-0 z-10 flex border-t border-gray-200 bg-white">
      <NavLink to="/admin/clotures" className={actif}>
        Clôtures
      </NavLink>
      <NavLink to="/admin/comptes" className={actif}>
        Comptes
      </NavLink>
    </nav>
  );
}

function Chargement() {
  return <div className="flex min-h-screen items-center justify-center text-gray-500">Chargement…</div>;
}

function CompteInvalide() {
  const { deconnexion } = useAuth();
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 px-4 text-center">
      <p className="text-gray-700">Ce compte n’a pas de profil actif. Contactez l’administrateur.</p>
      <button className="btn-secondaire max-w-xs" onClick={() => void deconnexion()}>
        Se déconnecter
      </button>
    </div>
  );
}

export function App() {
  return (
    <FournisseurAuth>
      <BrowserRouter>
        <Routage />
      </BrowserRouter>
    </FournisseurAuth>
  );
}
