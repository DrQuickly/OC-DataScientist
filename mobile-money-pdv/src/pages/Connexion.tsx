import { useState } from 'react';
import { useAuth } from '../auth/AuthContext';

export function Connexion() {
  const { connexion } = useAuth();
  const [identifiant, setIdentifiant] = useState('');
  const [motDePasse, setMotDePasse] = useState('');
  const [erreur, setErreur] = useState<string | null>(null);
  const [enCours, setEnCours] = useState(false);

  async function soumettre(e: React.FormEvent) {
    e.preventDefault();
    setErreur(null);
    setEnCours(true);
    const { erreur } = await connexion(identifiant.trim(), motDePasse);
    setEnCours(false);
    if (erreur) setErreur(erreur);
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <form onSubmit={soumettre} className="carte w-full max-w-sm space-y-4">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-marque">Gestion PDV</h1>
          <p className="text-sm text-gray-500">Mobile Money</p>
        </div>

        <label className="block">
          <span className="mb-1 block text-sm font-medium text-gray-600">Identifiant</span>
          <input
            className="champ-texte"
            autoCapitalize="none"
            autoCorrect="off"
            value={identifiant}
            onChange={(e) => setIdentifiant(e.target.value)}
            placeholder="votre identifiant"
          />
        </label>

        <label className="block">
          <span className="mb-1 block text-sm font-medium text-gray-600">Mot de passe</span>
          <input
            type="password"
            className="champ-texte"
            value={motDePasse}
            onChange={(e) => setMotDePasse(e.target.value)}
          />
        </label>

        {erreur && <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">{erreur}</p>}

        <button type="submit" className="btn-principal" disabled={enCours || !identifiant || !motDePasse}>
          {enCours ? 'Connexion…' : 'Se connecter'}
        </button>
      </form>
    </div>
  );
}
