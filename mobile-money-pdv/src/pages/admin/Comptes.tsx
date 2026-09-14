import { useCallback, useEffect, useState } from 'react';
import { useAuth } from '../../auth/AuthContext';
import { Entete } from '../../components/Entete';
import { supabase } from '../../lib/supabase';

interface Agent {
  id: string;
  nom_complet: string;
  actif: boolean;
}

export function Comptes() {
  const { pdvs } = useAuth();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [erreur, setErreur] = useState<string | null>(null);
  const [enCours, setEnCours] = useState(false);

  const [nom, setNom] = useState('');
  const [identifiant, setIdentifiant] = useState('');
  const [motDePasse, setMotDePasse] = useState('');
  const [pdvIds, setPdvIds] = useState<string[]>([]);

  const charger = useCallback(async () => {
    const { data } = await supabase
      .from('utilisateurs')
      .select('id, nom_complet, actif')
      .eq('role', 'AGENT')
      .order('nom_complet');
    setAgents(data ?? []);
  }, []);

  useEffect(() => {
    void charger();
  }, [charger]);

  async function creer() {
    setErreur(null);
    setMessage(null);
    if (!nom.trim() || !identifiant.trim() || motDePasse.length < 8) {
      setErreur('Nom, identifiant et mot de passe (8 caractères min.) obligatoires.');
      return;
    }
    setEnCours(true);
    const { data, error } = await supabase.functions.invoke('creer-agent', {
      body: {
        identifiant: identifiant.trim(),
        mot_de_passe: motDePasse,
        nom_complet: nom.trim(),
        pdv_ids: pdvIds,
      },
    });
    setEnCours(false);
    if (error || (data && (data as { erreur?: string }).erreur)) {
      setErreur((data as { erreur?: string })?.erreur ?? error?.message ?? 'Création impossible.');
      return;
    }
    setMessage(`Agent « ${nom} » créé.`);
    setNom('');
    setIdentifiant('');
    setMotDePasse('');
    setPdvIds([]);
    void charger();
  }

  async function basculerActif(a: Agent) {
    await supabase.from('utilisateurs').update({ actif: !a.actif }).eq('id', a.id);
    void charger();
  }

  return (
    <div className="min-h-screen">
      <Entete titre="Comptes agents" />
      <main className="mx-auto max-w-lg space-y-4 px-4 py-4">
        <section className="carte space-y-3">
          <h2 className="font-semibold">Créer un agent</h2>
          <input className="champ-texte" placeholder="Nom complet" value={nom} onChange={(e) => setNom(e.target.value)} />
          <input
            className="champ-texte"
            autoCapitalize="none"
            placeholder="Identifiant"
            value={identifiant}
            onChange={(e) => setIdentifiant(e.target.value)}
          />
          <input
            type="password"
            className="champ-texte"
            placeholder="Mot de passe (8 car. min.)"
            value={motDePasse}
            onChange={(e) => setMotDePasse(e.target.value)}
          />
          <div>
            <span className="mb-1 block text-sm font-medium text-gray-600">Points de vente affectés</span>
            <div className="space-y-1">
              {pdvs.map((p) => (
                <label key={p.id} className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={pdvIds.includes(p.id)}
                    onChange={(e) =>
                      setPdvIds((prev) =>
                        e.target.checked ? [...prev, p.id] : prev.filter((x) => x !== p.id),
                      )
                    }
                  />
                  {p.nom}
                </label>
              ))}
            </div>
          </div>
          {erreur && <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">{erreur}</p>}
          {message && <p className="rounded-lg bg-emerald-50 px-3 py-2 text-sm text-emerald-800">{message}</p>}
          <button className="btn-principal" onClick={() => void creer()} disabled={enCours}>
            {enCours ? 'Création…' : 'Créer le compte'}
          </button>
        </section>

        <section className="carte space-y-2">
          <h2 className="font-semibold">Agents</h2>
          {agents.length === 0 && <p className="text-sm text-gray-500">Aucun agent.</p>}
          {agents.map((a) => (
            <div key={a.id} className="flex items-center justify-between border-b border-gray-100 py-2 last:border-0">
              <div>
                <p className="font-medium">{a.nom_complet}</p>
                <p className="text-xs text-gray-500">{a.actif ? 'Actif' : 'Désactivé'}</p>
              </div>
              <button
                className={`rounded-lg px-3 py-1 text-sm ${a.actif ? 'text-red-600' : 'text-marque'}`}
                onClick={() => void basculerActif(a)}
              >
                {a.actif ? 'Désactiver' : 'Réactiver'}
              </button>
            </div>
          ))}
        </section>
      </main>
    </div>
  );
}
