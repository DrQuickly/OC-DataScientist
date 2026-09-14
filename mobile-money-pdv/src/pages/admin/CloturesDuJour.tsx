import { useCallback, useEffect, useState } from 'react';
import { useAuth } from '../../auth/AuthContext';
import { Entete } from '../../components/Entete';
import { supabase } from '../../lib/supabase';
import { formaterFcfa, dateMetierAujourdhui, formaterDate } from '../../lib/format';
import type { StatutCloture } from '../../domain/types';

interface Ligne {
  id: string;
  pdv_nom: string;
  agent_nom: string;
  statut: StatutCloture;
  especes_fin_attendue: number;
  especes_fin_constatee: number;
  ecart: number;
  photo_url: string | null;
}

export function CloturesDuJour() {
  const { profil } = useAuth();
  const [date, setDate] = useState(dateMetierAujourdhui());
  const [lignes, setLignes] = useState<Ligne[]>([]);
  const [chargement, setChargement] = useState(true);
  const [erreur, setErreur] = useState<string | null>(null);

  const charger = useCallback(async () => {
    setChargement(true);
    setErreur(null);
    try {
      const { data: clotures, error } = await supabase
        .from('clotures')
        .select('id, pdv_id, agent_id, statut, especes_fin_constatee, photo_url, pdv(nom), utilisateurs:agent_id(nom_complet)')
        .eq('date_cloture', date);
      if (error) throw error;

      const ids = (clotures ?? []).map((c) => c.id);
      const { data: calculs } = await supabase
        .from('v_cloture_calculs')
        .select('cloture_id, especes_fin_attendue, ecart')
        .in('cloture_id', ids.length ? ids : ['00000000-0000-0000-0000-000000000000']);

      const parId = new Map((calculs ?? []).map((c) => [c.cloture_id, c]));
      setLignes(
        (clotures ?? []).map((c) => {
          const calc = parId.get(c.id);
          // Les relations imbriquées peuvent être objet ou tableau selon la config PostgREST.
          const pdv = Array.isArray(c.pdv) ? c.pdv[0] : c.pdv;
          const ag = Array.isArray(c.utilisateurs) ? c.utilisateurs[0] : c.utilisateurs;
          return {
            id: c.id,
            pdv_nom: pdv?.nom ?? '—',
            agent_nom: ag?.nom_complet ?? '—',
            statut: c.statut,
            especes_fin_constatee: c.especes_fin_constatee,
            especes_fin_attendue: calc?.especes_fin_attendue ?? 0,
            ecart: calc?.ecart ?? 0,
            photo_url: c.photo_url,
          };
        }),
      );
    } catch (e) {
      setErreur(e instanceof Error ? e.message : 'Chargement impossible (réseau ?)');
    } finally {
      setChargement(false);
    }
  }, [date]);

  useEffect(() => {
    void charger();
  }, [charger]);

  async function statuer(id: string, statut: 'VALIDEE' | 'CONTESTEE') {
    if (!profil) return;
    const { error } = await supabase
      .from('clotures')
      .update({ statut, validee_par: profil.id })
      .eq('id', id);
    if (error) setErreur(error.message);
    else void charger();
  }

  return (
    <div className="min-h-screen">
      <Entete titre="Clôtures" />
      <main className="mx-auto max-w-lg space-y-4 px-4 py-4">
        <label className="block">
          <span className="mb-1 block text-sm font-medium text-gray-600">Date</span>
          <input type="date" className="champ-texte" value={date} onChange={(e) => setDate(e.target.value)} />
        </label>
        <p className="text-sm text-gray-500">{formaterDate(date)}</p>

        {erreur && <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">{erreur}</p>}
        {chargement && <p className="text-sm text-gray-500">Chargement…</p>}
        {!chargement && lignes.length === 0 && (
          <p className="text-sm text-gray-500">Aucune clôture pour cette date.</p>
        )}

        {lignes.map((l) => (
          <div key={l.id} className="carte space-y-2">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-semibold">{l.pdv_nom}</p>
                <p className="text-xs text-gray-500">{l.agent_nom}</p>
              </div>
              <BadgeStatut statut={l.statut} />
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Attendu</span>
              <span className="tabular-nums">{formaterFcfa(l.especes_fin_attendue)}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Compté</span>
              <span className="tabular-nums">{formaterFcfa(l.especes_fin_constatee)}</span>
            </div>
            <div
              className={`flex items-center justify-between rounded-lg px-2 py-1 text-sm font-semibold ${
                l.ecart === 0 ? 'bg-emerald-100 text-emerald-800' : 'bg-red-100 text-red-800'
              }`}
            >
              <span>Écart</span>
              <span className="tabular-nums">{formaterFcfa(l.ecart)}</span>
            </div>
            {l.photo_url && <LienPhoto chemin={l.photo_url} />}
            {l.statut === 'SOUMISE' && (
              <div className="flex gap-2">
                <button className="btn-principal" onClick={() => void statuer(l.id, 'VALIDEE')}>
                  Valider
                </button>
                <button className="btn-secondaire" onClick={() => void statuer(l.id, 'CONTESTEE')}>
                  Contester
                </button>
              </div>
            )}
          </div>
        ))}
      </main>
    </div>
  );
}

/** Le bucket photos est privé : on génère une URL signée à la demande. */
function LienPhoto({ chemin }: { chemin: string }) {
  const [chargement, setChargement] = useState(false);
  async function ouvrir() {
    setChargement(true);
    const { data } = await supabase.storage.from('photos-clotures').createSignedUrl(chemin, 60);
    setChargement(false);
    if (data?.signedUrl) window.open(data.signedUrl, '_blank', 'noreferrer');
  }
  return (
    <button className="text-sm text-marque underline" onClick={() => void ouvrir()} disabled={chargement}>
      {chargement ? 'Ouverture…' : 'Voir la photo du terminal'}
    </button>
  );
}

function BadgeStatut({ statut }: { statut: StatutCloture }) {
  const styles: Record<StatutCloture, string> = {
    BROUILLON: 'bg-gray-100 text-gray-700',
    SOUMISE: 'bg-blue-100 text-blue-800',
    VALIDEE: 'bg-emerald-100 text-emerald-800',
    CONTESTEE: 'bg-red-100 text-red-800',
  };
  return <span className={`rounded-full px-2 py-1 text-xs font-medium ${styles[statut]}`}>{statut}</span>;
}
