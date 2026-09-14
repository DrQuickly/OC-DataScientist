import { useEffect, useMemo, useState } from 'react';
import { v4 as uuid } from 'uuid';
import { useAuth } from '../../auth/AuthContext';
import { Entete } from '../../components/Entete';
import { ChampMontant } from '../../components/ChampMontant';
import { calculerCloture, TOLERANCE_DEFAUT } from '../../domain/calculs';
import type { EntreesCloture, SoldeReseau } from '../../domain/types';
import { formaterFcfa, dateMetierAujourdhui, formaterDate } from '../../lib/format';
import { chargerReseaux, chargerOuvertureParDefaut, type Reseau } from '../../lib/donnees';
import { db, type CreanceLocale, type DepenseLocale } from '../../lib/db';
import { compresserPhoto } from '../../lib/photo';
import { synchroniser } from '../../lib/sync';

interface EtatSolde {
  reseau_id: string;
  uv_debut: number;
  approvisionnement: number;
  uv_fin: number;
}

export function Cloture() {
  const { profil, pdvs } = useAuth();
  const [pdvId, setPdvId] = useState('');
  const [reseaux, setReseaux] = useState<Reseau[]>([]);
  const dateCloture = dateMetierAujourdhui();

  const [soldes, setSoldes] = useState<EtatSolde[]>([]);
  const [ouvertureRef, setOuvertureRef] = useState<{ especes: number; uv: Record<string, number> }>({
    especes: 0,
    uv: {},
  });
  const [especesDebut, setEspecesDebut] = useState(0);
  const [especesFin, setEspecesFin] = useState(0);
  const [apports, setApports] = useState(0);
  const [sorties, setSorties] = useState(0);
  const [corrigerOuverture, setCorrigerOuverture] = useState(false);
  const [motif, setMotif] = useState('');

  const [depenses, setDepenses] = useState<DepenseLocale[]>([]);
  const [creances, setCreances] = useState<CreanceLocale[]>([]);

  const [photoBlob, setPhotoBlob] = useState<Blob | null>(null);
  const [photoApercu, setPhotoApercu] = useState<string | null>(null);

  const [message, setMessage] = useState<string | null>(null);
  const [enCours, setEnCours] = useState(false);

  useEffect(() => {
    chargerReseaux().then(setReseaux);
  }, []);

  useEffect(() => {
    if (pdvs.length === 1) setPdvId(pdvs[0].id);
  }, [pdvs]);

  // Pré-remplissage de l'ouverture quand le PDV et les réseaux sont connus.
  useEffect(() => {
    if (!pdvId || reseaux.length === 0) return;
    let annule = false;
    chargerOuvertureParDefaut(pdvId, dateCloture).then((o) => {
      if (annule) return;
      setEspecesDebut(o.especes_debut);
      setOuvertureRef({ especes: o.especes_debut, uv: o.uv_debut });
      setSoldes(
        reseaux.map((r) => ({
          reseau_id: r.id,
          uv_debut: o.uv_debut[r.id] ?? 0,
          approvisionnement: 0,
          uv_fin: 0,
        })),
      );
    });
    return () => {
      annule = true;
    };
  }, [pdvId, reseaux, dateCloture]);

  const totalDepenses = depenses.reduce((a, d) => a + d.montant, 0);
  const totalCreances = creances.reduce((a, c) => a + c.montant, 0);

  const entrees: EntreesCloture = useMemo(
    () => ({
      soldes: soldes.map<SoldeReseau>((s) => ({
        reseau_id: s.reseau_id,
        uv_debut: s.uv_debut,
        approvisionnement: s.approvisionnement,
        uv_fin: s.uv_fin,
      })),
      especes_debut: especesDebut,
      especes_fin_constatee: especesFin,
      apports_especes: apports,
      sorties_especes: sorties,
      total_depenses: totalDepenses,
      total_creances_creees: totalCreances,
      total_creances_remboursees: 0, // remboursements gérés en Phase 2
    }),
    [soldes, especesDebut, especesFin, apports, sorties, totalDepenses, totalCreances],
  );

  const resultat = useMemo(() => calculerCloture(entrees, TOLERANCE_DEFAUT), [entrees]);

  const ouvertureDiverge =
    especesDebut !== ouvertureRef.especes ||
    soldes.some((s) => s.uv_debut !== (ouvertureRef.uv[s.reseau_id] ?? 0));

  function majSolde(reseau_id: string, champ: keyof EtatSolde, valeur: number) {
    setSoldes((prev) => prev.map((s) => (s.reseau_id === reseau_id ? { ...s, [champ]: valeur } : s)));
  }

  async function choisirPhoto(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    const blob = await compresserPhoto(f);
    setPhotoBlob(blob);
    setPhotoApercu(URL.createObjectURL(blob));
  }

  function erreurValidation(): string | null {
    if (!pdvId) return 'Choisissez un point de vente.';
    if (!photoBlob) return 'La photo du solde du terminal est obligatoire.';
    if (ouvertureDiverge && motif.trim().length === 0)
      return 'L’ouverture diffère de la veille : une justification écrite est obligatoire.';
    return null;
  }

  async function soumettre() {
    const err = erreurValidation();
    if (err) {
      setMessage(err);
      return;
    }
    if (!profil) return;
    setEnCours(true);
    setMessage(null);

    const clotureId = uuid();
    const maintenant = new Date().toISOString();
    try {
      await db.transaction('rw', db.clotures, db.soldes, db.depenses, db.creances, async () => {
        await db.clotures.put({
          id: clotureId,
          pdv_id: pdvId,
          agent_id: profil.id,
          date_cloture: dateCloture,
          especes_debut: especesDebut,
          especes_fin_constatee: especesFin,
          apports_especes: apports,
          sorties_especes: sorties,
          motif_derogation: ouvertureDiverge ? motif.trim() : null,
          photo_blob: photoBlob,
          photo_url: null,
          statut: 'BROUILLON',
          etat_sync: 'en_attente',
          erreur_sync: null,
          saisi_le_client: maintenant,
          maj_le: maintenant,
        });
        await db.soldes.bulkPut(
          soldes.map((s) => ({ id: uuid(), cloture_id: clotureId, ...s })),
        );
        await db.depenses.bulkPut(
          depenses.map((d) => ({ ...d, cloture_id: clotureId, pdv_id: pdvId, date: dateCloture })),
        );
        await db.creances.bulkPut(
          creances.map((c) => ({
            ...c,
            cloture_id: clotureId,
            pdv_id: pdvId,
            agent_id: profil.id,
            date_creation: dateCloture,
          })),
        );
      });

      setMessage('Clôture enregistrée. Elle sera synchronisée automatiquement.');
      // Réinitialise pour une éventuelle nouvelle saisie.
      setDepenses([]);
      setCreances([]);
      setPhotoBlob(null);
      setPhotoApercu(null);
      setEspecesFin(0);
      setApports(0);
      setSorties(0);
      void synchroniser();
    } catch (e) {
      setMessage(e instanceof Error ? e.message : 'Erreur lors de l’enregistrement.');
    } finally {
      setEnCours(false);
    }
  }

  return (
    <div className="min-h-screen pb-28">
      <Entete titre="Clôture du jour" />
      <main className="mx-auto max-w-lg space-y-4 px-4 py-4">
        <p className="text-sm text-gray-500">Date : {formaterDate(dateCloture)}</p>

        {pdvs.length > 1 && (
          <label className="block">
            <span className="mb-1 block text-sm font-medium text-gray-600">Point de vente</span>
            <select className="champ-texte" value={pdvId} onChange={(e) => setPdvId(e.target.value)}>
              <option value="">— choisir —</option>
              {pdvs.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.nom}
                </option>
              ))}
            </select>
          </label>
        )}

        {/* Soldes UV par réseau */}
        <section className="carte space-y-4">
          <h2 className="font-semibold">Soldes UV par réseau</h2>
          {soldes.map((s) => {
            const r = reseaux.find((x) => x.id === s.reseau_id);
            return (
              <div key={s.reseau_id} className="rounded-xl bg-gray-50 p-3">
                <p className="mb-2 font-medium">{r?.nom ?? 'Réseau'}</p>
                <div className="grid grid-cols-3 gap-2">
                  <div>
                    <span className="text-xs text-gray-500">Début</span>
                    <ChampMontant
                      valeur={s.uv_debut}
                      disabled={!corrigerOuverture}
                      onChange={(v) => majSolde(s.reseau_id, 'uv_debut', v)}
                    />
                  </div>
                  <div>
                    <span className="text-xs text-gray-500">Appro.</span>
                    <ChampMontant
                      valeur={s.approvisionnement}
                      onChange={(v) => majSolde(s.reseau_id, 'approvisionnement', v)}
                    />
                  </div>
                  <div>
                    <span className="text-xs text-gray-500">Fin</span>
                    <ChampMontant valeur={s.uv_fin} onChange={(v) => majSolde(s.reseau_id, 'uv_fin', v)} />
                  </div>
                </div>
              </div>
            );
          })}
        </section>

        {/* Espèces */}
        <section className="carte space-y-3">
          <h2 className="font-semibold">Caisse espèces</h2>
          <ChampMontant
            label="Espèces début (report de la veille)"
            valeur={especesDebut}
            disabled={!corrigerOuverture}
            onChange={setEspecesDebut}
          />
          <p className="text-xs text-gray-500">
            Apports / sorties = mouvements NON opérationnels (banque, propriétaire, transfert).
            Ne pas y mettre les dépôts/retraits clients.
          </p>
          <ChampMontant label="Apports d’espèces (non opérationnels)" valeur={apports} onChange={setApports} />
          <ChampMontant label="Sorties d’espèces (non opérationnelles)" valeur={sorties} onChange={setSorties} />

          <label className="flex items-center gap-2 text-sm text-gray-600">
            <input
              type="checkbox"
              checked={corrigerOuverture}
              onChange={(e) => setCorrigerOuverture(e.target.checked)}
            />
            Corriger les soldes d’ouverture (nécessite une justification)
          </label>
          {ouvertureDiverge && (
            <textarea
              className="champ-texte"
              placeholder="Justification de la correction d’ouverture (obligatoire)"
              value={motif}
              onChange={(e) => setMotif(e.target.value)}
            />
          )}
        </section>

        {/* Dépenses */}
        <SectionDepenses depenses={depenses} onChange={setDepenses} />

        {/* Créances */}
        <SectionCreances creances={creances} reseaux={reseaux} onChange={setCreances} />

        {/* Photo obligatoire */}
        <section className="carte space-y-2">
          <h2 className="font-semibold">Photo du solde du terminal (obligatoire)</h2>
          <input
            type="file"
            accept="image/*"
            capture="environment"
            onChange={choisirPhoto}
            className="block w-full text-sm"
          />
          {photoApercu && <img src={photoApercu} alt="Aperçu" className="mt-2 rounded-xl" />}
        </section>

        {message && (
          <p className="rounded-lg bg-blue-50 px-3 py-2 text-sm text-blue-800">{message}</p>
        )}
      </main>

      {/* Barre d'écart + soumission, fixée en bas (recompte avant de valider) */}
      <div className="fixed inset-x-0 bottom-0 border-t border-gray-200 bg-white">
        <div className="mx-auto max-w-lg space-y-2 px-4 py-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Espèces attendues</span>
            <span className="font-semibold tabular-nums">{formaterFcfa(resultat.especes_fin_attendue)}</span>
          </div>
          <ChampMontant label="Espèces comptées (fin)" valeur={especesFin} onChange={setEspecesFin} />
          <div
            className={`flex items-center justify-between rounded-xl px-3 py-2 font-semibold ${
              resultat.ecart === 0
                ? 'bg-emerald-100 text-emerald-800'
                : resultat.dans_tolerance
                  ? 'bg-amber-100 text-amber-800'
                  : 'bg-red-100 text-red-800'
            }`}
          >
            <span>Écart</span>
            <span className="tabular-nums">{formaterFcfa(resultat.ecart)}</span>
          </div>
          <button className="btn-principal" onClick={() => void soumettre()} disabled={enCours}>
            {enCours ? 'Enregistrement…' : 'Soumettre la clôture'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sous-sections dépenses / créances
// ---------------------------------------------------------------------------
function SectionDepenses({
  depenses,
  onChange,
}: {
  depenses: DepenseLocale[];
  onChange: (d: DepenseLocale[]) => void;
}) {
  const [montant, setMontant] = useState(0);
  const [description, setDescription] = useState('');

  function ajouter() {
    if (montant <= 0) return;
    onChange([
      ...depenses,
      {
        id: uuid(),
        cloture_id: '',
        pdv_id: '',
        date: '',
        categorie: null,
        montant,
        description: description.trim() || null,
        autorise_par: null,
      },
    ]);
    setMontant(0);
    setDescription('');
  }

  return (
    <section className="carte space-y-3">
      <h2 className="font-semibold">Dépenses de caisse</h2>
      {depenses.map((d) => (
        <div key={d.id} className="flex items-center justify-between text-sm">
          <span>{d.description ?? 'Dépense'}</span>
          <span className="flex items-center gap-2 tabular-nums">
            {formaterFcfa(d.montant)}
            <button
              className="text-red-600"
              onClick={() => onChange(depenses.filter((x) => x.id !== d.id))}
            >
              ✕
            </button>
          </span>
        </div>
      ))}
      <input
        className="champ-texte"
        placeholder="Description (ex. transport)"
        value={description}
        onChange={(e) => setDescription(e.target.value)}
      />
      <ChampMontant label="Montant" valeur={montant} onChange={setMontant} />
      <button className="btn-secondaire" onClick={ajouter}>
        + Ajouter la dépense
      </button>
    </section>
  );
}

function SectionCreances({
  creances,
  reseaux,
  onChange,
}: {
  creances: CreanceLocale[];
  reseaux: Reseau[];
  onChange: (c: CreanceLocale[]) => void;
}) {
  const [nom, setNom] = useState('');
  const [numero, setNumero] = useState('');
  const [montant, setMontant] = useState(0);
  const [reseauId, setReseauId] = useState('');
  const [autorisePar, setAutorisePar] = useState('');

  function ajouter() {
    if (montant <= 0 || !nom.trim() || !numero.trim() || !autorisePar.trim()) return;
    onChange([
      ...creances,
      {
        id: uuid(),
        cloture_id: '',
        pdv_id: '',
        agent_id: '',
        date_creation: '',
        client_nom: nom.trim(),
        client_numero: numero.trim(),
        reseau_id: reseauId || null,
        montant,
        motif: null,
        autorise_par: autorisePar.trim(),
        statut: 'EN_COURS',
      },
    ]);
    setNom('');
    setNumero('');
    setMontant(0);
    setReseauId('');
    setAutorisePar('');
  }

  return (
    <section className="carte space-y-3">
      <h2 className="font-semibold">Créances (dépôts non payés)</h2>
      <p className="text-xs text-gray-500">
        UV sorties sans encaissement d’espèces. Poste sensible : tout est tracé.
      </p>
      {creances.map((c) => (
        <div key={c.id} className="flex items-center justify-between text-sm">
          <span>
            {c.client_nom} · {c.client_numero}
          </span>
          <span className="flex items-center gap-2 tabular-nums">
            {formaterFcfa(c.montant)}
            <button
              className="text-red-600"
              onClick={() => onChange(creances.filter((x) => x.id !== c.id))}
            >
              ✕
            </button>
          </span>
        </div>
      ))}
      <input className="champ-texte" placeholder="Nom du client" value={nom} onChange={(e) => setNom(e.target.value)} />
      <input
        className="champ-texte"
        inputMode="tel"
        placeholder="Téléphone"
        value={numero}
        onChange={(e) => setNumero(e.target.value)}
      />
      <select className="champ-texte" value={reseauId} onChange={(e) => setReseauId(e.target.value)}>
        <option value="">Réseau (optionnel)</option>
        {reseaux.map((r) => (
          <option key={r.id} value={r.id}>
            {r.nom}
          </option>
        ))}
      </select>
      <input
        className="champ-texte"
        placeholder="Autorisé par (obligatoire)"
        value={autorisePar}
        onChange={(e) => setAutorisePar(e.target.value)}
      />
      <ChampMontant label="Montant" valeur={montant} onChange={setMontant} />
      <button className="btn-secondaire" onClick={ajouter}>
        + Ajouter la créance
      </button>
    </section>
  );
}
