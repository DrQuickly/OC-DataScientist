import { supabase } from './supabase';
import { db, type ClotureLocale } from './db';

/**
 * Moteur de synchronisation idempotent.
 * Pousse chaque clôture 'en_attente' vers le serveur :
 *   1. upload de la photo compressée -> URL Storage ;
 *   2. appel RPC soumettre_cloture (transaction atomique, idempotente) ;
 *   3. marquage local 'synchronise'.
 * Un échec marque 'erreur' (réessai au prochain passage).
 */

const BUCKET_PHOTOS = 'photos-clotures';

let enCours = false;

export type EcouteurSync = (etat: { enCours: boolean; enAttente: number }) => void;
const ecouteurs = new Set<EcouteurSync>();

export function surSync(cb: EcouteurSync): () => void {
  ecouteurs.add(cb);
  return () => ecouteurs.delete(cb);
}

async function notifier() {
  const enAttente = await db.clotures.where('etat_sync').anyOf('en_attente', 'erreur').count();
  for (const cb of ecouteurs) cb({ enCours, enAttente });
}

/** Tente de synchroniser toutes les clôtures en attente. Sûr à appeler souvent. */
export async function synchroniser(): Promise<void> {
  if (enCours || !navigator.onLine) return;
  enCours = true;
  await notifier();
  try {
    const aPousser = await db.clotures.where('etat_sync').anyOf('en_attente', 'erreur').toArray();
    for (const cloture of aPousser) {
      await pousserUne(cloture);
    }
  } finally {
    enCours = false;
    await notifier();
  }
}

async function pousserUne(cloture: ClotureLocale): Promise<void> {
  try {
    let photo_url = cloture.photo_url;

    // 1) Upload de la photo si nécessaire (idempotent : upsert sur le même chemin).
    //    On stocke le CHEMIN de l'objet (bucket privé) ; l'admin le signe à la lecture.
    if (!photo_url && cloture.photo_blob) {
      const chemin = `${cloture.id}.jpg`;
      const { error } = await supabase.storage
        .from(BUCKET_PHOTOS)
        .upload(chemin, cloture.photo_blob, { contentType: 'image/jpeg', upsert: true });
      if (error) throw error;
      photo_url = chemin;
    }

    // 2) Rassemble les enfants.
    const [soldes, depenses, creances] = await Promise.all([
      db.soldes.where('cloture_id').equals(cloture.id).toArray(),
      db.depenses.where('cloture_id').equals(cloture.id).toArray(),
      db.creances.where('cloture_id').equals(cloture.id).toArray(),
    ]);

    // 3) Soumission atomique côté serveur.
    const { error } = await supabase.rpc('soumettre_cloture', {
      p_cloture: {
        id: cloture.id,
        pdv_id: cloture.pdv_id,
        agent_id: cloture.agent_id,
        date_cloture: cloture.date_cloture,
        especes_debut: cloture.especes_debut,
        especes_fin_constatee: cloture.especes_fin_constatee,
        apports_especes: cloture.apports_especes,
        sorties_especes: cloture.sorties_especes,
        photo_url,
        motif_derogation: cloture.motif_derogation,
        saisi_le_client: cloture.saisi_le_client,
      },
      p_soldes: soldes,
      p_depenses: depenses,
      p_creances: creances,
    });
    if (error) throw error;

    // 4) Succès : on peut libérer le blob photo local.
    await db.clotures.update(cloture.id, {
      etat_sync: 'synchronise',
      statut: 'SOUMISE',
      photo_url,
      photo_blob: null,
      erreur_sync: null,
      maj_le: new Date().toISOString(),
    });
  } catch (e) {
    await db.clotures.update(cloture.id, {
      etat_sync: 'erreur',
      erreur_sync: e instanceof Error ? e.message : String(e),
      maj_le: new Date().toISOString(),
    });
  }
}

/** Démarre la synchro automatique : au retour du réseau et périodiquement. */
export function demarrerSyncAuto(): void {
  window.addEventListener('online', () => void synchroniser());
  // Filet de sécurité si l'événement 'online' est manqué.
  setInterval(() => void synchroniser(), 30_000);
  void synchroniser();
}
