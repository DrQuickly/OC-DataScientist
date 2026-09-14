import Dexie, { type Table } from 'dexie';
import type { StatutCloture } from '../domain/types';

/**
 * Base locale IndexedDB (via Dexie) : permet la saisie hors connexion.
 * Chaque clôture porte un UUID généré côté client -> synchronisation idempotente
 * (un renvoi ne crée pas de doublon).
 *
 * etat_sync suit le cycle local, DISTINCT du statut métier serveur :
 *   'local'        : brouillon en cours de saisie, non encore soumis
 *   'en_attente'   : soumis localement, à pousser vers le serveur
 *   'synchronise'  : confirmé côté serveur
 *   'erreur'       : échec de synchronisation (à réessayer)
 */
export type EtatSync = 'local' | 'en_attente' | 'synchronise' | 'erreur';

export interface ClotureLocale {
  id: string; // UUID client
  pdv_id: string;
  agent_id: string;
  date_cloture: string; // 'AAAA-MM-JJ'
  especes_debut: number;
  especes_fin_constatee: number;
  apports_especes: number;
  sorties_especes: number;
  motif_derogation: string | null;
  photo_blob: Blob | null; // photo compressée, en attente d'upload
  photo_url: string | null; // renseignée après upload Storage
  statut: StatutCloture;
  etat_sync: EtatSync;
  erreur_sync: string | null;
  saisi_le_client: string; // ISO, indicatif uniquement
  maj_le: string; // ISO
}

export interface SoldeLocal {
  id: string;
  cloture_id: string;
  reseau_id: string;
  uv_debut: number;
  approvisionnement: number;
  uv_fin: number;
}

export interface DepenseLocale {
  id: string;
  cloture_id: string;
  pdv_id: string;
  date: string;
  categorie: string | null;
  montant: number;
  description: string | null;
  autorise_par: string | null;
}

export interface CreanceLocale {
  id: string;
  cloture_id: string;
  pdv_id: string;
  agent_id: string;
  date_creation: string;
  client_nom: string;
  client_numero: string;
  reseau_id: string | null;
  montant: number;
  motif: string | null;
  autorise_par: string;
  statut: 'EN_COURS' | 'A_RELANCER' | 'ALERTE' | 'SOLDEE';
}

class BaseLocale extends Dexie {
  clotures!: Table<ClotureLocale, string>;
  soldes!: Table<SoldeLocal, string>;
  depenses!: Table<DepenseLocale, string>;
  creances!: Table<CreanceLocale, string>;

  constructor() {
    super('pdv_momo');
    this.version(1).stores({
      clotures: 'id, pdv_id, date_cloture, etat_sync, statut',
      soldes: 'id, cloture_id',
      depenses: 'id, cloture_id',
      creances: 'id, cloture_id',
    });
  }
}

export const db = new BaseLocale();

/** Nombre d'opérations en attente de synchronisation (indicateur d'en-tête). */
export async function nombreEnAttente(): Promise<number> {
  return db.clotures.where('etat_sync').anyOf('en_attente', 'erreur').count();
}
