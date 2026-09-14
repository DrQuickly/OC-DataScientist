import { supabase } from './supabase';
import { db } from './db';

export interface Reseau {
  id: string;
  nom: string;
}

const CLE_RESEAUX = 'cache_reseaux';

/** Réseaux actifs, avec cache localStorage pour le fonctionnement hors ligne. */
export async function chargerReseaux(): Promise<Reseau[]> {
  try {
    const { data, error } = await supabase.from('reseaux').select('id, nom').eq('actif', true).order('nom');
    if (!error && data) {
      localStorage.setItem(CLE_RESEAUX, JSON.stringify(data));
      return data;
    }
  } catch {
    /* hors ligne : on retombe sur le cache */
  }
  const cache = localStorage.getItem(CLE_RESEAUX);
  return cache ? (JSON.parse(cache) as Reseau[]) : [];
}

export interface OuvertureParDefaut {
  especes_debut: number;
  uv_debut: Record<string, number>; // reseau_id -> uv_fin de la clôture précédente
  source: 'serveur' | 'local' | 'aucune';
}

/**
 * Pré-remplit l'ouverture depuis la clôture PRÉCÉDENTE du PDV.
 * Cherche d'abord côté serveur (le plus fiable) ; à défaut, la dernière clôture
 * locale synchronisée. Aucune valeur trouvée => 0 (première clôture).
 */
export async function chargerOuvertureParDefaut(
  pdv_id: string,
  dateCloture: string,
): Promise<OuvertureParDefaut> {
  // 1) Serveur
  try {
    const { data: prev } = await supabase
      .from('clotures')
      .select('id, especes_fin_constatee, date_cloture')
      .eq('pdv_id', pdv_id)
      .lt('date_cloture', dateCloture)
      .order('date_cloture', { ascending: false })
      .limit(1)
      .maybeSingle();

    if (prev) {
      const { data: soldes } = await supabase
        .from('cloture_soldes')
        .select('reseau_id, uv_fin')
        .eq('cloture_id', prev.id);
      const uv: Record<string, number> = {};
      for (const s of soldes ?? []) uv[s.reseau_id] = s.uv_fin;
      return { especes_debut: prev.especes_fin_constatee, uv_debut: uv, source: 'serveur' };
    }
  } catch {
    /* hors ligne */
  }

  // 2) Local (dernière clôture synchronisée du PDV)
  const localesPrev = await db.clotures
    .where('pdv_id')
    .equals(pdv_id)
    .and((c) => c.date_cloture < dateCloture && c.etat_sync === 'synchronise')
    .sortBy('date_cloture');
  const derniere = localesPrev.at(-1);
  if (derniere) {
    const soldes = await db.soldes.where('cloture_id').equals(derniere.id).toArray();
    const uv: Record<string, number> = {};
    for (const s of soldes) uv[s.reseau_id] = s.uv_fin;
    return { especes_debut: derniere.especes_fin_constatee, uv_debut: uv, source: 'local' };
  }

  return { especes_debut: 0, uv_debut: {}, source: 'aucune' };
}
