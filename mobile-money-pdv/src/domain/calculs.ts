/**
 * Cœur métier : les cinq formules validées (sections 2.2 et 2.3 du cahier des charges).
 *
 * Règle fondamentale : chaque opération a deux jambes. La variation des UV prédit
 * la variation des espèces. C'est le seul mécanisme de contrôle fiable.
 *
 *   - Dépôt client  : UV sortent, espèces entrent.
 *   - Retrait client : UV entrent, espèces sortent.
 *
 * Ces fonctions sont PURES (aucun effet de bord, aucune dépendance réseau/horloge)
 * afin d'être testables et de produire exactement le même résultat que les vues
 * PostgreSQL côté serveur. Le calcul client sert d'aperçu ; le serveur fait foi.
 */

import type {
  ContinuiteReseau,
  EntreesCloture,
  ResultatCloture,
  SoldeReseau,
} from './types';

/** Tolérance par défaut sur l'écart, en FCFA. Paramétrable par l'admin. */
export const TOLERANCE_DEFAUT = 1000;

/**
 * flux_net_uv = somme sur les réseaux de (uv_debut + approvisionnement - uv_fin).
 *
 * Positif  => les UV ont globalement DIMINUÉ (net de dépôts) => des espèces ont dû ENTRER.
 * Négatif  => les UV ont globalement AUGMENTÉ (net de retraits) => des espèces ont dû SORTIR.
 */
export function calculerFluxNetUv(soldes: SoldeReseau[]): number {
  return soldes.reduce(
    (acc, s) => acc + (s.uv_debut + s.approvisionnement - s.uv_fin),
    0,
  );
}

/**
 * especes_fin_attendue =
 *     especes_debut
 *   + flux_net_uv
 *   + apports_especes
 *   - sorties_especes
 *   - total_depenses
 *   - total_creances_creees        (UV sorties mais espèces non encaissées)
 *   + total_creances_remboursees   (espèces encaissées après coup)
 */
export function calculerEspecesFinAttendue(entrees: EntreesCloture): number {
  const flux = calculerFluxNetUv(entrees.soldes);
  return (
    entrees.especes_debut +
    flux +
    entrees.apports_especes -
    entrees.sorties_especes -
    entrees.total_depenses -
    entrees.total_creances_creees +
    entrees.total_creances_remboursees
  );
}

/**
 * ecart = especes_fin_constatee - especes_fin_attendue.
 * Négatif => manquant. Positif => excédent (tout aussi anormal).
 */
export function calculerEcart(entrees: EntreesCloture): number {
  return entrees.especes_fin_constatee - calculerEspecesFinAttendue(entrees);
}

/**
 * Calcul complet d'une clôture, avec test de tolérance.
 */
export function calculerCloture(
  entrees: EntreesCloture,
  tolerance: number = TOLERANCE_DEFAUT,
): ResultatCloture {
  const flux_net_uv = calculerFluxNetUv(entrees.soldes);
  const especes_fin_attendue = calculerEspecesFinAttendue(entrees);
  const ecart = entrees.especes_fin_constatee - especes_fin_attendue;
  return {
    flux_net_uv,
    especes_fin_attendue,
    ecart,
    dans_tolerance: Math.abs(ecart) <= tolerance,
  };
}

/**
 * Continuité des espèces (section 2.3) :
 * continuite_especes = especes_debut(J) - especes_fin_constatee(J-1).
 * Un écart non nul signale une opération non déclarée ou une manipulation
 * entre la clôture de la veille et l'ouverture du jour.
 */
export function calculerContinuiteEspeces(
  especes_debut_jour: number,
  especes_fin_constatee_veille: number,
): number {
  return especes_debut_jour - especes_fin_constatee_veille;
}

/**
 * Continuité des UV (section 2.3), réseau par réseau :
 * continuite_uv = uv_debut(J) - uv_fin(J-1).
 *
 * @param soldesJour    soldes d'ouverture du jour
 * @param soldesVeille  soldes de clôture de la veille (indexés par reseau_id)
 * @returns un écart de continuité par réseau présent dans soldesJour
 */
export function calculerContinuiteUv(
  soldesJour: SoldeReseau[],
  soldesVeille: SoldeReseau[],
): ContinuiteReseau[] {
  const uvFinVeille = new Map<string, number>();
  for (const s of soldesVeille) {
    uvFinVeille.set(s.reseau_id, s.uv_fin);
  }
  return soldesJour.map((s) => ({
    reseau_id: s.reseau_id,
    continuite_uv: s.uv_debut - (uvFinVeille.get(s.reseau_id) ?? 0),
  }));
}
