/**
 * Bilan mensuel / annuel (sections 2.5 et 2.6 du cahier des charges).
 *
 * ERREUR À NE JAMAIS COMMETTRE : le volume d'UV qui circule n'est PAS un
 * chiffre d'affaires. C'est du stock qui tourne. Le vrai produit d'exploitation,
 * ce sont les COMMISSIONS versées par les opérateurs.
 *
 * Fonctions pures, testables, miroir des vues PostgreSQL côté serveur.
 */

/** Une charge commune répartie sur un PDV via une clé (en pourcentage). */
export interface CleRepartition {
  pdv_id: string;
  /** Pourcentage entier ou décimal, ex. 50 pour 50 %. */
  pourcentage: number;
}

/** Entrées du bilan mensuel d'un PDV. */
export interface EntreesBilan {
  /** Commissions effectivement reçues des opérateurs pour ce PDV sur le mois. */
  commissions_recues: number;
  /** Charges imputées à 100 % à ce PDV (loyer propre, etc.). */
  charges_directes: number;
  /** Quote-parts de charges communes affectées à ce PDV. */
  quoteparts_communes: number;
  /** Autres produits éventuels (par défaut 0). */
  autres_produits?: number;
  /** Créances non soldées à la clôture du mois (argent immobilisé/perdu). */
  creances_non_soldees: number;
}

/**
 * Quote-part d'une charge commune pour un PDV.
 * quote_part = montant_commun * pourcentage / 100, arrondie à l'entier FCFA.
 */
export function calculerQuotePart(
  montant_commun: number,
  pourcentage: number,
): number {
  return Math.round((montant_commun * pourcentage) / 100);
}

/**
 * Vérifie que, pour un type de charge donné, la somme des pourcentages sur
 * tous les PDV actifs fait EXACTEMENT 100 %. L'application doit bloquer sinon.
 *
 * @param cles      clés de répartition d'un même type de charge
 * @param epsilon   tolérance d'arrondi flottant (les pourcentages restent des %)
 */
export function validerClesRepartition(
  cles: CleRepartition[],
  epsilon = 1e-9,
): boolean {
  const somme = cles.reduce((acc, c) => acc + c.pourcentage, 0);
  return Math.abs(somme - 100) <= epsilon;
}

/**
 * resultat_net =
 *     commissions_recues
 *   + autres_produits
 *   - charges_directes
 *   - quoteparts_communes
 *   - creances_non_soldees
 */
export function calculerResultatNet(entrees: EntreesBilan): number {
  return (
    entrees.commissions_recues +
    (entrees.autres_produits ?? 0) -
    entrees.charges_directes -
    entrees.quoteparts_communes -
    entrees.creances_non_soldees
  );
}
