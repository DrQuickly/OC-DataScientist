/**
 * Types du domaine métier — gestion de points de vente Mobile Money.
 *
 * Convention : tous les montants sont des entiers en FCFA (XOF), sans décimales.
 * Aucune valeur calculée (flux_net_uv, especes_fin_attendue, ecart, continuités)
 * n'est stockée telle quelle : elle est toujours recalculée à partir des saisies,
 * ici côté client pour l'aperçu, et côté PostgreSQL (vues) pour la valeur qui fait foi.
 */

/** Statut métier d'une clôture, géré côté serveur. */
export type StatutCloture = 'BROUILLON' | 'SOUMISE' | 'VALIDEE' | 'CONTESTEE';

/**
 * Solde d'un réseau (Orange, MTN, Moov, Wave) pour une clôture donnée.
 * uv_debut : stock d'UV en début de journée (pré-rempli depuis la clôture précédente).
 * approvisionnement : injection d'UV pendant la journée (saisie agent, validée admin).
 * uv_fin : stock d'UV constaté en fin de journée.
 */
export interface SoldeReseau {
  reseau_id: string;
  uv_debut: number;
  approvisionnement: number;
  uv_fin: number;
}

/**
 * Entrées d'une clôture nécessaires au calcul de l'écart.
 * Ce sont uniquement des valeurs SAISIES (jamais des valeurs calculées).
 */
export interface EntreesCloture {
  /** Soldes UV par réseau. */
  soldes: SoldeReseau[];
  /** Espèces en caisse en début de journée (pré-remplies depuis la veille). */
  especes_debut: number;
  /** Espèces comptées physiquement en fin de journée. */
  especes_fin_constatee: number;
  /** Apports d'espèces dans la caisse pendant la journée (hors opérations MM). */
  apports_especes: number;
  /** Sorties d'espèces de la caisse pendant la journée (décaissements retraits clients). */
  sorties_especes: number;
  /** Somme des dépenses (charges de caisse) rattachées à la clôture. */
  total_depenses: number;
  /** Somme des créances créées ce jour (UV sorties, espèces non encaissées). */
  total_creances_creees: number;
  /** Somme des remboursements de créances encaissés ce jour. */
  total_creances_remboursees: number;
}

/** Résultat complet du calcul de clôture. */
export interface ResultatCloture {
  flux_net_uv: number;
  especes_fin_attendue: number;
  ecart: number;
  /** true si |ecart| <= tolérance : la clôture est considérée équilibrée. */
  dans_tolerance: boolean;
}

/** Écart de continuité d'UV pour un réseau entre J-1 et J. */
export interface ContinuiteReseau {
  reseau_id: string;
  /** uv_debut(J) - uv_fin(J-1). Doit valoir 0 en l'absence d'anomalie. */
  continuite_uv: number;
}
