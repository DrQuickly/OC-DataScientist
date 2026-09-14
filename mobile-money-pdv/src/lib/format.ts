/**
 * Formatage FCFA (XOF) entier sans décimales et dates JJ/MM/AAAA.
 * Fuseau métier : Africa/Abidjan (UTC+0).
 */

const FMT_FCFA = new Intl.NumberFormat('fr-FR', { maximumFractionDigits: 0 });

/** "1 285 000 FCFA" */
export function formaterFcfa(montant: number): string {
  return `${FMT_FCFA.format(Math.round(montant))} FCFA`;
}

/** "1 285 000" (sans suffixe, pour l'affichage compact) */
export function formaterMontant(montant: number): string {
  return FMT_FCFA.format(Math.round(montant));
}

/**
 * Parse une saisie utilisateur en entier FCFA. Tolère espaces et séparateurs.
 * Renvoie 0 pour une saisie vide.
 */
export function parserMontant(saisie: string): number {
  const nettoye = saisie.replace(/[^0-9-]/g, '');
  if (nettoye === '' || nettoye === '-') return 0;
  return Math.trunc(Number(nettoye));
}

/** Date métier du jour au fuseau Africa/Abidjan (UTC+0), format ISO 'AAAA-MM-JJ'. */
export function dateMetierAujourdhui(): string {
  // Abidjan = UTC+0 : la date UTC est la date métier.
  return new Date().toISOString().slice(0, 10);
}

/** 'AAAA-MM-JJ' -> 'JJ/MM/AAAA' */
export function formaterDate(iso: string): string {
  const [a, m, j] = iso.slice(0, 10).split('-');
  return `${j}/${m}/${a}`;
}

/** Mois métier courant, 'AAAA-MM'. */
export function moisMetierCourant(): string {
  return new Date().toISOString().slice(0, 7);
}
