import { describe, expect, it } from 'vitest';
import {
  calculerQuotePart,
  calculerResultatNet,
  validerClesRepartition,
} from '../bilan';
import type { CleRepartition } from '../bilan';

/**
 * Cas d'acceptation obligatoire — section 8, bilan mensuel septembre 2026, PDV 1.
 * Commission reçue 172 500 · loyer direct 75 000 · internet commun 30 000
 * réparti 50/30/20/0 sur quatre PDV · créance non soldée 25 000.
 */
describe('Bilan mensuel — cas d’acceptation (section 8)', () => {
  const clesInternet: CleRepartition[] = [
    { pdv_id: 'pdv1', pourcentage: 50 },
    { pdv_id: 'pdv2', pourcentage: 30 },
    { pdv_id: 'pdv3', pourcentage: 20 },
    { pdv_id: 'pdv4', pourcentage: 0 },
  ];

  it('la clé de répartition internet somme bien à 100 %', () => {
    expect(validerClesRepartition(clesInternet)).toBe(true);
  });

  it('quote-part internet PDV 1 = 15 000', () => {
    expect(calculerQuotePart(30_000, 50)).toBe(15_000);
  });

  it('resultat_net PDV 1 = 57 500', () => {
    const quotePartInternet = calculerQuotePart(30_000, 50);
    const resultat = calculerResultatNet({
      commissions_recues: 172_500,
      charges_directes: 75_000, // loyer direct
      quoteparts_communes: quotePartInternet, // 15 000
      creances_non_soldees: 25_000,
    });
    // 172 500 - 75 000 - 15 000 + 0 - 25 000 = 57 500
    expect(resultat).toBe(57_500);
  });
});

describe('validerClesRepartition (section 2.6)', () => {
  it('bloque une répartition qui ne fait pas 100 %', () => {
    expect(
      validerClesRepartition([
        { pdv_id: 'pdv1', pourcentage: 50 },
        { pdv_id: 'pdv2', pourcentage: 30 },
      ]),
    ).toBe(false);
  });

  it('accepte une répartition décimale valide', () => {
    expect(
      validerClesRepartition([
        { pdv_id: 'pdv1', pourcentage: 33.3 },
        { pdv_id: 'pdv2', pourcentage: 33.3 },
        { pdv_id: 'pdv3', pourcentage: 33.4 },
      ]),
    ).toBe(true);
  });

  it('accepte une charge affectée à 100 % à un seul PDV', () => {
    expect(validerClesRepartition([{ pdv_id: 'pdv1', pourcentage: 100 }])).toBe(true);
  });
});

describe('calculerQuotePart', () => {
  it('arrondit à l’entier FCFA', () => {
    expect(calculerQuotePart(10_000, 33.33)).toBe(3_333);
    expect(calculerQuotePart(30_000, 20)).toBe(6_000);
  });

  it('une quote-part à 0 % vaut 0', () => {
    expect(calculerQuotePart(30_000, 0)).toBe(0);
  });
});

describe('calculerResultatNet (section 2.5)', () => {
  it('le produit vient des commissions, pas du volume d’UV', () => {
    // Aucune commission, gros volume : le résultat ne doit pas gonfler.
    expect(
      calculerResultatNet({
        commissions_recues: 0,
        charges_directes: 0,
        quoteparts_communes: 0,
        creances_non_soldees: 0,
      }),
    ).toBe(0);
  });

  it('les créances non soldées pèsent négativement sur le résultat', () => {
    expect(
      calculerResultatNet({
        commissions_recues: 100_000,
        charges_directes: 0,
        quoteparts_communes: 0,
        creances_non_soldees: 40_000,
      }),
    ).toBe(60_000);
  });

  it('prend en compte les autres produits éventuels', () => {
    expect(
      calculerResultatNet({
        commissions_recues: 100_000,
        autres_produits: 10_000,
        charges_directes: 20_000,
        quoteparts_communes: 5_000,
        creances_non_soldees: 0,
      }),
    ).toBe(85_000);
  });
});
