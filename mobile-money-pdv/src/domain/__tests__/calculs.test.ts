import { describe, expect, it } from 'vitest';
import {
  TOLERANCE_DEFAUT,
  calculerCloture,
  calculerContinuiteEspeces,
  calculerContinuiteUv,
  calculerEcart,
  calculerEspecesFinAttendue,
  calculerFluxNetUv,
} from '../calculs';
import type { EntreesCloture, SoldeReseau } from '../types';

/**
 * Cas d'acceptation obligatoire — section 8 du cahier des charges.
 * Clôture du 01/09/2026, PDV 1.
 */
const soldesAcceptation: SoldeReseau[] = [
  { reseau_id: 'orange', uv_debut: 850_000, approvisionnement: 500_000, uv_fin: 410_000 },
  { reseau_id: 'mtn', uv_debut: 300_000, approvisionnement: 0, uv_fin: 355_000 },
  { reseau_id: 'moov', uv_debut: 150_000, approvisionnement: 0, uv_fin: 120_000 },
  { reseau_id: 'wave', uv_debut: 600_000, approvisionnement: 200_000, uv_fin: 430_000 },
];

const entreesAcceptation: EntreesCloture = {
  soldes: soldesAcceptation,
  especes_debut: 120_000,
  especes_fin_constatee: 665_000,
  apports_especes: 0,
  sorties_especes: 700_000,
  total_depenses: 15_000,
  total_creances_creees: 25_000,
  total_creances_remboursees: 0,
};

describe('Cas d’acceptation obligatoire (section 8)', () => {
  it('flux_net_uv = 1 285 000', () => {
    expect(calculerFluxNetUv(soldesAcceptation)).toBe(1_285_000);
  });

  it('especes_fin_attendue = 665 000', () => {
    expect(calculerEspecesFinAttendue(entreesAcceptation)).toBe(665_000);
  });

  it('ecart = 0', () => {
    expect(calculerEcart(entreesAcceptation)).toBe(0);
  });

  it('clôture complète : flux, attendue, écart et tolérance', () => {
    const r = calculerCloture(entreesAcceptation);
    expect(r).toEqual({
      flux_net_uv: 1_285_000,
      especes_fin_attendue: 665_000,
      ecart: 0,
      dans_tolerance: true,
    });
  });
});

describe('flux_net_uv (section 2.2)', () => {
  it('additionne (uv_debut + appro - uv_fin) réseau par réseau', () => {
    // Orange: 850000+500000-410000 = 940000
    // MTN:    300000+0-355000       = -55000
    // Moov:   150000+0-120000       =  30000
    // Wave:   600000+200000-430000  = 370000
    expect(calculerFluxNetUv(soldesAcceptation)).toBe(940_000 - 55_000 + 30_000 + 370_000);
  });

  it('vaut 0 si aucun mouvement d’UV', () => {
    expect(
      calculerFluxNetUv([{ reseau_id: 'orange', uv_debut: 100_000, approvisionnement: 0, uv_fin: 100_000 }]),
    ).toBe(0);
  });

  it('négatif si les UV augmentent (net de retraits)', () => {
    expect(
      calculerFluxNetUv([{ reseau_id: 'mtn', uv_debut: 100_000, approvisionnement: 0, uv_fin: 150_000 }]),
    ).toBe(-50_000);
  });
});

describe('especes_fin_attendue (section 2.2)', () => {
  it('les créances créées réduisent l’espèce attendue', () => {
    const base: EntreesCloture = {
      soldes: [{ reseau_id: 'orange', uv_debut: 0, approvisionnement: 0, uv_fin: 0 }],
      especes_debut: 100_000,
      especes_fin_constatee: 0,
      apports_especes: 0,
      sorties_especes: 0,
      total_depenses: 0,
      total_creances_creees: 0,
      total_creances_remboursees: 0,
    };
    expect(calculerEspecesFinAttendue(base)).toBe(100_000);
    expect(calculerEspecesFinAttendue({ ...base, total_creances_creees: 30_000 })).toBe(70_000);
  });

  it('les remboursements de créances augmentent l’espèce attendue', () => {
    const base: EntreesCloture = {
      soldes: [],
      especes_debut: 0,
      especes_fin_constatee: 0,
      apports_especes: 0,
      sorties_especes: 0,
      total_depenses: 0,
      total_creances_creees: 0,
      total_creances_remboursees: 40_000,
    };
    expect(calculerEspecesFinAttendue(base)).toBe(40_000);
  });
});

describe('ecart et tolérance', () => {
  it('écart négatif = manquant', () => {
    const r = calculerCloture({ ...entreesAcceptation, especes_fin_constatee: 660_000 });
    expect(r.ecart).toBe(-5_000);
    expect(r.dans_tolerance).toBe(false);
  });

  it('écart positif = excédent, anormal aussi', () => {
    const r = calculerCloture({ ...entreesAcceptation, especes_fin_constatee: 670_000 });
    expect(r.ecart).toBe(5_000);
    expect(r.dans_tolerance).toBe(false);
  });

  it('un écart dans la tolérance par défaut (1000) est accepté', () => {
    const r = calculerCloture({ ...entreesAcceptation, especes_fin_constatee: 665_800 });
    expect(r.ecart).toBe(800);
    expect(r.dans_tolerance).toBe(true);
  });

  it('la tolérance est paramétrable', () => {
    const r = calculerCloture({ ...entreesAcceptation, especes_fin_constatee: 668_000 }, 5_000);
    expect(r.ecart).toBe(3_000);
    expect(r.dans_tolerance).toBe(true);
  });

  it('TOLERANCE_DEFAUT vaut 1000', () => {
    expect(TOLERANCE_DEFAUT).toBe(1_000);
  });
});

describe('continuité (section 2.3)', () => {
  it('continuite_especes = especes_debut(J) - especes_fin_constatee(J-1)', () => {
    // Ouverture conforme à la clôture de la veille => 0
    expect(calculerContinuiteEspeces(665_000, 665_000)).toBe(0);
    // Ouverture supérieure => argent injecté non déclaré
    expect(calculerContinuiteEspeces(700_000, 665_000)).toBe(35_000);
    // Ouverture inférieure => argent retiré non déclaré
    expect(calculerContinuiteEspeces(600_000, 665_000)).toBe(-65_000);
  });

  it('continuite_uv réseau par réseau = uv_debut(J) - uv_fin(J-1)', () => {
    const veille: SoldeReseau[] = [
      { reseau_id: 'orange', uv_debut: 0, approvisionnement: 0, uv_fin: 410_000 },
      { reseau_id: 'mtn', uv_debut: 0, approvisionnement: 0, uv_fin: 355_000 },
    ];
    const jour: SoldeReseau[] = [
      { reseau_id: 'orange', uv_debut: 410_000, approvisionnement: 0, uv_fin: 0 }, // conforme
      { reseau_id: 'mtn', uv_debut: 300_000, approvisionnement: 0, uv_fin: 0 }, // -55000 : anomalie
    ];
    expect(calculerContinuiteUv(jour, veille)).toEqual([
      { reseau_id: 'orange', continuite_uv: 0 },
      { reseau_id: 'mtn', continuite_uv: -55_000 },
    ]);
  });

  it('un réseau absent de la veille est traité comme uv_fin = 0', () => {
    const jour: SoldeReseau[] = [{ reseau_id: 'wave', uv_debut: 50_000, approvisionnement: 0, uv_fin: 0 }];
    expect(calculerContinuiteUv(jour, [])).toEqual([{ reseau_id: 'wave', continuite_uv: 50_000 }]);
  });
});
