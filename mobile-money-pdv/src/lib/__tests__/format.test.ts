import { describe, expect, it } from 'vitest';
import { formaterFcfa, formaterMontant, parserMontant, formaterDate } from '../format';

describe('parserMontant', () => {
  it('ignore les espaces et séparateurs', () => {
    expect(parserMontant('1 285 000')).toBe(1_285_000);
    expect(parserMontant('665.000')).toBe(665_000);
    expect(parserMontant('120 000 FCFA')).toBe(120_000);
  });

  it('renvoie 0 pour une saisie vide ou non numérique', () => {
    expect(parserMontant('')).toBe(0);
    expect(parserMontant('abc')).toBe(0);
    expect(parserMontant('-')).toBe(0);
  });

  it('tronque vers l’entier (pas de décimales FCFA)', () => {
    expect(parserMontant('100')).toBe(100);
  });
});

const sansEspaces = (x: string) => x.replace(/\s/g, '');

describe('formatage', () => {
  it('formaterMontant met des séparateurs de milliers', () => {
    expect(sansEspaces(formaterMontant(1_285_000))).toBe('1285000');
  });

  it('formaterFcfa ajoute le suffixe FCFA', () => {
    expect(sansEspaces(formaterFcfa(665_000))).toBe('665000FCFA');
  });

  it('formaterDate convertit ISO en JJ/MM/AAAA', () => {
    expect(formaterDate('2026-09-01')).toBe('01/09/2026');
  });
});
