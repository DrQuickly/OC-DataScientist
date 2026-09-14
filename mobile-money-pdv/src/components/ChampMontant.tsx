import { formaterMontant, parserMontant } from '../lib/format';

/**
 * Champ de saisie d'un montant FCFA entier. Force le clavier numérique
 * (inputMode="numeric") et affiche la valeur formatée avec séparateurs.
 */
export function ChampMontant({
  valeur,
  onChange,
  label,
  autoFocus,
  disabled,
}: {
  valeur: number;
  onChange: (v: number) => void;
  label?: string;
  autoFocus?: boolean;
  disabled?: boolean;
}) {
  return (
    <label className="block">
      {label && <span className="mb-1 block text-sm font-medium text-gray-600">{label}</span>}
      <input
        type="text"
        inputMode="numeric"
        pattern="[0-9]*"
        autoFocus={autoFocus}
        disabled={disabled}
        className="champ-numerique disabled:bg-gray-100"
        value={valeur === 0 ? '' : formaterMontant(valeur)}
        placeholder="0"
        onChange={(e) => onChange(parserMontant(e.target.value))}
      />
    </label>
  );
}
