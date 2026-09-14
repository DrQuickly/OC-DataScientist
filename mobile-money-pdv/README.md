# Gestion de points de vente Mobile Money

Application de gestion et de contrôle anti-fraude pour points de vente (PDV) de
mobile money en Côte d'Ivoire. Le contrôle repose sur une règle unique : **la
variation des UV prédit la variation des espèces**.

- Langue : français. Devise : FCFA (XOF), entiers, sans décimales.
- Fuseau : Africa/Abidjan (UTC+0). Dates : JJ/MM/AAAA.
- Mobile d'abord, fonctionnement hors connexion, intégrité des données non négociable.

## Stack

- Frontend : React + TypeScript + Vite (PWA installable), Tailwind CSS.
- Hors ligne : IndexedDB via Dexie, file de synchronisation idempotente (UUID client).
- Backend : Supabase (PostgreSQL + Auth + Storage + Row-Level Security).
- Calculs sensibles : vues PostgreSQL (font foi) + miroir TypeScript testé (aperçu).
- Tests : Vitest.

## État d'avancement (Phase 1)

| Élément | État |
|---|---|
| Logique métier des 5 formules (§2.2, §2.3) | ✅ `src/domain/`, 28 tests verts |
| Cas d'acceptation §8 (clôture + bilan) | ✅ couverts par tests |
| Schéma SQL + vues de calcul | ✅ `supabase/migrations/` |
| Audit append-only + immuabilité + horodatage serveur | ✅ |
| RLS (cloisonnement des agents) | ✅ |
| Edge Function création de comptes agents | ⏳ à venir |
| Auth + écrans agent/admin | ⏳ à venir |
| Mode hors ligne + synchronisation | ⏳ à venir |

## Démarrage

```bash
npm install
npm test           # exécute la suite Vitest
cp .env.example .env   # puis renseigner les clés Supabase
npm run dev        # (disponible une fois l'UI en place)
```

Voir `supabase/README.md` pour l'application des migrations et les décisions
d'architecture (horodatage serveur vs hors ligne, immuabilité, RLS).
