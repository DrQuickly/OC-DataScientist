# Base de données Supabase — noyau Phase 1

Schéma, vues de calcul, audit, immuabilité et RLS pour l'application de gestion
de points de vente Mobile Money.

## Ordre d'application des migrations

Les fichiers de `migrations/` s'appliquent dans l'ordre numérique :

| Fichier | Rôle |
|---|---|
| `0001_schema.sql` | Tables, types énumérés, contraintes, index. |
| `0002_vues_calculs.sql` | Champs calculés (jamais stockés) : `flux_net_uv`, `especes_fin_attendue`, `ecart`, continuités, bilan mensuel. |
| `0003_audit_immuabilite.sql` | Journal append-only, immuabilité des clôtures soumises, photo obligatoire, horodatage serveur, contrôle des 100 %. |
| `0004_rls.sql` | Row-Level Security. Un agent ne peut jamais lire commissions/charges/rentabilité. |
| `0005_reference.sql` | Données de référence (les 4 réseaux). Aucune donnée de démo. |
| `0006_creances_lien_cloture_et_controles.sql` | Lien explicite créance↔clôture (création + remboursement), gel des dépenses/créances d'une clôture verrouillée, contrôle de continuité des soldes d'ouverture (motif obligatoire si divergence). |
| `0007_rpc_soumettre_cloture.sql` | Fonction `soumettre_cloture(...)` : soumission atomique et idempotente d'une clôture + enfants (appelée par la file de synchronisation hors ligne). |
| `0008_storage_photos.sql` | Bucket privé `photos-clotures` + politiques (écriture/lecture authentifiées). |

## Edge Function

`supabase/functions/creer-agent/` crée un compte AGENT (auth + profil + affectations)
via la clé `service_role`, après vérification que l'appelant est ADMIN.

```bash
supabase functions deploy creer-agent
# SUPABASE_URL / SUPABASE_ANON_KEY / SUPABASE_SERVICE_ROLE_KEY sont fournis
# automatiquement par Supabase à la fonction.
```

## Premier administrateur

Aucune donnée de démo n'est insérée. Pour créer le tout premier ADMIN (les
suivants et les agents se créent ensuite depuis l'app) :

1. Créer l'utilisateur dans **Authentication → Users** (ou via l'API admin).
2. Insérer son profil :

```sql
insert into public.utilisateurs (id, nom_complet, role, actif)
values ('<uuid-de-auth.users>', 'Nom Propriétaire', 'ADMIN', true);
```

## Point d'attention — photo (bucket privé)

Le bucket `photos-clotures` est **privé**. La synchro stocke le *chemin* de
l'objet dans `clotures.photo_url` ; l'écran admin génère une **URL signée**
(`createSignedUrl`) à la demande. Ne pas revenir à `getPublicUrl`.

### Avec le CLI Supabase

```bash
supabase db push          # applique les migrations au projet lié
# ou, base locale de dev :
supabase start
supabase db reset         # rejoue toutes les migrations
```

### Sans CLI

Coller chaque fichier, dans l'ordre, dans l'éditeur SQL du dashboard Supabase.

## Décisions d'architecture (à connaître avant de coder l'UI)

### Horodatage serveur vs saisie hors ligne
Contrainte du cahier : « horodatage serveur uniquement » ET « saisie hors ligne
obligatoire ». C'est contradictoire au moment de la saisie. Résolution :

- `clotures.saisi_le_client` : heure du téléphone. **Purement indicative**, jamais
  utilisée pour un contrôle.
- `clotures.horodatage_serveur` et `soumise_le` : posés par PostgreSQL (`now()`)
  au moment de la synchro. **Seuls à faire foi.**
- `date_cloture` est une **date métier** (quel jour on clôture), pas un timestamp.

### Écart calculé côté serveur, aperçu côté client
- L'agent voit un aperçu de l'écart hors ligne (calcul JS de `src/domain/calculs.ts`).
- La valeur qui fait foi est la vue `v_cloture_calculs` (PostgreSQL).
- Les deux formules sont identiques ; les tests Vitest prouvent la logique.

### Immuabilité
- Une clôture qui n'est plus en `BROUILLON` a ses valeurs financières figées
  (trigger `clotures_immuables`). Seule la transition de statut (validation admin)
  est permise. Toute correction passe par une **écriture d'annulation** tracée.
- `journal_audit` est **append-only** (ni UPDATE ni DELETE).

### Cloisonnement des agents (RLS)
- Les tables `commissions`, `charges_fixes`, `cles_repartition` et la vue
  `v_bilan_mensuel` n'ont **aucune** policy de lecture pour l'agent → RLS renvoie
  zéro ligne, y compris en cas de requête forgée. Garanti en base, pas seulement
  dans l'interface.

### Création des comptes agents
- Impossible en pur client (l'API admin de Supabase Auth exige la `service_role`).
- À faire via une **Edge Function** côté serveur (étape suivante de la Phase 1).

## Vérification du cas d'acceptation (section 8)

La logique métier est couverte par `src/domain/__tests__` (28 tests, verts).
Une fois un projet Supabase connecté, un test d'intégration devra confirmer que
`v_cloture_calculs` renvoie `especes_fin_attendue = 665 000` et `ecart = 0` pour
le jeu de données de la section 8 — c'est-à-dire que le SQL et le TS concordent.
