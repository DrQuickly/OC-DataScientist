# Guide — Créer et brancher le projet Supabase (Phase 1)

Objectif : passer de « code prêt » à « application qui tourne ». Compte ~20 min.
Aucune donnée de démo n'est créée : on met en place le strict nécessaire.

---

## Étape 1 — Créer le projet

1. Va sur https://supabase.com → **Sign in** (GitHub ou e-mail).
2. **New project**.
   - **Name** : `pdv-momo` (ou ce que tu veux).
   - **Database password** : choisis un mot de passe fort et **note-le** (tu ne le reverras pas).
   - **Region** : la plus proche de la Côte d'Ivoire proposée (ex. *West EU (London)* ou *Frankfurt*). Il n'y a pas de région en Afrique de l'Ouest ; l'Europe de l'Ouest est le meilleur compromis de latence.
3. **Create new project**, patiente ~2 min que la base se provisionne.

---

## Étape 2 — Récupérer les clés PUBLIQUES

Dans le projet : **Project Settings** (roue dentée) → **API**.

- Copie **Project URL** → ce sera `VITE_SUPABASE_URL`.
- Copie **anon public** (clé `anon`) → ce sera `VITE_SUPABASE_ANON_KEY`.

⚠️ Ne copie **jamais** la clé `service_role` dans l'app. Elle reste côté serveur.

---

## Étape 3 — Appliquer le schéma (SQL)

1. Menu de gauche → **SQL Editor** → **New query**.
2. Ouvre le fichier `supabase/schema_complet.sql` du dépôt, **copie tout**, colle dans l'éditeur.
3. Clique **Run**.
4. Résultat attendu : `Success. No rows returned`. (Si une erreur apparaît, envoie-la moi telle quelle.)

> Ce fichier contient les 8 migrations dans l'ordre : tables, vues de calcul,
> audit/immuabilité, RLS, réseaux de référence, liens créances, RPC de soumission,
> bucket photos.

---

## Étape 4 — Déployer l'Edge Function `creer-agent`

**Option A — Dashboard (le plus simple) :**
1. Menu → **Edge Functions** → **Deploy a new function**.
2. Nom : `creer-agent`.
3. Colle le contenu de `supabase/functions/creer-agent/index.ts`.
4. **Deploy**.

Les variables `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`
sont fournies automatiquement à la fonction : rien à configurer.

**Option B — CLI (si tu préfères) :**
```bash
npm i -g supabase
supabase login
supabase link --project-ref <ref-du-projet>   # visible dans l'URL du dashboard
supabase functions deploy creer-agent
```

---

## Étape 5 — Créer le premier ADMIN (toi)

L'app crée les agents, mais le tout premier administrateur se crée à la main.

1. Menu → **Authentication** → **Users** → **Add user** → **Create new user**.
   - Email : `admin@pdv.local` (ou ton e-mail).
   - Password : un mot de passe fort.
   - Coche **Auto Confirm User**.
2. Copie l'**UID** de l'utilisateur créé (colonne ID).
3. **SQL Editor** → nouvelle requête → colle en remplaçant l'UID :

```sql
insert into public.utilisateurs (id, nom_complet, role, actif)
values ('COLLE-ICI-L-UID', 'Propriétaire', 'ADMIN', true);
```

4. **Run**.

---

## Étape 6 — Créer tes points de vente

**SQL Editor** :

```sql
insert into public.pdv (nom, adresse) values
  ('PDV 1', 'Adresse 1'),
  ('PDV 2', 'Adresse 2'),
  ('PDV 3', 'Adresse 3'),
  ('PDV 4', 'Adresse 4');
```

(Tu pourras gérer les PDV depuis l'app plus tard ; pour l'instant on les pose ici.)

---

## Étape 7 — Lancer l'application

Dans le dossier `mobile-money-pdv/` :

```bash
cp .env.example .env
# édite .env : colle VITE_SUPABASE_URL et VITE_SUPABASE_ANON_KEY (étape 2)
npm install
npm run dev
```

Ouvre l'URL affichée (http://localhost:5173). Connecte-toi avec l'ADMIN de
l'étape 5. Tu dois voir l'écran **Clôtures** (vide) et l'onglet **Comptes**.

---

## Étape 8 — Vérification de bout en bout (le vrai test)

1. Dans **Comptes**, crée un agent et affecte-le à *PDV 1*.
2. Déconnecte-toi, reconnecte-toi **en tant que cet agent**.
3. Saisis exactement le cas d'acceptation (section 8) :
   - Orange : début 850 000, appro 500 000, fin 410 000
   - MTN : début 300 000, appro 0, fin 355 000
   - Moov : début 150 000, appro 0, fin 120 000
   - Wave : début 600 000, appro 200 000, fin 430 000
   - Espèces début 120 000 · Sorties 700 000 · une dépense 15 000 · une créance 25 000
   - Espèces comptées (fin) : **665 000**
4. L'écran doit afficher **Espèces attendues = 665 000** et **Écart = 0** (aperçu client).
5. Prends une photo (n'importe quelle image), **Soumettre**.
6. Reconnecte-toi en **ADMIN** → **Clôtures** : la clôture apparaît avec
   **Écart = 0** — cette valeur vient de la **vue serveur**, pas du téléphone.
   Si serveur et client concordent à 0, la chaîne complète est validée.

### Test hors ligne (facultatif)
Coupe le réseau (mode avion), saisis une clôture : elle s'enregistre, l'indicateur
passe à **En attente (1)**. Rallume le réseau : il repasse à **Synchronisé**.

---

## En cas de problème
Copie-moi le message d'erreur exact (SQL, console du navigateur, ou réponse de
l'Edge Function) et je te dis quoi corriger.
