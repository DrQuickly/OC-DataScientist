-- =====================================================================
-- 0001_schema.sql — Schéma de base (Phase 1)
-- Gestion de points de vente Mobile Money.
--
-- Conventions :
--   * Noms de tables et colonnes en français, snake_case.
--   * Montants : entiers FCFA (XOF), sans décimales -> type integer/bigint.
--   * Fuseau : Africa/Abidjan (UTC+0). Les horodatages font foi côté serveur.
--   * Les identifiants de clôture sont des UUID générés côté client (idempotence
--     de la file de synchronisation hors ligne).
-- =====================================================================

create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------
-- Points de vente
-- ---------------------------------------------------------------------
create table if not exists public.pdv (
  id         uuid primary key default gen_random_uuid(),
  nom        text not null,
  adresse    text,
  actif      boolean not null default true,
  cree_le    timestamptz not null default now()
);

-- ---------------------------------------------------------------------
-- Utilisateurs (profil applicatif adossé à auth.users)
-- L'id est identique à auth.users.id. La création des agents se fait via
-- une Edge Function (service_role) : jamais depuis le client.
-- ---------------------------------------------------------------------
do $$ begin
  create type public.role_utilisateur as enum ('ADMIN', 'AGENT');
exception when duplicate_object then null; end $$;

create table if not exists public.utilisateurs (
  id           uuid primary key references auth.users (id) on delete cascade,
  nom_complet  text not null,
  telephone    text,
  role         public.role_utilisateur not null default 'AGENT',
  actif        boolean not null default true,
  cree_le      timestamptz not null default now()
);

-- ---------------------------------------------------------------------
-- Affectations agent <-> PDV (un agent peut être réaffecté d'un jour à l'autre)
-- ---------------------------------------------------------------------
create table if not exists public.affectations (
  id             uuid primary key default gen_random_uuid(),
  utilisateur_id uuid not null references public.utilisateurs (id) on delete cascade,
  pdv_id         uuid not null references public.pdv (id) on delete cascade,
  date_debut     date not null default (now() at time zone 'UTC')::date,
  date_fin       date,
  cree_le        timestamptz not null default now(),
  check (date_fin is null or date_fin >= date_debut)
);
create index if not exists idx_affectations_utilisateur on public.affectations (utilisateur_id);
create index if not exists idx_affectations_pdv on public.affectations (pdv_id);

-- ---------------------------------------------------------------------
-- Réseaux (Orange, MTN, Moov, Wave)
-- ---------------------------------------------------------------------
create table if not exists public.reseaux (
  id     uuid primary key default gen_random_uuid(),
  nom    text not null unique,
  actif  boolean not null default true
);

-- ---------------------------------------------------------------------
-- Produits financiers (barèmes de commission en jsonb, remplis au fil de l'eau)
-- ---------------------------------------------------------------------
create table if not exists public.produits_financiers (
  id                 uuid primary key default gen_random_uuid(),
  nom                text not null,
  reseau_id          uuid references public.reseaux (id) on delete set null,
  type               text,
  bareme_commission  jsonb not null default '{}'::jsonb,
  actif              boolean not null default true
);

-- ---------------------------------------------------------------------
-- Clôtures journalières
-- Une seule clôture par (pdv, date). Immuable après soumission (trigger 0003).
-- ---------------------------------------------------------------------
do $$ begin
  create type public.statut_cloture as enum ('BROUILLON', 'SOUMISE', 'VALIDEE', 'CONTESTEE');
exception when duplicate_object then null; end $$;

create table if not exists public.clotures (
  id                     uuid primary key,               -- UUID généré côté client
  pdv_id                 uuid not null references public.pdv (id),
  agent_id               uuid not null references public.utilisateurs (id),
  date_cloture           date not null,
  especes_debut          bigint not null default 0,
  especes_fin_constatee  bigint not null default 0,
  apports_especes        bigint not null default 0,   -- NON opérationnels (banque, proprio, transfert) — jamais un dépôt client
  sorties_especes        bigint not null default 0,   -- NON opérationnelles (banque, proprio, transfert) — jamais un retrait client
  photo_url              text,                            -- obligatoire à la soumission (trigger)
  statut                 public.statut_cloture not null default 'BROUILLON',
  motif_derogation       text,                            -- justification si ouverture modifiée
  saisi_le_client        timestamptz,                     -- heure du téléphone : INDICATIVE
  soumise_le             timestamptz,                     -- posée par le serveur
  validee_par            uuid references public.utilisateurs (id),
  validee_le             timestamptz,
  horodatage_serveur     timestamptz not null default now(),  -- fait foi
  check (especes_debut >= 0),
  check (especes_fin_constatee >= 0),
  check (apports_especes >= 0),
  check (sorties_especes >= 0),
  unique (pdv_id, date_cloture)
);
create index if not exists idx_clotures_pdv_date on public.clotures (pdv_id, date_cloture desc);
create index if not exists idx_clotures_agent on public.clotures (agent_id);

-- ---------------------------------------------------------------------
-- Soldes UV par réseau, rattachés à une clôture
-- ---------------------------------------------------------------------
create table if not exists public.cloture_soldes (
  id                uuid primary key default gen_random_uuid(),
  cloture_id        uuid not null references public.clotures (id) on delete cascade,
  reseau_id         uuid not null references public.reseaux (id),
  uv_debut          bigint not null default 0,
  approvisionnement bigint not null default 0,
  uv_fin            bigint not null default 0,
  appro_valide      boolean not null default false,   -- l'admin valide l'approvisionnement
  check (uv_debut >= 0),
  check (approvisionnement >= 0),
  check (uv_fin >= 0),
  unique (cloture_id, reseau_id)
);
create index if not exists idx_cloture_soldes_cloture on public.cloture_soldes (cloture_id);

-- ---------------------------------------------------------------------
-- Dépenses (charges de caisse) rattachées à une clôture
-- ---------------------------------------------------------------------
create table if not exists public.depenses (
  id                uuid primary key default gen_random_uuid(),
  cloture_id        uuid references public.clotures (id) on delete cascade,
  pdv_id            uuid not null references public.pdv (id),
  date              date not null,
  categorie         text,
  montant           bigint not null check (montant >= 0),
  description       text,
  justificatif_url  text,
  autorise_par      text,
  cree_le           timestamptz not null default now()
);
create index if not exists idx_depenses_cloture on public.depenses (cloture_id);

-- ---------------------------------------------------------------------
-- Créances (« dépôts en attente de paiement ») — poste de fuite principal
-- ---------------------------------------------------------------------
do $$ begin
  create type public.statut_creance as enum ('EN_COURS', 'A_RELANCER', 'ALERTE', 'SOLDEE');
exception when duplicate_object then null; end $$;

create table if not exists public.creances (
  id                  uuid primary key default gen_random_uuid(),
  pdv_id              uuid not null references public.pdv (id),
  agent_id            uuid not null references public.utilisateurs (id),
  date_creation       date not null,
  client_nom          text not null,
  client_numero       text not null,
  reseau_id           uuid references public.reseaux (id),
  montant             bigint not null check (montant > 0),
  motif               text,
  autorise_par        text not null,
  statut              public.statut_creance not null default 'EN_COURS',
  date_remboursement  date,
  cree_le             timestamptz not null default now()
);
create index if not exists idx_creances_pdv_date on public.creances (pdv_id, date_creation);

-- ---------------------------------------------------------------------
-- Incidents
-- ---------------------------------------------------------------------
do $$ begin
  create type public.statut_incident as enum ('OUVERT', 'EN_COURS', 'RESOLU');
exception when duplicate_object then null; end $$;

create table if not exists public.incidents (
  id             uuid primary key default gen_random_uuid(),
  pdv_id         uuid not null references public.pdv (id),
  agent_id       uuid not null references public.utilisateurs (id),
  date           date not null,
  categorie      text,
  gravite        text,
  description    text not null,
  photo_url      text,
  statut         public.statut_incident not null default 'OUVERT',
  reponse_admin  text,
  resolu_le      timestamptz,
  cree_le        timestamptz not null default now()
);

-- ---------------------------------------------------------------------
-- Charges fixes (pdv_id nullable = charge commune)
-- ---------------------------------------------------------------------
create table if not exists public.charges_fixes (
  id           uuid primary key default gen_random_uuid(),
  mois         text not null,                    -- format 'AAAA-MM'
  pdv_id       uuid references public.pdv (id),  -- null = charge commune
  type_charge  text not null,
  montant      bigint not null check (montant >= 0),
  paye         boolean not null default false,
  commentaire  text
);
create index if not exists idx_charges_mois on public.charges_fixes (mois);

-- ---------------------------------------------------------------------
-- Clés de répartition (par type de charge et par PDV, en pourcentage)
-- Contrainte des 100 % vérifiée par trigger (0003).
-- ---------------------------------------------------------------------
create table if not exists public.cles_repartition (
  type_charge  text not null,
  pdv_id       uuid not null references public.pdv (id) on delete cascade,
  pourcentage  numeric(6,3) not null check (pourcentage >= 0 and pourcentage <= 100),
  primary key (type_charge, pdv_id)
);

-- ---------------------------------------------------------------------
-- Commissions (estimées vs reçues)
-- ---------------------------------------------------------------------
create table if not exists public.commissions (
  id                  uuid primary key default gen_random_uuid(),
  mois                text not null,             -- 'AAAA-MM'
  pdv_id              uuid not null references public.pdv (id),
  reseau_id           uuid references public.reseaux (id),
  commission_estimee  bigint not null default 0,
  commission_recue    bigint not null default 0,
  unique (mois, pdv_id, reseau_id)
);

-- ---------------------------------------------------------------------
-- Journal d'audit append-only (voir 0003 pour le verrou insert-only)
-- ---------------------------------------------------------------------
create table if not exists public.journal_audit (
  id                  bigint generated always as identity primary key,
  utilisateur_id      uuid references public.utilisateurs (id),
  table_cible         text not null,
  enregistrement_id   text not null,
  action              text not null,             -- INSERT | UPDATE | DELETE
  valeur_avant        jsonb,
  valeur_apres        jsonb,
  horodatage_serveur  timestamptz not null default now()
);
create index if not exists idx_audit_cible on public.journal_audit (table_cible, enregistrement_id);
