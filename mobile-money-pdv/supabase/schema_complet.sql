-- =====================================================================
-- schema_complet.sql — CONCATÉNATION des migrations 0001 à 0008, dans l'ordre.
-- Fichier généré pour un collage unique dans l'éditeur SQL de Supabase.
-- Source de vérité = migrations/. Ne pas éditer à la main.
-- =====================================================================


-- >>>>>>>>>>>>>>>>>>>> migrations/0001_schema.sql <<<<<<<<<<<<<<<<<<<<

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

-- >>>>>>>>>>>>>>>>>>>> migrations/0002_vues_calculs.sql <<<<<<<<<<<<<<<<<<<<

-- =====================================================================
-- 0002_vues_calculs.sql — Champs calculés en base (jamais stockés)
--
-- flux_net_uv, especes_fin_attendue, ecart, continuité UV et espèces
-- sont recalculés ici à partir des seules valeurs saisies. C'est cette
-- valeur serveur qui fait foi ; le calcul JS client n'est qu'un aperçu.
-- Les formules reproduisent exactement src/domain/calculs.ts.
-- =====================================================================

-- Tolérance globale paramétrable par l'admin (défaut 1000 FCFA).
create table if not exists public.parametres (
  cle    text primary key,
  valeur text not null
);
insert into public.parametres (cle, valeur)
values ('tolerance_ecart', '1000')
on conflict (cle) do nothing;

create or replace function public.tolerance_ecart()
returns bigint
language sql
stable
as $$
  select coalesce((select valeur::bigint from public.parametres where cle = 'tolerance_ecart'), 1000);
$$;

-- ---------------------------------------------------------------------
-- Agrégats UV par clôture : flux_net_uv = Σ (uv_debut + appro - uv_fin)
-- ---------------------------------------------------------------------
create or replace view public.v_cloture_flux_uv as
select
  cs.cloture_id,
  sum(cs.uv_debut + cs.approvisionnement - cs.uv_fin)::bigint as flux_net_uv
from public.cloture_soldes cs
group by cs.cloture_id;

-- ---------------------------------------------------------------------
-- Calcul complet d'une clôture :
--   especes_fin_attendue = especes_debut + flux_net_uv + apports
--                          - sorties - depenses - creances_creees
--                          + creances_remboursees
--   ecart = especes_fin_constatee - especes_fin_attendue
--
-- Rattachement des créances à la clôture par (pdv, date) :
--   * créées         : creances.date_creation      = clotures.date_cloture
--   * remboursées    : creances.date_remboursement = clotures.date_cloture
-- ---------------------------------------------------------------------
create or replace view public.v_cloture_calculs as
with dep as (
  select cloture_id, coalesce(sum(montant), 0)::bigint as total_depenses
  from public.depenses
  group by cloture_id
)
select
  c.id                     as cloture_id,
  c.pdv_id,
  c.date_cloture,
  coalesce(f.flux_net_uv, 0)                                   as flux_net_uv,
  coalesce(d.total_depenses, 0)                                as total_depenses,
  coalesce(cc.total, 0)                                        as total_creances_creees,
  coalesce(cr.total, 0)                                        as total_creances_remboursees,
  (
    c.especes_debut
    + coalesce(f.flux_net_uv, 0)
    + c.apports_especes
    - c.sorties_especes
    - coalesce(d.total_depenses, 0)
    - coalesce(cc.total, 0)
    + coalesce(cr.total, 0)
  )::bigint                                                    as especes_fin_attendue,
  (
    c.especes_fin_constatee
    - (
        c.especes_debut
        + coalesce(f.flux_net_uv, 0)
        + c.apports_especes
        - c.sorties_especes
        - coalesce(d.total_depenses, 0)
        - coalesce(cc.total, 0)
        + coalesce(cr.total, 0)
      )
  )::bigint                                                    as ecart
from public.clotures c
left join public.v_cloture_flux_uv f on f.cloture_id = c.id
left join dep d on d.cloture_id = c.id
left join lateral (
  select coalesce(sum(montant), 0)::bigint as total
  from public.creances
  where pdv_id = c.pdv_id and date_creation = c.date_cloture
) cc on true
left join lateral (
  select coalesce(sum(montant), 0)::bigint as total
  from public.creances
  where pdv_id = c.pdv_id and date_remboursement = c.date_cloture
) cr on true;

-- ---------------------------------------------------------------------
-- Continuité UV réseau par réseau : uv_debut(J) - uv_fin(J-1)
-- Compare l'ouverture d'une clôture à la clôture PRÉCÉDENTE du même PDV.
-- ---------------------------------------------------------------------
create or replace view public.v_continuite_uv as
select
  cs.cloture_id,
  cs.reseau_id,
  cs.uv_debut - coalesce(prev.uv_fin, 0) as continuite_uv
from public.cloture_soldes cs
join public.clotures c on c.id = cs.cloture_id
left join lateral (
  select cs2.uv_fin
  from public.clotures c2
  join public.cloture_soldes cs2
    on cs2.cloture_id = c2.id and cs2.reseau_id = cs.reseau_id
  where c2.pdv_id = c.pdv_id
    and c2.date_cloture < c.date_cloture
  order by c2.date_cloture desc
  limit 1
) prev on true;

-- ---------------------------------------------------------------------
-- Continuité espèces : especes_debut(J) - especes_fin_constatee(J-1)
-- ---------------------------------------------------------------------
create or replace view public.v_continuite_especes as
select
  c.id as cloture_id,
  c.pdv_id,
  c.date_cloture,
  c.especes_debut - coalesce(prev.especes_fin_constatee, 0) as continuite_especes
from public.clotures c
left join lateral (
  select c2.especes_fin_constatee
  from public.clotures c2
  where c2.pdv_id = c.pdv_id
    and c2.date_cloture < c.date_cloture
  order by c2.date_cloture desc
  limit 1
) prev on true;

-- ---------------------------------------------------------------------
-- Bilan mensuel par PDV (section 2.5) — le produit vient des COMMISSIONS,
-- pas du volume d'UV.
--   resultat_net = commissions_recues - charges_directes
--                  - quoteparts_communes - creances_non_soldees
-- ---------------------------------------------------------------------
create or replace view public.v_bilan_mensuel as
with comm as (
  select mois, pdv_id, sum(commission_recue)::bigint as commissions_recues
  from public.commissions
  group by mois, pdv_id
),
directes as (
  select mois, pdv_id, sum(montant)::bigint as charges_directes
  from public.charges_fixes
  where pdv_id is not null
  group by mois, pdv_id
),
quoteparts as (
  select cf.mois, cr.pdv_id,
         sum(round(cf.montant * cr.pourcentage / 100.0))::bigint as quoteparts_communes
  from public.charges_fixes cf
  join public.cles_repartition cr on cr.type_charge = cf.type_charge
  where cf.pdv_id is null                    -- charges communes uniquement
  group by cf.mois, cr.pdv_id
),
creances_ouvertes as (
  select pdv_id, to_char(date_creation, 'YYYY-MM') as mois,
         sum(montant)::bigint as creances_non_soldees
  from public.creances
  where statut <> 'SOLDEE'
  group by pdv_id, to_char(date_creation, 'YYYY-MM')
)
select
  p.id as pdv_id,
  m.mois,
  coalesce(comm.commissions_recues, 0)   as commissions_recues,
  coalesce(directes.charges_directes, 0) as charges_directes,
  coalesce(quoteparts.quoteparts_communes, 0) as quoteparts_communes,
  coalesce(creances_ouvertes.creances_non_soldees, 0) as creances_non_soldees,
  (
    coalesce(comm.commissions_recues, 0)
    - coalesce(directes.charges_directes, 0)
    - coalesce(quoteparts.quoteparts_communes, 0)
    - coalesce(creances_ouvertes.creances_non_soldees, 0)
  )::bigint as resultat_net
from public.pdv p
cross join (select distinct mois from public.charges_fixes
            union select distinct mois from public.commissions) m
left join comm on comm.pdv_id = p.id and comm.mois = m.mois
left join directes on directes.pdv_id = p.id and directes.mois = m.mois
left join quoteparts on quoteparts.pdv_id = p.id and quoteparts.mois = m.mois
left join creances_ouvertes on creances_ouvertes.pdv_id = p.id and creances_ouvertes.mois = m.mois;

-- >>>>>>>>>>>>>>>>>>>> migrations/0003_audit_immuabilite.sql <<<<<<<<<<<<<<<<<<<<

-- =====================================================================
-- 0003_audit_immuabilite.sql — Intégrité des données (non négociable)
--
--   * Journal d'audit append-only (insert-only).
--   * Clôture immuable après soumission : correction par écriture d'annulation,
--     jamais par modification en place. Seule la validation admin est permise.
--   * Photo obligatoire à la soumission.
--   * Contrôle des 100 % sur les clés de répartition.
--   * Horodatage serveur forcé (jamais l'heure du téléphone).
-- =====================================================================

-- ---------------------------------------------------------------------
-- Journal d'audit : interdit UPDATE et DELETE (append-only)
-- ---------------------------------------------------------------------
create or replace function public.journal_audit_append_only()
returns trigger
language plpgsql
as $$
begin
  raise exception 'journal_audit est append-only : ni modification ni suppression autorisée';
end;
$$;

drop trigger if exists trg_audit_no_update on public.journal_audit;
create trigger trg_audit_no_update
  before update or delete on public.journal_audit
  for each row execute function public.journal_audit_append_only();

-- ---------------------------------------------------------------------
-- Écriture générique dans le journal d'audit
-- ---------------------------------------------------------------------
create or replace function public.tracer_audit()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  v_id text;
begin
  if (tg_op = 'DELETE') then
    v_id := coalesce((to_jsonb(old) ->> 'id'), '');
    insert into public.journal_audit (utilisateur_id, table_cible, enregistrement_id, action, valeur_avant, valeur_apres)
    values (auth.uid(), tg_table_name, v_id, tg_op, to_jsonb(old), null);
    return old;
  else
    v_id := coalesce((to_jsonb(new) ->> 'id'), '');
    insert into public.journal_audit (utilisateur_id, table_cible, enregistrement_id, action, valeur_avant, valeur_apres)
    values (auth.uid(), tg_table_name, v_id, tg_op,
            case when tg_op = 'UPDATE' then to_jsonb(old) else null end,
            to_jsonb(new));
    return new;
  end if;
end;
$$;

drop trigger if exists trg_audit_clotures on public.clotures;
create trigger trg_audit_clotures
  after insert or update or delete on public.clotures
  for each row execute function public.tracer_audit();

drop trigger if exists trg_audit_cloture_soldes on public.cloture_soldes;
create trigger trg_audit_cloture_soldes
  after insert or update or delete on public.cloture_soldes
  for each row execute function public.tracer_audit();

drop trigger if exists trg_audit_creances on public.creances;
create trigger trg_audit_creances
  after insert or update or delete on public.creances
  for each row execute function public.tracer_audit();

drop trigger if exists trg_audit_depenses on public.depenses;
create trigger trg_audit_depenses
  after insert or update or delete on public.depenses
  for each row execute function public.tracer_audit();

-- ---------------------------------------------------------------------
-- Horodatage serveur forcé sur les clôtures.
-- On ignore toute valeur d'horodatage_serveur venue du client.
-- soumise_le est posée par le serveur au passage en SOUMISE.
-- ---------------------------------------------------------------------
create or replace function public.clotures_horodatage_serveur()
returns trigger
language plpgsql
as $$
begin
  if (tg_op = 'INSERT') then
    new.horodatage_serveur := now();
  end if;

  -- Passage à SOUMISE : le serveur pose soumise_le, jamais le client.
  if (new.statut = 'SOUMISE') and (tg_op = 'INSERT' or old.statut <> 'SOUMISE') then
    new.soumise_le := now();
    if (new.photo_url is null or length(trim(new.photo_url)) = 0) then
      raise exception 'Photo obligatoire : la clôture ne peut être soumise sans photo du solde du terminal';
    end if;
  end if;

  -- Validation / contestation : le serveur pose validee_le.
  if (new.statut in ('VALIDEE', 'CONTESTEE')) and (tg_op = 'UPDATE') and (old.statut <> new.statut) then
    new.validee_le := now();
  end if;

  return new;
end;
$$;

drop trigger if exists trg_clotures_horodatage on public.clotures;
create trigger trg_clotures_horodatage
  before insert or update on public.clotures
  for each row execute function public.clotures_horodatage_serveur();

-- ---------------------------------------------------------------------
-- Immuabilité : une clôture non BROUILLON ne peut plus voir ses valeurs
-- financières modifiées. Seule la transition de statut (validation admin)
-- est permise. Toute correction se fait par une écriture d'annulation.
-- ---------------------------------------------------------------------
create or replace function public.clotures_immuables()
returns trigger
language plpgsql
as $$
begin
  if (old.statut <> 'BROUILLON') then
    if (new.especes_debut, new.especes_fin_constatee, new.apports_especes,
        new.sorties_especes, new.pdv_id, new.agent_id, new.date_cloture, new.photo_url)
       is distinct from
       (old.especes_debut, old.especes_fin_constatee, old.apports_especes,
        old.sorties_especes, old.pdv_id, old.agent_id, old.date_cloture, old.photo_url)
    then
      raise exception 'Clôture % non modifiable (statut %). Corriger par une écriture d''annulation tracée.', old.id, old.statut;
    end if;
  end if;
  return new;
end;
$$;

drop trigger if exists trg_clotures_immuables on public.clotures;
create trigger trg_clotures_immuables
  before update on public.clotures
  for each row execute function public.clotures_immuables();

-- Les soldes UV d'une clôture soumise sont figés aussi.
create or replace function public.cloture_soldes_immuables()
returns trigger
language plpgsql
as $$
declare
  v_statut public.statut_cloture;
begin
  select statut into v_statut from public.clotures
   where id = coalesce(new.cloture_id, old.cloture_id);
  if (v_statut is not null and v_statut <> 'BROUILLON') then
    raise exception 'Soldes UV figés : la clôture n''est plus en BROUILLON';
  end if;
  return coalesce(new, old);
end;
$$;

drop trigger if exists trg_cloture_soldes_immuables on public.cloture_soldes;
create trigger trg_cloture_soldes_immuables
  before insert or update or delete on public.cloture_soldes
  for each row execute function public.cloture_soldes_immuables();

-- ---------------------------------------------------------------------
-- Contrôle des 100 % sur les clés de répartition (section 2.6).
-- Pour un type de charge donné, la somme des pourcentages doit faire 100.
-- Vérifié en contrainte différée à la fin de la transaction.
-- ---------------------------------------------------------------------
create or replace function public.verifier_cles_repartition()
returns trigger
language plpgsql
as $$
declare
  v_type text;
  v_somme numeric;
begin
  v_type := coalesce(new.type_charge, old.type_charge);
  select coalesce(sum(pourcentage), 0) into v_somme
  from public.cles_repartition
  where type_charge = v_type;

  -- On tolère l'état vide (aucune clé pour ce type). Sinon = 100 exact.
  if (v_somme <> 0 and abs(v_somme - 100) > 0.001) then
    raise exception 'Clé de répartition « % » invalide : somme = % (doit être 100).', v_type, v_somme;
  end if;
  return null;
end;
$$;

drop trigger if exists trg_cles_repartition_100 on public.cles_repartition;
create constraint trigger trg_cles_repartition_100
  after insert or update or delete on public.cles_repartition
  deferrable initially deferred
  for each row execute function public.verifier_cles_repartition();

-- >>>>>>>>>>>>>>>>>>>> migrations/0004_rls.sql <<<<<<<<<<<<<<<<<<<<

-- =====================================================================
-- 0004_rls.sql — Row-Level Security
--
-- Le contrôle d'accès est appliqué EN BASE, pas seulement dans l'interface.
-- Garantie clé : un AGENT ne voit jamais les commissions, les charges fixes,
-- les clés de répartition, ni la rentabilité — même en forgeant des requêtes.
-- =====================================================================

-- ---------------------------------------------------------------------
-- Fonctions d'aide
-- ---------------------------------------------------------------------
create or replace function public.est_admin()
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1 from public.utilisateurs
    where id = auth.uid() and role = 'ADMIN' and actif
  );
$$;

-- L'agent a accès à un PDV s'il y a une affectation active (date_fin null ou future).
create or replace function public.agent_a_acces_pdv(p_pdv_id uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1 from public.affectations a
    where a.utilisateur_id = auth.uid()
      and a.pdv_id = p_pdv_id
      and a.date_debut <= (now() at time zone 'UTC')::date
      and (a.date_fin is null or a.date_fin >= (now() at time zone 'UTC')::date)
  );
$$;

-- ---------------------------------------------------------------------
-- Activation RLS sur toutes les tables
-- ---------------------------------------------------------------------
alter table public.pdv                 enable row level security;
alter table public.utilisateurs        enable row level security;
alter table public.affectations        enable row level security;
alter table public.reseaux             enable row level security;
alter table public.produits_financiers enable row level security;
alter table public.clotures            enable row level security;
alter table public.cloture_soldes      enable row level security;
alter table public.depenses            enable row level security;
alter table public.creances            enable row level security;
alter table public.incidents           enable row level security;
alter table public.charges_fixes       enable row level security;
alter table public.cles_repartition    enable row level security;
alter table public.commissions         enable row level security;
alter table public.journal_audit       enable row level security;
alter table public.parametres          enable row level security;

-- ---------------------------------------------------------------------
-- PDV : admin tout ; agent lecture de ses PDV affectés
-- ---------------------------------------------------------------------
create policy pdv_admin_all on public.pdv
  for all using (public.est_admin()) with check (public.est_admin());
create policy pdv_agent_read on public.pdv
  for select using (public.agent_a_acces_pdv(id));

-- ---------------------------------------------------------------------
-- Utilisateurs : admin tout ; agent lecture de son propre profil
-- ---------------------------------------------------------------------
create policy utilisateurs_admin_all on public.utilisateurs
  for all using (public.est_admin()) with check (public.est_admin());
create policy utilisateurs_self_read on public.utilisateurs
  for select using (id = auth.uid());

-- ---------------------------------------------------------------------
-- Affectations : admin tout ; agent lecture des siennes
-- ---------------------------------------------------------------------
create policy affectations_admin_all on public.affectations
  for all using (public.est_admin()) with check (public.est_admin());
create policy affectations_agent_read on public.affectations
  for select using (utilisateur_id = auth.uid());

-- ---------------------------------------------------------------------
-- Réseaux & produits : lecture pour tous les authentifiés ; écriture admin
-- (les barèmes de commission ne révèlent pas la rentabilité)
-- ---------------------------------------------------------------------
create policy reseaux_read on public.reseaux
  for select using (auth.uid() is not null);
create policy reseaux_admin_write on public.reseaux
  for all using (public.est_admin()) with check (public.est_admin());

create policy produits_read on public.produits_financiers
  for select using (auth.uid() is not null);
create policy produits_admin_write on public.produits_financiers
  for all using (public.est_admin()) with check (public.est_admin());

-- ---------------------------------------------------------------------
-- Clôtures : admin tout ; agent voit/saisit celles de ses PDV.
-- L'agent ne peut modifier que ses BROUILLONS (immuabilité renforcée en 0003).
-- ---------------------------------------------------------------------
create policy clotures_admin_all on public.clotures
  for all using (public.est_admin()) with check (public.est_admin());

create policy clotures_agent_select on public.clotures
  for select using (public.agent_a_acces_pdv(pdv_id));

create policy clotures_agent_insert on public.clotures
  for insert with check (
    public.agent_a_acces_pdv(pdv_id)
    and agent_id = auth.uid()
    and statut in ('BROUILLON', 'SOUMISE')
  );

create policy clotures_agent_update on public.clotures
  for update using (
    public.agent_a_acces_pdv(pdv_id)
    and agent_id = auth.uid()
    and statut = 'BROUILLON'
  ) with check (
    public.agent_a_acces_pdv(pdv_id)
    and agent_id = auth.uid()
    and statut in ('BROUILLON', 'SOUMISE')
  );

-- ---------------------------------------------------------------------
-- Soldes UV : suivent l'accès à la clôture parente
-- ---------------------------------------------------------------------
create policy cloture_soldes_admin_all on public.cloture_soldes
  for all using (public.est_admin()) with check (public.est_admin());

create policy cloture_soldes_agent_rw on public.cloture_soldes
  for all using (
    exists (select 1 from public.clotures c
            where c.id = cloture_id and public.agent_a_acces_pdv(c.pdv_id))
  ) with check (
    exists (select 1 from public.clotures c
            where c.id = cloture_id and c.agent_id = auth.uid())
  );

-- ---------------------------------------------------------------------
-- Dépenses & créances : admin tout ; agent sur ses PDV
-- ---------------------------------------------------------------------
create policy depenses_admin_all on public.depenses
  for all using (public.est_admin()) with check (public.est_admin());
create policy depenses_agent_rw on public.depenses
  for all using (public.agent_a_acces_pdv(pdv_id))
  with check (public.agent_a_acces_pdv(pdv_id));

create policy creances_admin_all on public.creances
  for all using (public.est_admin()) with check (public.est_admin());
create policy creances_agent_rw on public.creances
  for all using (public.agent_a_acces_pdv(pdv_id))
  with check (public.agent_a_acces_pdv(pdv_id) and agent_id = auth.uid());

-- ---------------------------------------------------------------------
-- Incidents : admin tout ; agent crée/lit les siens
-- ---------------------------------------------------------------------
create policy incidents_admin_all on public.incidents
  for all using (public.est_admin()) with check (public.est_admin());
create policy incidents_agent_rw on public.incidents
  for all using (public.agent_a_acces_pdv(pdv_id))
  with check (public.agent_a_acces_pdv(pdv_id) and agent_id = auth.uid());

-- ---------------------------------------------------------------------
-- DONNÉES SENSIBLES — ADMIN UNIQUEMENT (aucune policy agent)
-- charges_fixes, cles_repartition, commissions, parametres
-- Sans policy de lecture pour l'agent, RLS renvoie zéro ligne.
-- ---------------------------------------------------------------------
create policy charges_admin_all on public.charges_fixes
  for all using (public.est_admin()) with check (public.est_admin());
create policy cles_admin_all on public.cles_repartition
  for all using (public.est_admin()) with check (public.est_admin());
create policy commissions_admin_all on public.commissions
  for all using (public.est_admin()) with check (public.est_admin());

create policy parametres_read on public.parametres
  for select using (auth.uid() is not null);   -- tolérance lisible par l'agent
create policy parametres_admin_write on public.parametres
  for all using (public.est_admin()) with check (public.est_admin());

-- ---------------------------------------------------------------------
-- Journal d'audit : lecture admin uniquement ; insertion via triggers
-- (security definer). Aucune écriture directe par le client.
-- ---------------------------------------------------------------------
create policy audit_admin_read on public.journal_audit
  for select using (public.est_admin());

-- ---------------------------------------------------------------------
-- Vues : security_invoker pour que la RLS des tables sous-jacentes
-- s'applique à l'appelant (sinon un agent verrait le bilan).
-- ---------------------------------------------------------------------
alter view public.v_cloture_flux_uv     set (security_invoker = true);
alter view public.v_cloture_calculs      set (security_invoker = true);
alter view public.v_continuite_uv        set (security_invoker = true);
alter view public.v_continuite_especes   set (security_invoker = true);
alter view public.v_bilan_mensuel        set (security_invoker = true);

-- >>>>>>>>>>>>>>>>>>>> migrations/0005_reference.sql <<<<<<<<<<<<<<<<<<<<

-- =====================================================================
-- 0005_reference.sql — Données DE RÉFÉRENCE (pas des données de démo)
--
-- Les quatre réseaux exploités. Nécessaires au fonctionnement, idempotents.
-- Aucune donnée de démonstration (PDV, agents, clôtures) n'est insérée ici :
-- interdit en production.
-- =====================================================================

insert into public.reseaux (nom, actif) values
  ('Orange Money', true),
  ('MTN MoMo', true),
  ('Moov Money', true),
  ('Wave', true)
on conflict (nom) do nothing;

-- >>>>>>>>>>>>>>>>>>>> migrations/0006_creances_lien_cloture_et_controles.sql <<<<<<<<<<<<<<<<<<<<

-- =====================================================================
-- 0006 — Rattachement explicite des créances aux clôtures (décision ①),
-- gel des enfants d'une clôture verrouillée, et contrôle de continuité
-- des soldes d'ouverture (décision ③).
-- =====================================================================

-- ---------------------------------------------------------------------
-- ① Liens explicites créance <-> clôture
--    cloture_id                : clôture de CRÉATION (UV sortie, réduit l'espèce
--                                attendue de cette clôture)
--    cloture_remboursement_id  : clôture où l'espèce du REMBOURSEMENT entre
--                                (augmente l'espèce attendue de cette clôture)
--    on delete set null : la créance survit à la suppression d'une clôture
--                         (ancienneté, relances — Phase 2).
-- ---------------------------------------------------------------------
alter table public.creances
  add column if not exists cloture_id uuid references public.clotures (id) on delete set null,
  add column if not exists cloture_remboursement_id uuid references public.clotures (id) on delete set null;

create index if not exists idx_creances_cloture on public.creances (cloture_id);
create index if not exists idx_creances_cloture_remb on public.creances (cloture_remboursement_id);

-- ---------------------------------------------------------------------
-- Vue de calcul revue : les créances sont désormais rattachées par lien
-- explicite, plus par date. Formule inchangée (identique à src/domain/calculs.ts).
-- ---------------------------------------------------------------------
create or replace view public.v_cloture_calculs as
with dep as (
  select cloture_id, coalesce(sum(montant), 0)::bigint as total_depenses
  from public.depenses
  where cloture_id is not null
  group by cloture_id
),
cc as (
  select cloture_id, coalesce(sum(montant), 0)::bigint as total
  from public.creances
  where cloture_id is not null
  group by cloture_id
),
cr as (
  select cloture_remboursement_id as cloture_id, coalesce(sum(montant), 0)::bigint as total
  from public.creances
  where cloture_remboursement_id is not null
  group by cloture_remboursement_id
)
select
  c.id                     as cloture_id,
  c.pdv_id,
  c.date_cloture,
  coalesce(f.flux_net_uv, 0)  as flux_net_uv,
  coalesce(d.total_depenses, 0) as total_depenses,
  coalesce(cc.total, 0)       as total_creances_creees,
  coalesce(cr.total, 0)       as total_creances_remboursees,
  (
    c.especes_debut
    + coalesce(f.flux_net_uv, 0)
    + c.apports_especes
    - c.sorties_especes
    - coalesce(d.total_depenses, 0)
    - coalesce(cc.total, 0)
    + coalesce(cr.total, 0)
  )::bigint                   as especes_fin_attendue,
  (
    c.especes_fin_constatee
    - (
        c.especes_debut
        + coalesce(f.flux_net_uv, 0)
        + c.apports_especes
        - c.sorties_especes
        - coalesce(d.total_depenses, 0)
        - coalesce(cc.total, 0)
        + coalesce(cr.total, 0)
      )
  )::bigint                   as ecart
from public.clotures c
left join public.v_cloture_flux_uv f on f.cloture_id = c.id
left join dep d on d.cloture_id = c.id
left join cc on cc.cloture_id = c.id
left join cr on cr.cloture_id = c.id;

alter view public.v_cloture_calculs set (security_invoker = true);

-- ---------------------------------------------------------------------
-- Gel des dépenses rattachées à une clôture non BROUILLON.
-- (Sinon ajouter/retirer une dépense après soumission ferait dériver
--  l'écart déjà soumis.)
-- ---------------------------------------------------------------------
create or replace function public.depenses_integrite()
returns trigger
language plpgsql
as $$
declare
  v_statut public.statut_cloture;
begin
  -- Clôture visée par la ligne (nouvelle ou ancienne)
  if (tg_op <> 'DELETE') and new.cloture_id is not null then
    select statut into v_statut from public.clotures where id = new.cloture_id;
    if v_statut is not null and v_statut <> 'BROUILLON' then
      raise exception 'Dépense figée : la clôture % n''est plus en BROUILLON', new.cloture_id;
    end if;
  end if;
  if (tg_op <> 'INSERT') and old.cloture_id is not null then
    select statut into v_statut from public.clotures where id = old.cloture_id;
    if v_statut is not null and v_statut <> 'BROUILLON' then
      raise exception 'Dépense figée : la clôture d''origine % n''est plus en BROUILLON', old.cloture_id;
    end if;
  end if;
  return coalesce(new, old);
end;
$$;

drop trigger if exists trg_depenses_integrite on public.depenses;
create trigger trg_depenses_integrite
  before insert or update or delete on public.depenses
  for each row execute function public.depenses_integrite();

-- ---------------------------------------------------------------------
-- Intégrité des créances vis-à-vis des clôtures :
--   * création : la clôture de création doit être BROUILLON ;
--   * une fois la clôture de création verrouillée, les champs de création
--     sont figés — seul l'enregistrement du remboursement reste possible ;
--   * un remboursement doit viser une clôture BROUILLON (celle du jour où
--     l'espèce entre).
-- ---------------------------------------------------------------------
create or replace function public.creances_integrite()
returns trigger
language plpgsql
as $$
declare
  v_statut public.statut_cloture;
begin
  if tg_op = 'INSERT' then
    if new.cloture_id is not null then
      select statut into v_statut from public.clotures where id = new.cloture_id;
      if v_statut is not null and v_statut <> 'BROUILLON' then
        raise exception 'Créance : la clôture de création % n''est plus en BROUILLON', new.cloture_id;
      end if;
    end if;
    if new.cloture_remboursement_id is not null then
      select statut into v_statut from public.clotures where id = new.cloture_remboursement_id;
      if v_statut is not null and v_statut <> 'BROUILLON' then
        raise exception 'Créance : le remboursement doit être rattaché à une clôture en BROUILLON';
      end if;
    end if;
    return new;
  end if;

  if tg_op = 'DELETE' then
    if old.cloture_id is not null then
      select statut into v_statut from public.clotures where id = old.cloture_id;
      if v_statut is not null and v_statut <> 'BROUILLON' then
        raise exception 'Créance non supprimable : clôture de création % verrouillée', old.cloture_id;
      end if;
    end if;
    return old;
  end if;

  -- UPDATE : si la clôture de création est verrouillée, on gèle les champs de création.
  if old.cloture_id is not null then
    select statut into v_statut from public.clotures where id = old.cloture_id;
    if v_statut is not null and v_statut <> 'BROUILLON' then
      if (new.montant, new.cloture_id, new.pdv_id, new.reseau_id, new.date_creation,
          new.client_nom, new.client_numero, new.autorise_par)
         is distinct from
         (old.montant, old.cloture_id, old.pdv_id, old.reseau_id, old.date_creation,
          old.client_nom, old.client_numero, old.autorise_par)
      then
        raise exception 'Créance figée (clôture de création verrouillée). Seul le remboursement peut être enregistré.';
      end if;
    end if;
  end if;

  -- Le remboursement doit viser une clôture BROUILLON.
  if new.cloture_remboursement_id is not null
     and new.cloture_remboursement_id is distinct from old.cloture_remboursement_id then
    select statut into v_statut from public.clotures where id = new.cloture_remboursement_id;
    if v_statut is not null and v_statut <> 'BROUILLON' then
      raise exception 'Remboursement : la clôture visée n''est plus en BROUILLON';
    end if;
  end if;

  return new;
end;
$$;

drop trigger if exists trg_creances_integrite on public.creances;
create trigger trg_creances_integrite
  before insert or update or delete on public.creances
  for each row execute function public.creances_integrite();

-- ---------------------------------------------------------------------
-- ③ Continuité des soldes d'ouverture (section 2.3).
-- À la soumission d'une clôture, l'ouverture doit correspondre à la clôture
-- précédente du même PDV (espèces ET UV réseau par réseau). Sinon, une
-- justification écrite (motif_derogation) est OBLIGATOIRE.
-- ---------------------------------------------------------------------
create or replace function public.clotures_continuite_ouverture()
returns trigger
language plpgsql
as $$
declare
  v_prev public.clotures%rowtype;
  r record;
  v_divergence boolean := false;
begin
  -- Ne s'applique qu'au passage effectif en SOUMISE.
  if new.statut <> 'SOUMISE' then
    return new;
  end if;
  if tg_op = 'UPDATE' and old.statut = 'SOUMISE' then
    return new;
  end if;

  select * into v_prev
  from public.clotures
  where pdv_id = new.pdv_id and date_cloture < new.date_cloture
  order by date_cloture desc
  limit 1;

  if not found then
    return new;  -- première clôture du PDV : aucune continuité à vérifier
  end if;

  -- Continuité des espèces
  if new.especes_debut <> v_prev.especes_fin_constatee then
    v_divergence := true;
  end if;

  -- Continuité des UV, réseau par réseau
  for r in
    select cs.uv_debut,
           (select cs2.uv_fin
              from public.cloture_soldes cs2
             where cs2.cloture_id = v_prev.id and cs2.reseau_id = cs.reseau_id) as uv_fin_prev
    from public.cloture_soldes cs
    where cs.cloture_id = new.id
  loop
    if r.uv_debut <> coalesce(r.uv_fin_prev, 0) then
      v_divergence := true;
    end if;
  end loop;

  if v_divergence and (new.motif_derogation is null or length(trim(new.motif_derogation)) = 0) then
    raise exception 'Ouverture différente de la clôture précédente : une justification écrite (motif_derogation) est obligatoire.';
  end if;

  return new;
end;
$$;

drop trigger if exists trg_clotures_continuite on public.clotures;
create trigger trg_clotures_continuite
  before insert or update on public.clotures
  for each row execute function public.clotures_continuite_ouverture();

-- >>>>>>>>>>>>>>>>>>>> migrations/0007_rpc_soumettre_cloture.sql <<<<<<<<<<<<<<<<<<<<

-- =====================================================================
-- 0007 — RPC de soumission atomique et idempotente d'une clôture.
--
-- La file de synchronisation hors ligne peut renvoyer la même clôture
-- plusieurs fois (réseau instable). Cette fonction :
--   * upsert la clôture + ses soldes + dépenses + créances dans UNE transaction ;
--   * pose le statut SOUMISE seulement après avoir inséré les enfants
--     (respecte les triggers d'immuabilité et de continuité) ;
--   * est idempotente : un renvoi à l'identique ne crée pas de doublon et ne
--     viole pas l'immuabilité (aucune valeur ne change).
--
-- security invoker : la RLS de l'appelant (l'agent) s'applique aux écritures.
-- =====================================================================
create or replace function public.soumettre_cloture(
  p_cloture   jsonb,
  p_soldes    jsonb,
  p_depenses  jsonb,
  p_creances  jsonb
)
returns void
language plpgsql
security invoker
as $$
declare
  v_id uuid := (p_cloture ->> 'id')::uuid;
begin
  -- 1) Clôture en BROUILLON (le statut n'est jamais rétrogradé sur conflit).
  insert into public.clotures (
    id, pdv_id, agent_id, date_cloture,
    especes_debut, especes_fin_constatee, apports_especes, sorties_especes,
    photo_url, motif_derogation, saisi_le_client, statut
  )
  values (
    v_id,
    (p_cloture ->> 'pdv_id')::uuid,
    (p_cloture ->> 'agent_id')::uuid,
    (p_cloture ->> 'date_cloture')::date,
    coalesce((p_cloture ->> 'especes_debut')::bigint, 0),
    coalesce((p_cloture ->> 'especes_fin_constatee')::bigint, 0),
    coalesce((p_cloture ->> 'apports_especes')::bigint, 0),
    coalesce((p_cloture ->> 'sorties_especes')::bigint, 0),
    p_cloture ->> 'photo_url',
    p_cloture ->> 'motif_derogation',
    nullif(p_cloture ->> 'saisi_le_client', '')::timestamptz,
    'BROUILLON'
  )
  on conflict (id) do update set
    especes_debut         = excluded.especes_debut,
    especes_fin_constatee = excluded.especes_fin_constatee,
    apports_especes       = excluded.apports_especes,
    sorties_especes       = excluded.sorties_especes,
    photo_url             = excluded.photo_url,
    motif_derogation      = excluded.motif_derogation;

  -- 2) Soldes UV par réseau.
  insert into public.cloture_soldes (id, cloture_id, reseau_id, uv_debut, approvisionnement, uv_fin)
  select
    (s ->> 'id')::uuid,
    v_id,
    (s ->> 'reseau_id')::uuid,
    coalesce((s ->> 'uv_debut')::bigint, 0),
    coalesce((s ->> 'approvisionnement')::bigint, 0),
    coalesce((s ->> 'uv_fin')::bigint, 0)
  from jsonb_array_elements(coalesce(p_soldes, '[]'::jsonb)) as s
  on conflict (id) do update set
    uv_debut          = excluded.uv_debut,
    approvisionnement = excluded.approvisionnement,
    uv_fin            = excluded.uv_fin;

  -- 3) Dépenses.
  insert into public.depenses (id, cloture_id, pdv_id, date, categorie, montant, description, autorise_par)
  select
    (d ->> 'id')::uuid,
    v_id,
    (d ->> 'pdv_id')::uuid,
    (d ->> 'date')::date,
    d ->> 'categorie',
    coalesce((d ->> 'montant')::bigint, 0),
    d ->> 'description',
    d ->> 'autorise_par'
  from jsonb_array_elements(coalesce(p_depenses, '[]'::jsonb)) as d
  on conflict (id) do update set
    montant     = excluded.montant,
    categorie   = excluded.categorie,
    description = excluded.description;

  -- 4) Créances (rattachées à la clôture de création).
  insert into public.creances (
    id, pdv_id, agent_id, date_creation, client_nom, client_numero,
    reseau_id, montant, motif, autorise_par, statut, cloture_id
  )
  select
    (cr ->> 'id')::uuid,
    (cr ->> 'pdv_id')::uuid,
    (cr ->> 'agent_id')::uuid,
    (cr ->> 'date_creation')::date,
    cr ->> 'client_nom',
    cr ->> 'client_numero',
    nullif(cr ->> 'reseau_id', '')::uuid,
    coalesce((cr ->> 'montant')::bigint, 0),
    cr ->> 'motif',
    cr ->> 'autorise_par',
    coalesce(cr ->> 'statut', 'EN_COURS')::public.statut_creance,
    v_id
  from jsonb_array_elements(coalesce(p_creances, '[]'::jsonb)) as cr
  on conflict (id) do update set
    montant       = excluded.montant,
    client_nom    = excluded.client_nom,
    client_numero = excluded.client_numero,
    motif         = excluded.motif;

  -- 5) Passage en SOUMISE seulement si encore en BROUILLON (idempotent).
  update public.clotures
     set statut = 'SOUMISE'
   where id = v_id and statut = 'BROUILLON';
end;
$$;

-- >>>>>>>>>>>>>>>>>>>> migrations/0008_storage_photos.sql <<<<<<<<<<<<<<<<<<<<

-- =====================================================================
-- 0008 — Stockage des photos de clôture (bucket + politiques).
--
-- La photo du solde du terminal est jointe à chaque clôture. On la stocke
-- dans un bucket dédié. Écriture réservée aux utilisateurs authentifiés ;
-- lecture réservée aux authentifiés (pas d'accès anonyme public).
-- =====================================================================

insert into storage.buckets (id, name, public)
values ('photos-clotures', 'photos-clotures', false)
on conflict (id) do nothing;

-- Écriture : tout utilisateur authentifié peut déposer/mettre à jour une photo
-- dans ce bucket (l'upload précède la soumission RPC).
drop policy if exists photos_insert on storage.objects;
create policy photos_insert on storage.objects
  for insert to authenticated
  with check (bucket_id = 'photos-clotures');

drop policy if exists photos_update on storage.objects;
create policy photos_update on storage.objects
  for update to authenticated
  using (bucket_id = 'photos-clotures')
  with check (bucket_id = 'photos-clotures');

-- Lecture : utilisateurs authentifiés uniquement.
drop policy if exists photos_select on storage.objects;
create policy photos_select on storage.objects
  for select to authenticated
  using (bucket_id = 'photos-clotures');

-- NOTE : le bucket étant privé, l'affichage admin de la photo doit utiliser une
-- URL signée (createSignedUrl) plutôt que getPublicUrl. À ajuster côté client
-- lors du branchement du vrai projet Supabase (voir supabase/README.md).
