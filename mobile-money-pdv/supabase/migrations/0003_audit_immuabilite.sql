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
