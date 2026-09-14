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
