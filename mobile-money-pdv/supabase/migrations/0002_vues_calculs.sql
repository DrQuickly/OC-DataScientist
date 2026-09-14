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
