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
