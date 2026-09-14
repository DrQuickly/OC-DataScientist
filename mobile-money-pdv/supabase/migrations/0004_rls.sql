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
