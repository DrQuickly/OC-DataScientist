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
