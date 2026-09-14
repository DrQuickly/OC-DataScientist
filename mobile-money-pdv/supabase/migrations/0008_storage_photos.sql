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
