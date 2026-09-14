import { createClient } from '@supabase/supabase-js';

/**
 * Client Supabase (clés PUBLIQUES uniquement, protégées par la RLS).
 * Aucune clé secrète côté client. La service_role reste dans les Edge Functions.
 */
const url = import.meta.env.VITE_SUPABASE_URL;
const anon = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!url || !anon) {
  // Message explicite plutôt qu'un échec silencieux au premier appel réseau.
  console.warn(
    'Configuration Supabase absente : renseigner VITE_SUPABASE_URL et VITE_SUPABASE_ANON_KEY dans .env',
  );
}

export const supabase = createClient(url ?? '', anon ?? '', {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: false,
  },
});

export const supabaseConfigure = Boolean(url && anon);
