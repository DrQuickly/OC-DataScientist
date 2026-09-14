// Edge Function : création d'un compte AGENT par un ADMIN.
//
// La création de comptes exige la clé service_role, qui ne doit JAMAIS être
// exposée au client. Elle vit ici, côté serveur uniquement.
//
// Déploiement :
//   supabase functions deploy creer-agent
//   supabase secrets set SERVICE_ROLE_KEY=... (ou via SUPABASE_SERVICE_ROLE_KEY géré par Supabase)
//
// Sécurité : on vérifie que l'appelant est bien un ADMIN actif avant d'agir.

import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const SUPABASE_URL = Deno.env.get('SUPABASE_URL')!;
const SERVICE_ROLE = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
const ANON = Deno.env.get('SUPABASE_ANON_KEY')!;

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

interface Corps {
  identifiant: string; // identifiant simple ou e-mail
  mot_de_passe: string;
  nom_complet: string;
  telephone?: string;
  pdv_ids: string[];
}

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders });

  try {
    const authHeader = req.headers.get('Authorization') ?? '';
    // Client "appelant" : sert uniquement à vérifier l'identité et le rôle.
    const clientAppelant = createClient(SUPABASE_URL, ANON, {
      global: { headers: { Authorization: authHeader } },
    });
    const { data: userData } = await clientAppelant.auth.getUser();
    if (!userData.user) return json({ erreur: 'Non authentifié.' }, 401);

    const { data: profil } = await clientAppelant
      .from('utilisateurs')
      .select('role, actif')
      .eq('id', userData.user.id)
      .single();
    if (!profil || profil.role !== 'ADMIN' || !profil.actif) {
      return json({ erreur: 'Réservé à un administrateur.' }, 403);
    }

    const corps = (await req.json()) as Corps;
    if (!corps.identifiant || !corps.mot_de_passe || !corps.nom_complet) {
      return json({ erreur: 'Champs obligatoires manquants.' }, 400);
    }
    if (corps.mot_de_passe.length < 8) {
      return json({ erreur: 'Mot de passe trop court (8 caractères minimum).' }, 400);
    }

    const email = corps.identifiant.includes('@') ? corps.identifiant : `${corps.identifiant}@pdv.local`;

    // Client "admin" : service_role, crée l'utilisateur.
    const admin = createClient(SUPABASE_URL, SERVICE_ROLE);
    const { data: created, error: errCreate } = await admin.auth.admin.createUser({
      email,
      password: corps.mot_de_passe,
      email_confirm: true,
      user_metadata: { nom_complet: corps.nom_complet },
    });
    if (errCreate || !created.user) return json({ erreur: errCreate?.message ?? 'Création impossible.' }, 400);

    const uid = created.user.id;
    const { error: errProfil } = await admin.from('utilisateurs').insert({
      id: uid,
      nom_complet: corps.nom_complet,
      telephone: corps.telephone ?? null,
      role: 'AGENT',
      actif: true,
    });
    if (errProfil) {
      // Compensation : on retire l'utilisateur auth si le profil échoue.
      await admin.auth.admin.deleteUser(uid);
      return json({ erreur: errProfil.message }, 400);
    }

    if (corps.pdv_ids?.length) {
      const { error: errAff } = await admin.from('affectations').insert(
        corps.pdv_ids.map((pdv_id) => ({ utilisateur_id: uid, pdv_id })),
      );
      if (errAff) return json({ erreur: `Compte créé mais affectation échouée : ${errAff.message}` }, 400);
    }

    return json({ ok: true, id: uid, identifiant: email });
  } catch (e) {
    return json({ erreur: e instanceof Error ? e.message : 'Erreur serveur.' }, 500);
  }
});

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}
