import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import type { Session } from '@supabase/supabase-js';
import { supabase } from '../lib/supabase';

export type Role = 'ADMIN' | 'AGENT';

export interface Profil {
  id: string;
  nom_complet: string;
  role: Role;
  actif: boolean;
}

export interface Pdv {
  id: string;
  nom: string;
}

interface ContexteAuth {
  session: Session | null;
  profil: Profil | null;
  pdvs: Pdv[]; // PDV affectés (agent) ou tous (admin)
  chargement: boolean;
  connexion: (identifiant: string, motDePasse: string) => Promise<{ erreur?: string }>;
  deconnexion: () => Promise<void>;
}

const Ctx = createContext<ContexteAuth | null>(null);

export function FournisseurAuth({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [profil, setProfil] = useState<Profil | null>(null);
  const [pdvs, setPdvs] = useState<Pdv[]>([]);
  const [chargement, setChargement] = useState(true);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      if (!data.session) setChargement(false);
    });
    const { data: sub } = supabase.auth.onAuthStateChange((_e, s) => {
      setSession(s);
      if (!s) {
        setProfil(null);
        setPdvs([]);
      }
    });
    return () => sub.subscription.unsubscribe();
  }, []);

  useEffect(() => {
    if (!session) return;
    let annule = false;
    (async () => {
      setChargement(true);
      const { data: p } = await supabase
        .from('utilisateurs')
        .select('id, nom_complet, role, actif')
        .eq('id', session.user.id)
        .single();

      let listePdv: Pdv[] = [];
      if (p?.role === 'ADMIN') {
        const { data } = await supabase.from('pdv').select('id, nom').eq('actif', true).order('nom');
        listePdv = data ?? [];
      } else {
        // Les PDV visibles sont filtrés en base par la RLS (affectations actives).
        const { data } = await supabase.from('pdv').select('id, nom').eq('actif', true).order('nom');
        listePdv = data ?? [];
      }
      if (!annule) {
        setProfil(p as Profil | null);
        setPdvs(listePdv);
        setChargement(false);
      }
    })();
    return () => {
      annule = true;
    };
  }, [session]);

  const valeur = useMemo<ContexteAuth>(
    () => ({
      session,
      profil,
      pdvs,
      chargement,
      connexion: async (identifiant, motDePasse) => {
        // Identifiant = e-mail Supabase. On accepte un identifiant simple en le
        // mappant sur un domaine interne si besoin (voir création de comptes).
        const email = identifiant.includes('@') ? identifiant : `${identifiant}@pdv.local`;
        const { error } = await supabase.auth.signInWithPassword({ email, password: motDePasse });
        return error ? { erreur: traduireErreurAuth(error.message) } : {};
      },
      deconnexion: async () => {
        await supabase.auth.signOut();
      },
    }),
    [session, profil, pdvs, chargement],
  );

  return <Ctx.Provider value={valeur}>{children}</Ctx.Provider>;
}

export function useAuth(): ContexteAuth {
  const c = useContext(Ctx);
  if (!c) throw new Error('useAuth doit être utilisé dans FournisseurAuth');
  return c;
}

function traduireErreurAuth(message: string): string {
  if (/invalid login credentials/i.test(message)) return 'Identifiant ou mot de passe incorrect.';
  if (/email not confirmed/i.test(message)) return 'Compte non confirmé. Contactez l’administrateur.';
  return 'Connexion impossible. Vérifiez votre réseau.';
}
