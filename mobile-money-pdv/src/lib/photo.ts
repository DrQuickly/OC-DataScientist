/**
 * Compression de la photo du solde du terminal avant stockage/envoi.
 * Objectif : rester sous ~300 ko (sobriété réseau). Redimensionne et ré-encode
 * en JPEG à qualité dégressive jusqu'à atteindre la cible.
 */
const CIBLE_OCTETS = 300 * 1024;
const LARGEUR_MAX = 1280;

export async function compresserPhoto(fichier: File | Blob): Promise<Blob> {
  const bitmap = await createImageBitmap(fichier);
  const ratio = Math.min(1, LARGEUR_MAX / bitmap.width);
  const largeur = Math.round(bitmap.width * ratio);
  const hauteur = Math.round(bitmap.height * ratio);

  const canvas = document.createElement('canvas');
  canvas.width = largeur;
  canvas.height = hauteur;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas indisponible pour la compression photo');
  ctx.drawImage(bitmap, 0, 0, largeur, hauteur);
  bitmap.close();

  let qualite = 0.8;
  let blob = await encoder(canvas, qualite);
  while (blob.size > CIBLE_OCTETS && qualite > 0.35) {
    qualite -= 0.15;
    blob = await encoder(canvas, qualite);
  }
  return blob;
}

function encoder(canvas: HTMLCanvasElement, qualite: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('Échec de l’encodage de la photo'))),
      'image/jpeg',
      qualite,
    );
  });
}
