/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Couleur principale (mobile money, sobre)
        marque: {
          DEFAULT: '#0f766e',
          fonce: '#115e59',
        },
      },
    },
  },
  plugins: [],
};
