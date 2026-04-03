/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        modernism: '#6366f1',
        postmodernism: '#a855f7',
        critical: '#f97316',
        unknown: '#6b7280',
        bg: {
          primary: '#0f1117',
          secondary: '#1a1d27',
          tertiary: '#252836',
        },
      },
    },
  },
  plugins: [],
};
