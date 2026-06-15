/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{vue,js}'],
  important: '#app',
  theme: {
    extend: {
      colors: {
        primary: { DEFAULT: '#3B82F6', dark: '#2563EB' }
      }
    }
  },
  plugins: []
}
