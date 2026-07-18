/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          green: '#10b981', // Emerald 500
          dark: '#0f172a', // Slate 900
          light: '#f8fafc', // Slate 50
          amber: '#f59e0b', // Amber 500
          red: '#ef4444', // Red 500
        }
      },
      fontFamily: {
        sans: ['Outfit', 'Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
