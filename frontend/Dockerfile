# Stage 1: Build the React app
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .

# 1. Accept the build argument from Railway
ARG VITE_API_URL

# 2. Bind it to an environment variable so Vite can see it during build
ENV VITE_API_URL=$VITE_API_URL

# Now Vite will properly bake the Railway URL into the production bundle
RUN npm run build

# Stage 2: Serve the build files using Nginx
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]