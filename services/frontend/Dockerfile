# Frontend static site served by Nginx
FROM nginx:alpine

# Copy static files
WORKDIR /usr/share/nginx/html
COPY index.html ./

# Expose default HTTP port
EXPOSE 80

# Default Nginx command
CMD ["nginx", "-g", "daemon off;"]
