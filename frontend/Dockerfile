# Use ultra-lightweight Nginx image (~20MB)
FROM nginx:alpine

# Copy static frontend files to Nginx web root
COPY . /usr/share/nginx/html

# Expose HTTP port
EXPOSE 80

# Start Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]