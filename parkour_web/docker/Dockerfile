# Serves the built page from a container (any host that runs Docker). Build from the parkour_web folder after
# `npm run build`:  docker build -f docker/Dockerfile -t beni-parkour .   The page listens on port 7860.
FROM node:22-slim
WORKDIR /app
COPY docker/server.mjs ./
COPY dist ./dist
ENV PORT=7860
EXPOSE 7860
USER node
CMD ["node", "server.mjs"]
